#include "h2h_engine.hpp"

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <algorithm>
#include <cstdio>
#include <ctime>
#include <sched.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <thread>

static inline int getCurrentCpu() {
  unsigned cpu = 0;
#if defined(__linux__)
  syscall(SYS_getcpu, &cpu, nullptr, nullptr);
#endif
  return static_cast<int>(cpu);
}

#include "../hal/host_rt.hpp"
#include "../memory/shared_state.hpp"
#include "../topology/numa.hpp"
#include "../topology/topology.hpp"
#include "../util/inflight_drain.hpp"
#include "../util/log.hpp"
#include "../util/prof.hpp"
#include "../util/trace_writer.hpp"

namespace pccl {

namespace {

template <typename T>
T reduceValue(T a, T b, ReduceOpKind op) {
  switch (op) {
    case ReduceOpKind::Sum:
      return static_cast<T>(a + b);
    case ReduceOpKind::Max:
      return a > b ? a : b;
    case ReduceOpKind::Min:
      return a < b ? a : b;
  }
  return static_cast<T>(a + b);
}

template <typename T>
void reduceTyped(void* dst, const void* src1, const void* src2, size_t count, ReduceOpKind op) {
  auto* d = static_cast<T*>(dst);
  const auto* a = static_cast<const T*>(src1);
  const auto* b = static_cast<const T*>(src2);
  for (size_t i = 0; i < count; ++i)
    d[i] = reduceValue<T>(a[i], b[i], op);
}

void reduceHostBuffer(void* dst, const void* src1, const void* src2, size_t bytes, ScalarDataType type,
                      ReduceOpKind op) {
  switch (type) {
    case ScalarDataType::Int8:
      reduceTyped<int8_t>(dst, src1, src2, bytes / sizeof(int8_t), op);
      break;
    case ScalarDataType::Uint8:
      reduceTyped<uint8_t>(dst, src1, src2, bytes / sizeof(uint8_t), op);
      break;
    case ScalarDataType::Int32:
      reduceTyped<int32_t>(dst, src1, src2, bytes / sizeof(int32_t), op);
      break;
    case ScalarDataType::Uint32:
      reduceTyped<uint32_t>(dst, src1, src2, bytes / sizeof(uint32_t), op);
      break;
    case ScalarDataType::Int64:
      reduceTyped<int64_t>(dst, src1, src2, bytes / sizeof(int64_t), op);
      break;
    case ScalarDataType::Uint64:
      reduceTyped<uint64_t>(dst, src1, src2, bytes / sizeof(uint64_t), op);
      break;
    case ScalarDataType::Float32:
      reduceTyped<float>(dst, src1, src2, bytes / sizeof(float), op);
      break;
    case ScalarDataType::Float64:
      reduceTyped<double>(dst, src1, src2, bytes / sizeof(double), op);
      break;
    case ScalarDataType::Float16:
    case ScalarDataType::Bfloat16:
      LOG_ERROR("H2HEngine: host reduction does not support 16-bit floating point datatype");
      break;
  }
}

size_t scalarElementSize(ScalarDataType type) {
  switch (type) {
    case ScalarDataType::Int8:
    case ScalarDataType::Uint8:
      return 1;
    case ScalarDataType::Float16:
    case ScalarDataType::Bfloat16:
      return 2;
    case ScalarDataType::Int32:
    case ScalarDataType::Uint32:
    case ScalarDataType::Float32:
      return 4;
    case ScalarDataType::Int64:
    case ScalarDataType::Uint64:
    case ScalarDataType::Float64:
      return 8;
  }
  return 1;
}

}  // namespace

H2HEngine::H2HEngine(SharedStateManager* state_mgr, const LocalTopology* topo)
    : state_mgr_(state_mgr),
      topo_(topo),
      num_numa_nodes_(std::max(1, std::min(topo ? topo->num_numa_nodes : 1, static_cast<int>(MAX_NUMA_NODES)))),
      stop_(false),
      pending_count_(0),
      initialized_(false) {}

H2HEngine::~H2HEngine() {
  shutdown();
}

int H2HEngine::initialize() {
  if (initialized_)
    return 0;

  stop_.store(false);
  // Workers are created lazily on first graph work submission per NUMA pool.

  initialized_ = true;
  LOG_INFO("H2HEngine: Initialized (lazy worker creation, %d NUMA nodes)", num_numa_nodes_);
  return 0;
}

void H2HEngine::shutdown() {
  if (!initialized_)
    return;
  stop_.store(true, std::memory_order_release);

  for (int n = 0; n < num_numa_nodes_; n++) {
    numa_pools_[n].cv.notify_all();
  }

  for (int n = 0; n < num_numa_nodes_; n++) {
    for (auto& w : numa_pools_[n].workers) {
      if (w.joinable())
        w.join();
    }
    numa_pools_[n].workers.clear();
  }

  initialized_ = false;
  LOG_INFO("H2HEngine: Shutdown");
}

static char* resolveGraphHostPtr(const RuntimeBinding& binding, const ProgressKey& key, size_t subchunk_idx,
                                 size_t chunk_size, bool ring_buffered) {
  const size_t SUBCHUNK_SLICE_SIZE = pcclSubchunkSliceFor(chunk_size);
  char* base = binding.host_base[key.owner];
  if (!base)
    return nullptr;
  size_t chunk_offset = static_cast<size_t>(key.chunk) * chunk_size;
  size_t sub_offset = ring_buffered ? ((subchunk_idx % pcclRingSlotsFor(chunk_size)) * SUBCHUNK_SLICE_SIZE)
                                    : (subchunk_idx * SUBCHUNK_SLICE_SIZE);
  return base + chunk_offset + sub_offset;
}

int H2HEngine::submitGraphWork(const InstructionNode& node, const RuntimeBinding& binding, size_t subchunk_idx,
                               size_t chunk_size, size_t bytes, int epoch,
                               std::shared_ptr<std::atomic<int>> done_counter) {
  if (node.op != OpCode::H2H && node.op != OpCode::H2HReduce)
    return 0;
  if (node.inputs.empty())
    return 0;

  int dst_numa = std::max(0, std::min(node.output.owner, num_numa_nodes_ - 1));
  if (!numa_pools_[dst_numa].spawned) {
    auto& pool = numa_pools_[dst_numa];
    std::call_once(pool.spawn_once, [&] {
      pool.workers.reserve(MAX_H2H_WORKERS_PER_NUMA);
      for (int w = 0; w < MAX_H2H_WORKERS_PER_NUMA; w++)
        pool.workers.emplace_back(&H2HEngine::workerLoop, this, dst_numa, w);
      pool.spawned = true;
      LOG_INFO("H2HEngine: NUMA %d pool started (%d workers, lazy)", dst_numa, MAX_H2H_WORKERS_PER_NUMA);
    });
  }

  char* dst = resolveGraphHostPtr(binding, node.output, subchunk_idx, chunk_size, node.output_ring_buffered);
  const char* src1 = resolveGraphHostPtr(binding, node.inputs[0].key, subchunk_idx, chunk_size,
                                         node.inputs[0].ring_buffered);
  const char* src2 = nullptr;
  if (node.op == OpCode::H2HReduce && node.inputs.size() > 1) {
    src2 = resolveGraphHostPtr(binding, node.inputs[1].key, subchunk_idx, chunk_size, node.inputs[1].ring_buffered);
  }
  if (!dst || !src1 || (node.op == OpCode::H2HReduce && !src2))
    return 0;
  int num_threads = std::max(1, std::min(MAX_H2H_THREADS_PER_TASK, MAX_H2H_WORKERS_PER_NUMA));

  auto& pool = numa_pools_[dst_numa];
  {
    std::lock_guard<std::mutex> lock(pool.mutex);
    for (int tid = 0; tid < num_threads; ++tid) {
      pool.queue.push_back({dst, src1, src2,
                            node.inputs[0].key.owner,
                            (node.op == OpCode::H2HReduce && node.inputs.size() > 1) ? node.inputs[1].key.owner : -1,
                            dst_numa, node.output.chunk,
                            node.op == OpCode::H2H, binding.scalar_type, binding.reduce_op, binding.element_size,
                            tid, num_threads, bytes, done_counter});
    }
  }
  pending_count_.fetch_add(num_threads, std::memory_order_relaxed);
  pool.cv.notify_all();
  return num_threads;
}

// ── Per-NUMA resident worker loop ───────────────────────────────────

void H2HEngine::workerLoop(int numa_node, int worker_idx) {
  inflightMarkThreadInternal();
  // Pin BEFORE any work — first-touch guarantee.
  int physical = topo_ ? topo_->getPhysicalNUMA(numa_node) : numa_node;
  int pin_numa = (worker_idx < H2H_SIBLING_WORKERS)
                     ? getNumaSibling(physical)
                     : physical;
  pinThreadToNuma(pin_numa);

  if (trace::isActive()) {
    trace::setThreadTid(trace::h2hWorkerTid(numa_node, worker_idx));
    char name[48];
    std::snprintf(name, sizeof(name), "H2H NUMA%d/#%d", numa_node, worker_idx);
    trace::emitThreadName(name);
  }

  auto& pool = numa_pools_[numa_node];

  while (!stop_.load(std::memory_order_acquire)) {
    H2HWorkItem item;
    bool got = false;
    {
      std::unique_lock<std::mutex> lock(pool.mutex);
      pool.cv.wait(lock, [&] { return !pool.queue.empty() || stop_.load(std::memory_order_relaxed); });
      if (stop_.load(std::memory_order_relaxed))
        break;
      item = pool.queue.front();
      pool.queue.pop_front();
      got = true;
    }

    if (!got)
      continue;

    int64_t wg_t0 = trace::nowMicros();
    processGraphWork(item);
    int64_t wg_t1 = trace::nowMicros();

    if (trace::isActive()) {
      char wg_label[80];
      if (item.is_copy) {
        std::snprintf(wg_label, sizeof(wg_label),
                      "H2H Copy NUMA%d→%d c=%d #%d",
                      item.src1_numa, item.dst_numa,
                      item.dst_chunk_idx, item.tid);
      } else {
        std::snprintf(wg_label, sizeof(wg_label),
                      "H2H Reduce NUMA(%d,%d)→%d c=%d #%d",
                      item.src1_numa, item.src2_numa, item.dst_numa,
                      item.dst_chunk_idx, item.tid);
      }
      uint32_t wg_color = item.is_copy ? PCCL_PROF_COLOR_COPY_OUTER
                                       : PCCL_PROF_COLOR_REDUCE_OUTER;
      trace::emitComplete(wg_label, "h2h", wg_t0, wg_t1 - wg_t0, wg_color);
    }
    pending_count_.fetch_sub(1, std::memory_order_release);
  }

  pinThreadToNuma(-1);
}

// ── Graph subchunk processing ───────────────────────────────────────

void H2HEngine::processGraphWork(const H2HWorkItem& item) {
  size_t unit = item.is_copy ? 1 : std::max<size_t>(1, item.element_size ? item.element_size
                                                                         : scalarElementSize(item.scalar_type));
  size_t units = item.graph_bytes / unit;
  size_t per_thread_units = (units + item.num_threads - 1) / item.num_threads;
  size_t my_start = std::min(static_cast<size_t>(item.tid) * per_thread_units * unit, item.graph_bytes);
  size_t my_end = std::min(my_start + per_thread_units * unit, item.graph_bytes);
  if (my_start < my_end) {
    size_t my_bytes = my_end - my_start;
    if (item.is_copy) {
      hostMemcpy(item.dst + my_start, item.src1 + my_start, my_bytes);
    } else {
      reduceHostBuffer(item.dst + my_start, item.src1 + my_start, item.src2 + my_start, my_bytes,
                       item.scalar_type, item.reduce_op);
    }
  }
  if (item.graph_done)
    item.graph_done->fetch_add(1, std::memory_order_release);
}

}  // namespace pccl
