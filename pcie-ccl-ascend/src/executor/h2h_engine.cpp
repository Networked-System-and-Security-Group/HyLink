#include "h2h_engine.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../memory/shared_state.hpp"
#include "../topology/numa.hpp"
#include "../topology/topology.hpp"
#include "../util/inflight_drain.hpp"
#include "../util/log.hpp"
#include "../util/prof.hpp"
#include "../util/trace_writer.hpp"
#include "host_copy.hpp"

namespace pccl {

namespace {

int h2hWorkersPerNumaFromEnv() {
  const char* e = std::getenv("PCCL_H2H_WORKERS_PER_NUMA");
  if (!e || e[0] == '\0')
    return 8;
  char* end = nullptr;
  long parsed = std::strtol(e, &end, 10);
  if (end == e || parsed <= 0)
    return 8;
  return static_cast<int>(std::max<long>(1, std::min<long>(32, parsed)));
}

size_t h2hShardBytesFromEnv() {
  const char* e = std::getenv("PCCL_H2H_SHARD_BYTES");
  if (!e || e[0] == '\0')
    return SUBCHUNK_SLICE_SIZE / 8;
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(e, &end, 10);
  if (end == e || parsed == 0)
    return SUBCHUNK_SLICE_SIZE / 8;
  return static_cast<size_t>(std::max<unsigned long long>(64 * 1024, parsed));
}

}  // namespace

H2HEngine::H2HEngine(SharedStateManager* state_mgr, const LocalTopology* topo)
    : state_mgr_(state_mgr),
      topo_(topo),
      num_numa_nodes_(std::max(1, std::min(topo ? topo->num_numa_nodes : 1, MAX_NUMA_NODES))),
      workers_per_numa_(h2hWorkersPerNumaFromEnv()),
      shard_bytes_(h2hShardBytesFromEnv()),
      stop_(false),
      initialized_(false) {}

H2HEngine::~H2HEngine() {
  shutdown();
}

int H2HEngine::initialize() {
  if (initialized_)
    return 0;
  if (!state_mgr_) {
    LOG_ERROR("H2HEngine: SharedStateManager is not initialized");
    return -1;
  }
  stop_.store(false, std::memory_order_release);
  initialized_ = true;
  LOG_INFO("H2HEngine: Initialized (lazy worker pools, numa_nodes=%d workers_per_numa=%d shard_bytes=%zu)",
           num_numa_nodes_,
           workers_per_numa_,
           shard_bytes_);
  return 0;
}

void H2HEngine::shutdown() {
  if (!initialized_)
    return;

  stop_.store(true, std::memory_order_release);
  for (int n = 0; n < num_numa_nodes_; ++n)
    pools_[n].cv.notify_all();

  for (int n = 0; n < num_numa_nodes_; ++n) {
    for (auto& worker : pools_[n].workers) {
      if (worker.joinable())
        worker.join();
    }
    pools_[n].workers.clear();
    {
      std::lock_guard<std::mutex> lock(pools_[n].mutex);
      pools_[n].queue.clear();
      pools_[n].queue_size.store(0, std::memory_order_relaxed);
    }
  }

  {
    std::lock_guard<std::mutex> lock(completions_mutex_);
    completions_.clear();
  }
  num_completions_.store(0, std::memory_order_relaxed);

  initialized_ = false;
  LOG_INFO("H2HEngine: Shutdown");
}

int H2HEngine::workersPerNuma() const {
  return workers_per_numa_;
}

void H2HEngine::spawnPool(int dst_numa) {
  if (dst_numa < 0 || dst_numa >= num_numa_nodes_)
    return;
  auto& pool = pools_[dst_numa];
  if (pool.spawned)
    return;
  std::call_once(pool.spawn_once, [&] {
    int workers = workersPerNuma();
    pool.workers.reserve(static_cast<size_t>(workers));
    for (int w = 0; w < workers; ++w)
      pool.workers.emplace_back(&H2HEngine::workerLoop, this, dst_numa, w);
    pool.spawned = true;
    LOG_INFO("H2HEngine: NUMA %d worker pool started (%d workers)", dst_numa, workers);
  });
}

int H2HEngine::submitCopy(const CopyRequest& request) {
  if (!initialized_ || stop_.load(std::memory_order_acquire))
    return -1;
  if (!request.dst || !request.src || request.bytes == 0)
    return -1;

  int dst_numa = std::max(0, std::min(request.dst_numa, num_numa_nodes_ - 1));
  spawnPool(dst_numa);

  const size_t min_shard_bytes = std::max<size_t>(64 * 1024, shard_bytes_);
  int shards_by_size = static_cast<int>((request.bytes + min_shard_bytes - 1) / min_shard_bytes);
  int num_shards = std::max(1, std::min(workers_per_numa_, shards_by_size));
  size_t shard_bytes = (request.bytes + static_cast<size_t>(num_shards) - 1) / static_cast<size_t>(num_shards);
  auto remaining = std::make_shared<std::atomic<int>>(num_shards);

  auto& pool = pools_[dst_numa];
  int enqueued = 0;
  {
    std::lock_guard<std::mutex> lock(pool.mutex);
    for (int shard = 0; shard < num_shards; ++shard) {
      size_t offset = static_cast<size_t>(shard) * shard_bytes;
      if (offset >= request.bytes)
        break;
      size_t bytes = std::min(shard_bytes, request.bytes - offset);
      WorkItem item;
      item.kind = WorkItem::Kind::CopyShard;
      item.request = request;
      item.offset = offset;
      item.bytes = bytes;
      item.remaining = remaining;
      pool.queue.push_back(item);
      ++enqueued;
    }
  }
  // Wake exactly as many idle workers as there are new items: num_shards is
  // already size-aware (min(workers_per_numa_, shards_by_size)), but
  // notify_all() previously woke the full pool (up to workers_per_numa_
  // threads, default 16) regardless of shard count. For small messages that
  // only need a couple of shards, the rest would wake, contend for
  // pool.mutex for nothing, and go back to sleep -- a thundering herd that
  // scaled with pool size, not with actual work.
  pool.queue_size.fetch_add(enqueued, std::memory_order_release);
  for (int i = 0; i < enqueued; ++i)
    pool.cv.notify_one();
  return 0;
}

int H2HEngine::submitCopyRange(const CopyRangeRequest& request) {
  if (!initialized_ || stop_.load(std::memory_order_acquire))
    return -1;
  if (!request.dst_base || !request.src_base || request.bytes == 0 || request.num_subchunks == 0)
    return -1;

  int dst_numa = std::max(0, std::min(request.dst_numa, num_numa_nodes_ - 1));
  spawnPool(dst_numa);

  int num_threads = request.threads_per_task;
  if (num_threads <= 0)
    num_threads = std::min(kMaxRangeThreadsPerTask, workers_per_numa_);
  num_threads = std::max(1, std::min(std::min(kMaxRangeThreadsPerTask, workers_per_numa_), num_threads));
  auto sync = std::make_shared<RangeSync>();
  CopyRangeRequest normalized = request;
  normalized.threads_per_task = num_threads;

  auto& pool = pools_[dst_numa];
  {
    std::lock_guard<std::mutex> lock(pool.mutex);
    for (int tid = 0; tid < num_threads; ++tid) {
      WorkItem item;
      item.kind = WorkItem::Kind::CopyRange;
      item.range = normalized;
      item.thread_idx = tid;
      item.num_threads = num_threads;
      item.range_sync = sync;
      pool.queue.push_back(item);
    }
  }
  // See submitCopy(): wake exactly num_threads workers instead of the whole
  // pool.
  pool.queue_size.fetch_add(num_threads, std::memory_order_release);
  for (int i = 0; i < num_threads; ++i)
    pool.cv.notify_one();
  return 0;
}

size_t H2HEngine::pollCompletions(const void* owner, uint64_t work_seq, std::vector<Completion>& out,
                                  size_t max_completions) {
  if (num_completions_.load(std::memory_order_acquire) == 0)
    return 0;
  size_t count = 0;
  std::lock_guard<std::mutex> lock(completions_mutex_);
  for (auto it = completions_.begin(); it != completions_.end() && count < max_completions;) {
    if (it->owner == owner && it->work_seq == work_seq) {
      out.push_back(*it);
      it = completions_.erase(it);
      ++count;
      continue;
    }
    if (it->owner == owner && it->work_seq < work_seq) {
      it = completions_.erase(it);
      continue;
    }
    ++it;
  }
  if (count > 0)
    num_completions_.fetch_sub(count, std::memory_order_relaxed);
  return count;
}

void H2HEngine::workerLoop(int dst_numa, int worker_idx) {
  inflightMarkThreadInternal();

  int physical = topo_ ? topo_->getPhysicalNUMA(dst_numa) : dst_numa;
  int pin_numa = getNumaSibling(physical);
  pinThreadToNuma(pin_numa);

  // Node-wide NUMA pinning (numa_run_on_node) still leaves the OS scheduler
  // free to place/migrate this worker across every core of pin_numa. With up
  // to kMaxWorkersPerNuma threads sharing one node, that freedom measurably
  // adds run-to-run latency jitter. Narrow the affinity mask to one specific
  // core per worker so the scheduler has nothing left to decide at each
  // wake-up. Round-robin over the node's core list; if the list can't be
  // queried (no libnuma), fall back to the node-wide pin above.
  int cpus[64];
  int num_cpus = getNumaCpus(pin_numa, cpus, static_cast<int>(sizeof(cpus) / sizeof(cpus[0])));
  if (num_cpus > 0) {
    pinThreadToCpu(cpus[worker_idx % num_cpus]);
  }

  if (trace::isActive()) {
    trace::setThreadTid(9000 + static_cast<uint32_t>(dst_numa) * 100 + static_cast<uint32_t>(worker_idx));
    char name[48];
    std::snprintf(name, sizeof(name), "H2H NUMA%d/#%d", dst_numa, worker_idx);
    trace::emitThreadName(name);
  }

  auto& pool = pools_[dst_numa];
  while (true) {
    WorkItem item;
    bool got_item = false;
    for (int spin = 0; spin < 2000; ++spin) {
      if (stop_.load(std::memory_order_relaxed))
        break;
      if (pool.queue_size.load(std::memory_order_relaxed) > 0) {
        std::unique_lock<std::mutex> lock(pool.mutex, std::try_to_lock);
        if (lock.owns_lock() && !pool.queue.empty()) {
          item = pool.queue.front();
          pool.queue.pop_front();
          pool.queue_size.fetch_sub(1, std::memory_order_relaxed);
          got_item = true;
          break;
        }
      }
#if defined(__aarch64__)
      asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
      _mm_pause();
#endif
    }

    if (!got_item) {
      std::unique_lock<std::mutex> lock(pool.mutex);
      pool.cv.wait(lock, [&] {
        return stop_.load(std::memory_order_acquire) || !pool.queue.empty();
      });
      if (pool.queue.empty()) {
        if (stop_.load(std::memory_order_acquire))
          break;
        continue;
      }
      item = pool.queue.front();
      pool.queue.pop_front();
      pool.queue_size.fetch_sub(1, std::memory_order_relaxed);
    }

    if (item.kind == WorkItem::Kind::CopyRange) {
      processRangeWorkItem(item);
      continue;
    }

    copyHost(item.request.dst + item.offset, item.request.src + item.offset, item.bytes,
             item.request.temporal_write);
    int64_t done_ts = trace::nowMicros();

    if (trace::isActive()) {
      trace::TraceArg args[] = {
          trace::TraceArg::Int("bytes", static_cast<int64_t>(item.bytes)),
          trace::TraceArg::Int("src_numa", item.request.src_numa),
          trace::TraceArg::Int("dst_numa", item.request.dst_numa),
      };
      trace::emitComplete("H2H Copy Shard", "h2h", item.request.submit_ts, done_ts - item.request.submit_ts,
                          PCCL_PROF_COLOR_COPY_OUTER, 0, args, 3);
    }

    completeWorkItem(item, done_ts);
  }

  pinThreadToNuma(-1);
}

void H2HEngine::completeWorkItem(const WorkItem& item, int64_t complete_ts) {
  if (!item.remaining || item.remaining->fetch_sub(1, std::memory_order_acq_rel) != 1)
    return;
  const auto& request = item.request;
  pushCompletion(Completion{request.owner,
                            request.work_seq,
                            request.node_idx,
                            request.subchunk_idx,
                            request.bytes,
                            request.src_numa,
                            request.dst_numa,
                            request.submit_ts,
                            complete_ts});
}

void H2HEngine::processRangeWorkItem(const WorkItem& item) {
  const auto& request = item.range;
  const int tid = item.thread_idx;
  const int num_threads = std::max(1, item.num_threads);
  const size_t align = 256;

  for (size_t sc = 0; sc < request.num_subchunks; ++sc) {
    size_t sc_offset = sc * request.subchunk_bytes;
    if (sc_offset >= request.bytes)
      break;
    size_t this_bytes = std::min(request.subchunk_bytes, request.bytes - sc_offset);

    while (state_mgr_->getWriteProgress(request.comm_id, request.input_key, request.epoch) < sc + 1) {
#if defined(__aarch64__)
      asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
      _mm_pause();
#endif
    }
    if (request.has_overwrite_guard) {
      if (request.overwrite_guard.version > 0) {
        while (state_mgr_->getWriteProgress(request.comm_id, request.overwrite_guard, request.epoch) < sc + 1) {
#if defined(__aarch64__)
          asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
          _mm_pause();
#endif
        }
      }
      if (state_mgr_->getNumReaders(request.comm_id, request.overwrite_guard) > 0) {
        while (state_mgr_->getConsumeProgress(request.comm_id, request.overwrite_guard, request.epoch) < sc + 1) {
#if defined(__aarch64__)
          asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
          _mm_pause();
#endif
        }
      }
    }

    size_t src_offset = request.input_ring_buffered ? ((sc & 1) * request.subchunk_bytes) : sc_offset;
    size_t dst_offset = request.output_ring_buffered ? ((sc & 1) * request.subchunk_bytes) : sc_offset;
    size_t per_thread = ((this_bytes + static_cast<size_t>(num_threads) - 1) /
                         static_cast<size_t>(num_threads) + align - 1) &
                        ~(align - 1);
    size_t my_start = static_cast<size_t>(tid) * per_thread;
    size_t my_end = std::min(my_start + per_thread, this_bytes);
    int64_t t0 = trace::nowMicros();
    if (my_start < my_end) {
      copyHost(request.dst_base + dst_offset + my_start, request.src_base + src_offset + my_start,
               my_end - my_start, request.temporal_write);
    }
    int64_t t1 = trace::nowMicros();

    if (trace::isActive() && my_start < my_end) {
      trace::TraceArg args[] = {
          trace::TraceArg::Int("bytes", static_cast<int64_t>(my_end - my_start)),
          trace::TraceArg::Int("src_numa", request.src_numa),
          trace::TraceArg::Int("dst_numa", request.dst_numa),
          trace::TraceArg::Int("thread", tid),
      };
      trace::emitComplete("H2H Range Shard", "h2h", t0, t1 - t0, PCCL_PROF_COLOR_COPY_OUTER, 0, args, 4);
    }

    const uint64_t generation =
        item.range_sync->barrier_generation.load(std::memory_order_acquire);
    int finished = item.range_sync->barrier_count.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (finished == num_threads) {
      item.range_sync->barrier_count.store(0, std::memory_order_relaxed);
      std::atomic_thread_fence(std::memory_order_seq_cst);
      state_mgr_->advanceWriteProgress(request.comm_id, request.output_key, sc + 1, request.epoch);
      if (request.input_reader_id >= 0)
        state_mgr_->advanceReaderDone(request.comm_id, request.input_key, request.input_reader_id,
                                      sc + 1, request.epoch);
      item.range_sync->barrier_generation.store(generation + 1, std::memory_order_release);
    } else {
      while (item.range_sync->barrier_generation.load(std::memory_order_acquire) == generation) {
#if defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
        _mm_pause();
#endif
      }
    }
  }

  int done = item.range_sync->threads_done.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (done == num_threads) {
    int64_t complete_ts = trace::nowMicros();
    pushCompletion(Completion{request.owner,
                              request.work_seq,
                              request.node_idx,
                              kRangeCompletionSubchunk,
                              request.bytes,
                              request.src_numa,
                              request.dst_numa,
                              request.submit_ts,
                              complete_ts});
  }
}

void H2HEngine::pushCompletion(const Completion& completion) {
  {
    std::lock_guard<std::mutex> lock(completions_mutex_);
    completions_.push_back(completion);
    // Publish the nonempty hint before releasing the visible queue. Otherwise
    // a descheduled producer can leave a completed copy invisible to polling.
    num_completions_.fetch_add(1, std::memory_order_release);
  }
}

}  // namespace pccl
