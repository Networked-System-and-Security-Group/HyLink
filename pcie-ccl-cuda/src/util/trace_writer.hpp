#ifndef PCCL_TRACE_WRITER_HPP
#define PCCL_TRACE_WRITER_HPP

// ─────────────────────────────────────────────────────────────────────────────
// Minimal Chrome Trace Event Format (JSON) writer for Perfetto
//
// Header-only, C++14, zero external dependencies.
// Thread-safe via thread-local event shards (lock-free hot path).
//
// Usage:
//   trace::initialize("/path/to/trace.json");  // or from PCCL_TRACE_FILE env
//   trace::pushRange("MyRange", 0xFF0077BB);
//   trace::popRange();
//   trace::emitComplete("DMA", "gpu", t0, dur, color);
//   trace::finalize();  // also registered via atexit
//
// Output is loadable by https://ui.perfetto.dev or chrome://tracing
// ─────────────────────────────────────────────────────────────────────────────

#if defined(PCCL_TRACE_ENABLED) && PCCL_TRACE_ENABLED

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <process.h>
#define PCCL_GETPID() _getpid()
#else
#include <unistd.h>
#define PCCL_GETPID() static_cast<uint32_t>(::getpid())
#endif

namespace pccl {
namespace trace {

// ── Trace event struct ───────────────────────────────────────────────────────

struct TraceArg {
  const char* key;
  enum Kind { kDouble, kInt64, kString } kind;
  union {
    double d;
    int64_t i;
  };
  char s_buf[32];  // inline copy for string values

  static TraceArg Double(const char* k, double v) {
    TraceArg a;
    a.key = k;
    a.kind = kDouble;
    a.d = v;
    a.s_buf[0] = '\0';
    return a;
  }
  static TraceArg Int(const char* k, int64_t v) {
    TraceArg a;
    a.key = k;
    a.kind = kInt64;
    a.i = v;
    a.s_buf[0] = '\0';
    return a;
  }
  static TraceArg Str(const char* k, const char* v) {
    TraceArg a;
    a.key = k;
    a.kind = kString;
    a.i = 0;
    if (v) {
      std::strncpy(a.s_buf, v, sizeof(a.s_buf) - 1);
      a.s_buf[sizeof(a.s_buf) - 1] = '\0';
    } else {
      a.s_buf[0] = '\0';
    }
    return a;
  }
};

static constexpr int MAX_TRACE_ARGS = 12;

struct TraceEvent {
  char name[128];
  const char* category;      // string literal
  char ph;                   // 'B','E','X','s','f','C'
  int64_t ts;                // microseconds since trace epoch
  int64_t dur;               // for ph='X' only
  uint32_t pid;
  uint32_t tid;
  uint32_t color;            // ARGB
  uint32_t flow_id;          // for flow events (ph='s'/'f')
  TraceArg args[MAX_TRACE_ARGS];
  int num_args;
};

// ── Thread-local shard ───────────────────────────────────────────────────────

struct ThreadShard {
  std::vector<TraceEvent> events;
  int64_t range_stack[32];   // begin timestamps for nested push/pop
  int range_depth;
  uint32_t tid;

  ThreadShard() : range_depth(0), tid(0) {
    events.reserve(4096);
  }
};

// ── Global state ─────────────────────────────────────────────────────────────

struct GlobalState {
  std::mutex mutex;
  std::vector<ThreadShard*> shards;
  char output_path[512];
  std::chrono::steady_clock::time_point epoch;
  int64_t wall_epoch_us;  // wall-clock µs at epoch (for msprof alignment)
  uint32_t pid;
  std::atomic<bool> active{false};
  std::atomic<bool> finalized{false};
  std::atomic<uint32_t> flow_counter{0};
  std::atomic<uint32_t> next_tid{1};  // sequential tid allocator (0 reserved)
};

inline GlobalState& global() {
  static GlobalState g;
  return g;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

inline int64_t nowMicros() {
  auto& g = global();
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(now - g.epoch).count();
}

inline uint32_t allocateTid() {
  return global().next_tid.fetch_add(1, std::memory_order_relaxed);
}

inline uint32_t nextFlowId() {
  return global().flow_counter.fetch_add(1, std::memory_order_relaxed);
}

// ── Thread-local shard access ────────────────────────────────────────────────
// Shards are heap-allocated and owned by GlobalState so they outlive their
// threads (thread_local destructors fire before atexit → dangling pointer).

inline ThreadShard& localShard() {
  thread_local ThreadShard* shard = nullptr;
  if (!shard) {
    shard = new ThreadShard();
    shard->tid = allocateTid();
    auto& g = global();
    std::lock_guard<std::mutex> lock(g.mutex);
    g.shards.push_back(shard);
  }
  return *shard;
}

// Override this thread's tid to a fixed value for deterministic track ordering.
// Must be called before any events are emitted on this thread.
inline void setThreadTid(uint32_t tid) {
  localShard().tid = tid;
}

// ── Deterministic TID scheme ─────────────────────────────────────────────────
// Track ordering in Perfetto follows tid order. Assign fixed tids so tracks
// appear in a logical grouping:
//
//   Rank 0:  D2H thread=100, D2H Stream=110, H2D thread=120, H2D Stream=130
//   Rank 1:  D2H thread=200, D2H Stream=210, H2D thread=220, H2D Stream=230
//   ...
//   H2H NUMA0/#0=1001, /#1=1002, ...
//   H2H NUMA1/#0=1101, /#1=1102, ...
//
inline uint32_t rankD2HTid(int rank)                  { return 100 + rank * 100; }
inline uint32_t rankH2DTid(int rank)                  { return 120 + rank * 100; }
inline uint32_t h2hWorkerTid(int numa, int worker)    { return 1001 + numa * 100 + worker; }

// ── Public API ───────────────────────────────────────────────────────────────

// Forward-declare so atexit lambda in initialize() can reference them
inline void finalize();
inline void finalizeAtExit();

inline void initialize(const char* output_path) {
  auto& g = global();
  if (!output_path || output_path[0] == '\0')
    return;
  std::lock_guard<std::mutex> lock(g.mutex);
  if (g.active.load(std::memory_order_relaxed))
    return;  // already initialized
  std::strncpy(g.output_path, output_path, sizeof(g.output_path) - 1);
  g.output_path[sizeof(g.output_path) - 1] = '\0';
  g.epoch = std::chrono::steady_clock::now();
  g.wall_epoch_us = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  g.pid = PCCL_GETPID();
  g.finalized.store(false, std::memory_order_relaxed);
  g.active.store(true, std::memory_order_release);

  // Safety net — write trace on process exit if not already done.
  // Uses finalizeAtExit (no shard deletion) since threads may still be live.
  std::atexit([]() { finalizeAtExit(); });
}

inline bool isActive() {
  return global().active.load(std::memory_order_acquire);
}

inline void pushRange(const char* name, uint32_t color) {
  if (!isActive())
    return;
  auto& shard = localShard();
  int64_t ts = nowMicros();

  // Push begin timestamp onto stack
  if (shard.range_depth < 32) {
    shard.range_stack[shard.range_depth] = ts;
  }
  shard.range_depth++;

  TraceEvent ev;
  std::strncpy(ev.name, name, sizeof(ev.name) - 1);
  ev.name[sizeof(ev.name) - 1] = '\0';
  ev.category = "pccl";
  ev.ph = 'B';
  ev.ts = ts;
  ev.dur = 0;
  ev.pid = global().pid;
  ev.tid = shard.tid;
  ev.color = color;
  ev.flow_id = 0;
  ev.num_args = 0;
  shard.events.push_back(ev);
}

inline void popRange() {
  if (!isActive())
    return;
  auto& shard = localShard();
  if (shard.range_depth <= 0)
    return;
  shard.range_depth--;

  TraceEvent ev;
  ev.name[0] = '\0';
  ev.category = "pccl";
  ev.ph = 'E';
  ev.ts = nowMicros();
  ev.dur = 0;
  ev.pid = global().pid;
  ev.tid = shard.tid;
  ev.color = 0;
  ev.flow_id = 0;
  ev.num_args = 0;
  shard.events.push_back(ev);
}

inline void emitComplete(const char* name, const char* cat,
                         int64_t begin_us, int64_t dur_us,
                         uint32_t color, uint32_t tid_override = 0,
                         const TraceArg* args = nullptr, int num_args = 0) {
  if (!isActive())
    return;
  auto& shard = localShard();

  TraceEvent ev;
  std::strncpy(ev.name, name, sizeof(ev.name) - 1);
  ev.name[sizeof(ev.name) - 1] = '\0';
  ev.category = cat;
  ev.ph = 'X';
  ev.ts = begin_us;
  ev.dur = dur_us;
  ev.pid = global().pid;
  ev.tid = tid_override ? tid_override : shard.tid;
  ev.color = color;
  ev.flow_id = 0;
  int n = (num_args < MAX_TRACE_ARGS) ? num_args : MAX_TRACE_ARGS;
  for (int i = 0; i < n; i++)
    ev.args[i] = args[i];
  ev.num_args = n;
  shard.events.push_back(ev);
}

inline void emitFlowStart(const char* name, const char* cat,
                          int64_t ts, uint32_t flow_id, uint32_t color = 0) {
  if (!isActive())
    return;
  auto& shard = localShard();
  TraceEvent ev;
  std::strncpy(ev.name, name, sizeof(ev.name) - 1);
  ev.name[sizeof(ev.name) - 1] = '\0';
  ev.category = cat;
  ev.ph = 's';
  ev.ts = ts;
  ev.dur = 0;
  ev.pid = global().pid;
  ev.tid = shard.tid;
  ev.color = color;
  ev.flow_id = flow_id;
  ev.num_args = 0;
  shard.events.push_back(ev);
}

inline void emitFlowFinish(const char* name, const char* cat,
                           int64_t ts, uint32_t flow_id, uint32_t color = 0) {
  if (!isActive())
    return;
  auto& shard = localShard();
  TraceEvent ev;
  std::strncpy(ev.name, name, sizeof(ev.name) - 1);
  ev.name[sizeof(ev.name) - 1] = '\0';
  ev.category = cat;
  ev.ph = 'f';
  ev.ts = ts;
  ev.dur = 0;
  ev.pid = global().pid;
  ev.tid = shard.tid;
  ev.color = color;
  ev.flow_id = flow_id;
  ev.num_args = 0;
  shard.events.push_back(ev);
}

// Emit thread name metadata so Perfetto shows readable track names
inline void emitThreadName(const char* name, uint32_t tid_override = 0) {
  if (!isActive())
    return;
  auto& shard = localShard();
  TraceEvent ev;
  std::strncpy(ev.name, "thread_name", sizeof(ev.name) - 1);
  ev.name[sizeof(ev.name) - 1] = '\0';
  ev.category = "__metadata";
  ev.ph = 'M';  // metadata event
  ev.ts = 0;
  ev.dur = 0;
  ev.pid = global().pid;
  ev.tid = tid_override ? tid_override : shard.tid;
  ev.color = 0;
  ev.flow_id = 0;
  ev.args[0] = TraceArg::Str("name", name);
  ev.num_args = 1;
  shard.events.push_back(ev);
}

inline void finalize() {
  auto& g = global();
  if (!g.active.load(std::memory_order_acquire))
    return;
  bool expected = false;
  if (!g.finalized.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
    return;  // already finalized (e.g. atexit raced)
  g.active.store(false, std::memory_order_release);

  FILE* fp = std::fopen(g.output_path, "w");
  if (!fp)
    return;
  std::fprintf(fp, "{\"traceEvents\":[\n");
  bool first = true;
  std::lock_guard<std::mutex> lock(g.mutex);
  for (auto* shard : g.shards) {
    for (auto& ev : shard->events) {
      if (!first)
        std::fprintf(fp, ",\n");
      first = false;

      // Metadata events have different format
      if (ev.ph == 'M') {
        std::fprintf(fp, "{\"name\":\"%s\",\"ph\":\"M\",\"pid\":%u,\"tid\":%u",
                     ev.name, ev.pid, ev.tid);
        if (ev.num_args > 0) {
          std::fprintf(fp, ",\"args\":{");
          for (int a = 0; a < ev.num_args; a++) {
            if (a > 0) std::fprintf(fp, ",");
            auto& arg = ev.args[a];
            if (arg.kind == TraceArg::kString)
              std::fprintf(fp, "\"%s\":\"%s\"", arg.key, arg.s_buf);
            else if (arg.kind == TraceArg::kInt64)
              std::fprintf(fp, "\"%s\":%" PRId64, arg.key, arg.i);
            else
              std::fprintf(fp, "\"%s\":%.4f", arg.key, arg.d);
          }
          std::fprintf(fp, "}");
        }
        std::fprintf(fp, "}");
        continue;
      }

      std::fprintf(fp, "{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"%c\","
                   "\"ts\":%" PRId64 ",\"pid\":%u,\"tid\":%u",
                   ev.name, ev.category ? ev.category : "pccl",
                   ev.ph, ev.ts, ev.pid, ev.tid);

      if (ev.ph == 'X')
        std::fprintf(fp, ",\"dur\":%" PRId64, ev.dur);
      if (ev.ph == 's' || ev.ph == 'f') {
        std::fprintf(fp, ",\"id\":%u", ev.flow_id);
        if (ev.ph == 'f')
          std::fprintf(fp, ",\"bp\":\"e\"");
      }

      bool has_args = (ev.num_args > 0) || (ev.color != 0);
      if (has_args) {
        std::fprintf(fp, ",\"args\":{");
        bool first_arg = true;
        if (ev.color != 0) {
          std::fprintf(fp, "\"color\":\"#%02X%02X%02X\"",
                       (ev.color >> 16) & 0xFF,
                       (ev.color >> 8) & 0xFF,
                       ev.color & 0xFF);
          first_arg = false;
        }
        for (int a = 0; a < ev.num_args; a++) {
          if (!first_arg)
            std::fprintf(fp, ",");
          first_arg = false;
          auto& arg = ev.args[a];
          switch (arg.kind) {
            case TraceArg::kDouble:
              std::fprintf(fp, "\"%s\":%.4f", arg.key, arg.d);
              break;
            case TraceArg::kInt64:
              std::fprintf(fp, "\"%s\":%" PRId64, arg.key, arg.i);
              break;
            case TraceArg::kString:
              std::fprintf(fp, "\"%s\":\"%s\"", arg.key, arg.s_buf);
              break;
          }
        }
        std::fprintf(fp, "}");
      }
      std::fprintf(fp, "}");
    }
  }
  std::fprintf(fp, "\n],\"wall_epoch_us\":%" PRId64 "}\n", g.wall_epoch_us);
  std::fclose(fp);

  // Free heap-allocated shards.
  // Safe because all producer threads must be joined before calling finalize().
  for (auto* shard : g.shards)
    delete shard;
  g.shards.clear();
}

// Variant that skips shard deletion — used by atexit where threads may
// still be live.  Leaking memory at process exit is harmless.
inline void finalizeAtExit() {
  auto& g = global();
  if (!g.active.load(std::memory_order_acquire))
    return;
  bool expected = false;
  if (!g.finalized.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
    return;
  g.active.store(false, std::memory_order_release);

  FILE* fp = std::fopen(g.output_path, "w");
  if (!fp)
    return;
  std::fprintf(fp, "{\"traceEvents\":[\n");
  bool first = true;
  std::lock_guard<std::mutex> lock(g.mutex);
  for (auto* shard : g.shards) {
    for (auto& ev : shard->events) {
      if (!first)
        std::fprintf(fp, ",\n");
      first = false;

      if (ev.ph == 'M') {
        std::fprintf(fp, "{\"name\":\"%s\",\"ph\":\"M\",\"pid\":%u,\"tid\":%u",
                     ev.name, ev.pid, ev.tid);
        if (ev.num_args > 0) {
          std::fprintf(fp, ",\"args\":{");
          for (int a = 0; a < ev.num_args; a++) {
            if (a > 0) std::fprintf(fp, ",");
            auto& arg = ev.args[a];
            if (arg.kind == TraceArg::kString)
              std::fprintf(fp, "\"%s\":\"%s\"", arg.key, arg.s_buf);
            else if (arg.kind == TraceArg::kInt64)
              std::fprintf(fp, "\"%s\":%" PRId64, arg.key, arg.i);
            else
              std::fprintf(fp, "\"%s\":%.4f", arg.key, arg.d);
          }
          std::fprintf(fp, "}");
        }
        std::fprintf(fp, "}");
        continue;
      }

      std::fprintf(fp, "{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"%c\","
                   "\"ts\":%" PRId64 ",\"pid\":%u,\"tid\":%u",
                   ev.name, ev.category ? ev.category : "pccl",
                   ev.ph, ev.ts, ev.pid, ev.tid);
      if (ev.ph == 'X')
        std::fprintf(fp, ",\"dur\":%" PRId64, ev.dur);
      if (ev.ph == 's' || ev.ph == 'f') {
        std::fprintf(fp, ",\"id\":%u", ev.flow_id);
        if (ev.ph == 'f')
          std::fprintf(fp, ",\"bp\":\"e\"");
      }
      bool has_args = (ev.num_args > 0) || (ev.color != 0);
      if (has_args) {
        std::fprintf(fp, ",\"args\":{");
        bool first_arg = true;
        if (ev.color != 0) {
          std::fprintf(fp, "\"color\":\"#%02X%02X%02X\"",
                       (ev.color >> 16) & 0xFF,
                       (ev.color >> 8) & 0xFF,
                       ev.color & 0xFF);
          first_arg = false;
        }
        for (int a = 0; a < ev.num_args; a++) {
          if (!first_arg)
            std::fprintf(fp, ",");
          first_arg = false;
          auto& arg = ev.args[a];
          switch (arg.kind) {
            case TraceArg::kDouble:
              std::fprintf(fp, "\"%s\":%.4f", arg.key, arg.d);
              break;
            case TraceArg::kInt64:
              std::fprintf(fp, "\"%s\":%" PRId64, arg.key, arg.i);
              break;
            case TraceArg::kString:
              std::fprintf(fp, "\"%s\":\"%s\"", arg.key, arg.s_buf);
              break;
          }
        }
        std::fprintf(fp, "}");
      }
      std::fprintf(fp, "}");
    }
  }
  std::fprintf(fp, "\n],\"wall_epoch_us\":%" PRId64 "}\n", g.wall_epoch_us);
  std::fclose(fp);
  // Do NOT delete shards — threads may still hold pointers. Process is exiting anyway.
}

// ── Virtual TID for GPU stream tracks (uses deterministic scheme above) ──────
inline uint32_t gpuStreamTid(int rank, int executor_type, int stream_idx) {
  uint32_t base = (executor_type == 0) ? rankD2HTid(rank) : rankH2DTid(rank);
  return base + 10 + static_cast<uint32_t>(stream_idx);
}

}  // namespace trace
}  // namespace pccl

#else  // PCCL_TRACE_ENABLED not defined

#include <cstdint>
#include <cstring>

// Stub namespace so conditional code can reference trace:: without #ifdefs everywhere
namespace pccl {
namespace trace {

struct TraceArg {
  const char* key;
  enum Kind { kDouble, kInt64, kString } kind;
  union { double d; int64_t i; };
  char s_buf[32];
  static TraceArg Double(const char* k, double v) { TraceArg a; a.key=k; a.kind=kDouble; a.d=v; a.s_buf[0]='\0'; return a; }
  static TraceArg Int(const char* k, int64_t v) { TraceArg a; a.key=k; a.kind=kInt64; a.i=v; a.s_buf[0]='\0'; return a; }
  static TraceArg Str(const char* k, const char* v) { TraceArg a; a.key=k; a.kind=kString; a.i=0; if(v){std::strncpy(a.s_buf,v,31);a.s_buf[31]='\0';}else{a.s_buf[0]='\0';} return a; }
};

inline void initialize(const char*) {}
inline bool isActive() { return false; }
inline int64_t nowMicros() { return 0; }
inline uint32_t nextFlowId() { return 0; }
inline void pushRange(const char*, uint32_t) {}
inline void popRange() {}
inline void emitComplete(const char*, const char*, int64_t, int64_t,
                         uint32_t, uint32_t = 0,
                         const TraceArg* = nullptr, int = 0) {}
inline void emitFlowStart(const char*, const char*, int64_t, uint32_t, uint32_t = 0) {}
inline void emitFlowFinish(const char*, const char*, int64_t, uint32_t, uint32_t = 0) {}
inline void emitThreadName(const char*, uint32_t = 0) {}
inline void finalize() {}
inline void finalizeAtExit() {}
inline void setThreadTid(uint32_t) {}
inline uint32_t gpuStreamTid(int, int, int) { return 0; }
inline uint32_t rankD2HTid(int) { return 0; }
inline uint32_t rankH2DTid(int) { return 0; }
inline uint32_t h2hWorkerTid(int, int) { return 0; }
}  // namespace trace
}  // namespace pccl

#endif  // PCCL_TRACE_ENABLED

#endif  // PCCL_TRACE_WRITER_HPP
