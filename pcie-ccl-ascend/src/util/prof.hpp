#ifndef PCCL_PROF_HPP
#define PCCL_PROF_HPP

#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// Unified profiling interface for PCCL
//
// Provides:
//   PcclProfRangeGuard(msg [, color])  — RAII range, same-thread
//   PCCL_PROF_PUSH(msg, color)          — explicit push
//   PCCL_PROF_POP()                     — explicit pop
//   PCCL_PROF_COLOR_*                   — colour constants
//
// Backend selection (compile-time):
//   PCCL_TRACE_ENABLED=1  → Perfetto trace (Chrome JSON)
//   otherwise             → no-op
// ─────────────────────────────────────────────────────────────────────────────

#if defined(PCCL_TRACE_ENABLED) && PCCL_TRACE_ENABLED
// ── Perfetto-compatible trace path ──────────────────────────────────────────

#include "trace_writer.hpp"

namespace pccl {
class PcclProfRangeGuard {
 public:
  PcclProfRangeGuard(const char* msg, uint32_t color = 0) {
    trace::pushRange(msg, color);
  }

  ~PcclProfRangeGuard() {
    trace::popRange();
  }

  PcclProfRangeGuard(const PcclProfRangeGuard&) = delete;
  PcclProfRangeGuard& operator=(const PcclProfRangeGuard&) = delete;
};
}  // namespace pccl

#define PCCL_PROF_PUSH(msg, clr) ::pccl::trace::pushRange((msg), (clr))
#define PCCL_PROF_POP()          ::pccl::trace::popRange()

// Color constants (ARGB)
#define PCCL_PROF_COLOR_COPY_OUTER    0xFF0077BB
#define PCCL_PROF_COLOR_COPY_THREAD   0xFF66AADD
#define PCCL_PROF_COLOR_REDUCE_OUTER  0xFFCC3311
#define PCCL_PROF_COLOR_REDUCE_THREAD 0xFFDD7766
#define PCCL_PROF_COLOR_SPIN_WAIT     0xFF888888
#define PCCL_PROF_COLOR_BARRIER       0xFF555555
#define PCCL_PROF_COLOR_SLOT_DEP      0xFFAAAA00
#define PCCL_PROF_COLOR_COMPUTE       0xFF33AA33
#define PCCL_PROF_COLOR_SUBMIT        0xFF9933FF
#define PCCL_PROF_COLOR_D2H_EXEC      0xFFFF8800
#define PCCL_PROF_COLOR_H2D_EXEC      0xFF00BBCC
#define PCCL_PROF_COLOR_NOTIFY        0xFFFF00FF

#else
// ── No-op path ────────────────────────────────────────────────────────────────

#define PCCL_PROF_PUSH(msg, clr) ((void)0)
#define PCCL_PROF_POP()          ((void)0)

namespace pccl {
class PcclProfRangeGuard {
 public:
  PcclProfRangeGuard(const char*, uint32_t = 0) {}
  PcclProfRangeGuard(const PcclProfRangeGuard&) = delete;
  PcclProfRangeGuard& operator=(const PcclProfRangeGuard&) = delete;
};
}  // namespace pccl

#define PCCL_PROF_COLOR_COPY_OUTER    0
#define PCCL_PROF_COLOR_COPY_THREAD   0
#define PCCL_PROF_COLOR_REDUCE_OUTER  0
#define PCCL_PROF_COLOR_REDUCE_THREAD 0
#define PCCL_PROF_COLOR_SPIN_WAIT     0
#define PCCL_PROF_COLOR_BARRIER       0
#define PCCL_PROF_COLOR_SLOT_DEP      0
#define PCCL_PROF_COLOR_COMPUTE       0
#define PCCL_PROF_COLOR_SUBMIT        0
#define PCCL_PROF_COLOR_D2H_EXEC      0
#define PCCL_PROF_COLOR_H2D_EXEC      0
#define PCCL_PROF_COLOR_NOTIFY        0

#endif  // backend selection

#endif  // PCCL_PROF_HPP
