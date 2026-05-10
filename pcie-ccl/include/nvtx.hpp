#ifndef PCIECCL_NVTX_HPP
#define PCIECCL_NVTX_HPP

#if defined(PCCL_NVTX_ENABLED) && PCCL_NVTX_ENABLED && defined(PCCL_DEVICE_CUDA)

#include <nvToolsExt.h>
#include <cstdio>

// Push a named, colored NVTX range (clr is ARGB uint32_t)
#define PCCL_NVTX_RANGE_PUSH(msg, clr)            \
  do {                                             \
    nvtxEventAttributes_t attr = {};               \
    attr.version = NVTX_VERSION;                   \
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;     \
    attr.colorType = NVTX_COLOR_ARGB;              \
    attr.color = (clr);                            \
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;    \
    attr.message.ascii = (msg);                    \
    nvtxRangePushEx(&attr);                        \
  } while (0)

#define PCCL_NVTX_RANGE_POP() nvtxRangePop()

namespace pccl {

// RAII guard: pushes on construction, pops on destruction
class NvtxRangeGuard {
 public:
  NvtxRangeGuard(const char* msg, uint32_t color) {
    nvtxEventAttributes_t attr = {};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = color;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = msg;
    nvtxRangePushEx(&attr);
  }
  ~NvtxRangeGuard() { nvtxRangePop(); }

  NvtxRangeGuard(const NvtxRangeGuard&) = delete;
  NvtxRangeGuard& operator=(const NvtxRangeGuard&) = delete;
};

}  // namespace pccl

// Color palette (ARGB)
#define PCCL_NVTX_COLOR_COPY_OUTER    0xFF0077BB  // blue
#define PCCL_NVTX_COLOR_COPY_THREAD   0xFF66AADD  // light blue
#define PCCL_NVTX_COLOR_REDUCE_OUTER  0xFFCC3311  // red
#define PCCL_NVTX_COLOR_REDUCE_THREAD 0xFFDD7766  // light red
#define PCCL_NVTX_COLOR_SPIN_WAIT     0xFF888888  // gray
#define PCCL_NVTX_COLOR_BARRIER       0xFF555555  // dark gray
#define PCCL_NVTX_COLOR_SLOT_DEP      0xFFAAAA00  // yellow
#define PCCL_NVTX_COLOR_COMPUTE       0xFF33AA33  // green
// Scheduler critical-path colors
#define PCCL_NVTX_COLOR_SUBMIT        0xFF9933FF  // purple  - submit() overall
#define PCCL_NVTX_COLOR_D2H_EXEC      0xFFFF8800  // orange  - D2H executor work
#define PCCL_NVTX_COLOR_H2D_EXEC      0xFF00BBCC  // teal    - H2D executor work
#define PCCL_NVTX_COLOR_NOTIFY        0xFFFF00FF  // magenta - cv notify / thread create

#else  // NVTX disabled

#define PCCL_NVTX_RANGE_PUSH(msg, clr) ((void)0)
#define PCCL_NVTX_RANGE_POP() ((void)0)

namespace pccl {

class NvtxRangeGuard {
 public:
  NvtxRangeGuard(const char*, uint32_t) {}
};

}  // namespace pccl

#define PCCL_NVTX_COLOR_COPY_OUTER    0
#define PCCL_NVTX_COLOR_COPY_THREAD   0
#define PCCL_NVTX_COLOR_REDUCE_OUTER  0
#define PCCL_NVTX_COLOR_REDUCE_THREAD 0
#define PCCL_NVTX_COLOR_SPIN_WAIT     0
#define PCCL_NVTX_COLOR_BARRIER       0
#define PCCL_NVTX_COLOR_SLOT_DEP      0
#define PCCL_NVTX_COLOR_COMPUTE       0
#define PCCL_NVTX_COLOR_SUBMIT        0
#define PCCL_NVTX_COLOR_D2H_EXEC      0
#define PCCL_NVTX_COLOR_H2D_EXEC      0
#define PCCL_NVTX_COLOR_NOTIFY        0

#endif  // PCCL_NVTX_ENABLED && PCCL_DEVICE_CUDA

#endif  // PCIECCL_NVTX_HPP
