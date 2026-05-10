#include <cstdint>
#include <cstring>

namespace pccl {

struct WaitCompletionKernelArgs {
  void* flag_ptr;
};

constexpr const char* WAIT_COMPLETION_KERNEL_NAME = "WaitCompletionKernel";
constexpr uint32_t WAIT_COMPLETION_KERNEL_BLOCK_DIM = 1;

}  // namespace pccl

#ifndef ASCENDC_CPU_DEBUG
extern "C" void WaitCompletionKernel_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* flag_ptr);
#endif
