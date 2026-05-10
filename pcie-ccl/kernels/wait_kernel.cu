/*
 * wait_kernel.cu - CUDA WaitCompletionKernel implementation
 *
 * Polls device memory completion flag (uint32_t) until it becomes non-zero.
 * Used to synchronize user_stream with CPU-driven D2H/H2D completion.
 */

#include <cuda_runtime.h>
#include <stdint.h>

// Poll device memory completion flag until it becomes non-zero
__global__ void WaitCompletionKernelImpl(uint32_t* flag_ptr) {
  volatile uint32_t* v_flag = flag_ptr;
  while (*v_flag == 0) {
  }
}

// C wrapper function for launching WaitCompletionKernel from host code
extern "C" void WaitCompletionKernel_cuda(uint32_t* flag_ptr, cudaStream_t stream) {
  WaitCompletionKernelImpl<<<1, 1, 0, stream>>>(flag_ptr);
}
