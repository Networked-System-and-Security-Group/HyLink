// wait_kernel.cpp - Kernel declarations for In-Kernel Handshake synchronization
//
// WaitCompletionKernel: Polls a device memory completion flag (uint32_t) until it becomes non-zero.
// Used to synchronize user_stream with CPU-driven D2H/H2D completion.
//
// The actual Ascend C kernel implementation is in wait_kernel_ascendc.cpp.
// This file provides the host-side declarations and structures.

#include <cstdint>
#include <cstring>

namespace pccl {

// WaitCompletionKernel arguments structure for devLaunchKernel
// Polls a uint32_t flag until it becomes non-zero
struct WaitCompletionKernelArgs {
  void* flag_ptr;  // Pointer to uint32_t completion flag on device
};

// Kernel name constant
constexpr const char* WAIT_COMPLETION_KERNEL_NAME = "WaitCompletionKernel";

// Block dimension for WaitCompletionKernel (single core execution)
constexpr uint32_t WAIT_COMPLETION_KERNEL_BLOCK_DIM = 1;

}  // namespace pccl

// External declaration for the Ascend C kernel wrapper function
// Defined in wait_kernel_ascendc.cpp, linked when compiled with Ascend C toolchain
#ifndef ASCENDC_CPU_DEBUG
extern "C" void WaitCompletionKernel_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* flag_ptr);
#endif
