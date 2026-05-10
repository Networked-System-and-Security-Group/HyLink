#ifndef CUDA_RT_IMPL_HPP
#define CUDA_RT_IMPL_HPP

#include <cuda_runtime.h>

#include <cstring>

#include "../../include/hal/device_rt.hpp"
#include "../../include/hal/device_tags.hpp"

// External declarations for CUDA kernel wrapper functions
extern "C" void WaitCompletionKernel_cuda(uint32_t* flag_ptr, cudaStream_t stream);

namespace pccl {
namespace detail {

inline devStatus devInitImpl(device_tag::CudaTag) {
  // Lazy initialization - cudaFree(0) triggers CUDA context creation
  return cudaFree(0);
}

inline devStatus devGetDeviceImpl(device_tag::CudaTag, int* deviceId) {
  return cudaGetDevice(deviceId);
}

inline devStatus devSetDeviceImpl(device_tag::CudaTag, int deviceId) {
  return cudaSetDevice(deviceId);
}

inline devStatus devMallocImpl(device_tag::CudaTag, void** devPtr, size_t size) {
  return cudaMalloc(devPtr, size);
}

inline devStatus devFreeImpl(device_tag::CudaTag, void* devPtr) {
  return cudaFree(devPtr);
}

inline devStatus devMemcpyAsyncImpl(device_tag::CudaTag, void* dst, const void* src, size_t size, MemcpyDirection dir,
                                    devStream stream) {
  cudaStream_t cudaStream = static_cast<cudaStream_t>(stream.handle);
  cudaMemcpyKind kind;
  switch (dir) {
    case MemcpyDirection::H2H:
      kind = cudaMemcpyHostToHost;
      break;
    case MemcpyDirection::H2D:
      kind = cudaMemcpyHostToDevice;
      break;
    case MemcpyDirection::D2H:
      kind = cudaMemcpyDeviceToHost;
      break;
    case MemcpyDirection::D2D:
      kind = cudaMemcpyDeviceToDevice;
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaMemcpyAsync(dst, src, size, kind, cudaStream);
}

inline devStatus devMallocHostImpl(device_tag::CudaTag, void** hostPtr, size_t size) {
  return cudaMallocHost(hostPtr, size);
}

inline devStatus devFreeHostImpl(device_tag::CudaTag, void* hostPtr) {
  return cudaFreeHost(hostPtr);
}

inline devStatus devCreateStreamImpl(device_tag::CudaTag, devStream* stream) {
  cudaStream_t cudaStream;
  devStatus status = cudaStreamCreate(&cudaStream);
  stream->handle = static_cast<void*>(cudaStream);
  return status;
}

inline devStatus devSynchronizeStreamImpl(device_tag::CudaTag, devStream stream) {
  cudaStream_t cudaStream = static_cast<cudaStream_t>(stream.handle);
  return cudaStreamSynchronize(cudaStream);
}

inline devStatus devDestroyStreamImpl(device_tag::CudaTag, devStream stream) {
  cudaStream_t cudaStream = static_cast<cudaStream_t>(stream.handle);
  return cudaStreamDestroy(cudaStream);
}

inline devStatus devCreateEventImpl(device_tag::CudaTag, devEvent* event) {
  cudaEvent_t cudaEvent;
  // Disable timing for lower overhead
  devStatus status = cudaEventCreateWithFlags(&cudaEvent, cudaEventDisableTiming);
  event->handle = static_cast<void*>(cudaEvent);
  return status;
}

inline devStatus devCreateSyncEventImpl(device_tag::CudaTag, devEvent* event) {
  cudaEvent_t cudaEvent;
  devStatus status = cudaEventCreateWithFlags(&cudaEvent, cudaEventDisableTiming);
  event->handle = static_cast<void*>(cudaEvent);
  return status;
}

inline devStatus devDestroyEventImpl(device_tag::CudaTag, devEvent event) {
  cudaEvent_t cudaEvent = static_cast<cudaEvent_t>(event.handle);
  return cudaEventDestroy(cudaEvent);
}

inline devStatus devRecordEventImpl(device_tag::CudaTag, devEvent event, devStream stream) {
  cudaEvent_t cudaEvent = static_cast<cudaEvent_t>(event.handle);
  cudaStream_t cudaStream = static_cast<cudaStream_t>(stream.handle);
  return cudaEventRecord(cudaEvent, cudaStream);
}

inline devStatus devQueryEventImpl(device_tag::CudaTag, devEvent event, devEventStatus* status) {
  cudaEvent_t cudaEvent = static_cast<cudaEvent_t>(event.handle);
  cudaError_t result = cudaEventQuery(cudaEvent);
  if (result == cudaSuccess) {
    *status = devEventStatusComplete;
    return cudaSuccess;
  } else if (result == cudaErrorNotReady) {
    *status = devEventStatusNotReady;
    return cudaSuccess;  // Not an error, just not ready
  }
  return result;
}

inline devStatus devStreamWaitEventImpl(device_tag::CudaTag, devStream stream, devEvent event) {
  cudaStream_t cudaStream = static_cast<cudaStream_t>(stream.handle);
  cudaEvent_t cudaEvent = static_cast<cudaEvent_t>(event.handle);
  return cudaStreamWaitEvent(cudaStream, cudaEvent, 0);
}

inline devStatus devCreateTimingEventImpl(device_tag::CudaTag, devEvent* event) {
  cudaEvent_t cudaEvent;
  // Default flags enable timing
  devStatus status = cudaEventCreate(&cudaEvent);
  event->handle = static_cast<void*>(cudaEvent);
  return status;
}

inline devStatus devEventElapsedTimeImpl(device_tag::CudaTag, float* elapsed_ms, devEvent start, devEvent end) {
  cudaEvent_t cudaStart = static_cast<cudaEvent_t>(start.handle);
  cudaEvent_t cudaEnd = static_cast<cudaEvent_t>(end.handle);
  return cudaEventElapsedTime(elapsed_ms, cudaStart, cudaEnd);
}

inline devStatus devMemcpySyncImpl(device_tag::CudaTag, void* dst, const void* src, size_t size, MemcpyDirection dir) {
  cudaMemcpyKind kind;
  switch (dir) {
    case MemcpyDirection::H2H:
      kind = cudaMemcpyHostToHost;
      break;
    case MemcpyDirection::H2D:
      kind = cudaMemcpyHostToDevice;
      break;
    case MemcpyDirection::D2H:
      kind = cudaMemcpyDeviceToHost;
      break;
    case MemcpyDirection::D2D:
      kind = cudaMemcpyDeviceToDevice;
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaMemcpy(dst, src, size, kind);
}

inline devStatus devLaunchKernelImpl(device_tag::CudaTag, const char* kernel_name, uint32_t, void* args, size_t,
                                     devStream stream) {
  cudaStream_t cudaStream = static_cast<cudaStream_t>(stream.handle);

  if (std::strcmp(kernel_name, "WaitCompletionKernel") == 0) {
    struct {
      void* flag_ptr;
    }* pargs = static_cast<decltype(pargs)>(args);

    WaitCompletionKernel_cuda(static_cast<uint32_t*>(pargs->flag_ptr), cudaStream);
    return cudaSuccess;
  }

  return cudaErrorInvalidValue;
}

inline devStatus devHostRegisterImpl(device_tag::CudaTag, void* ptr, size_t size) {
  return cudaHostRegister(ptr, size, cudaHostRegisterPortable);
}

inline devStatus devHostUnregisterImpl(device_tag::CudaTag, void* ptr) {
  return cudaHostUnregister(ptr);
}

}  // namespace detail
}  // namespace pccl

#endif /* CUDA_RT_IMPL_HPP */
