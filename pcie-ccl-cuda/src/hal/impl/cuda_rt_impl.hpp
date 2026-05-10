#ifndef PCCL_CUDA_RT_IMPL_HPP
#define PCCL_CUDA_RT_IMPL_HPP

#include <cuda_runtime.h>
#include <cuda.h>

#include <cstdio>
#include <cstring>

#include "../device_rt.hpp"

extern "C" void WaitCompletionKernel_cuda(uint32_t* flag_ptr, cudaStream_t stream);
extern "C" cudaError_t PcclCollectiveKernel_cuda(const char* kernel_name, uint32_t* complete_flag,
                                                uint64_t* ready_seq, uint64_t work_seq,
                                                cudaStream_t stream);
extern "C" cudaError_t PcclCollectiveKernelPrepare_cuda();
extern "C" void D2DReduceKernel_cuda(void* dst, const void* src, size_t bytes, int scalar_type, int reduce_op,
                                     cudaStream_t stream);
extern "C" cudaError_t D2DReduceKernelPrepare_cuda();
extern "C" cudaError_t PcclWarmupKernel_cuda();

namespace pccl {
namespace detail {

inline devStatus devInitImpl(device_tag::CudaTag) {
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

inline devStatus devMallocMappedHostImpl(device_tag::CudaTag, void** hostPtr, void** devicePtr, size_t size) {
  if (!hostPtr || !devicePtr)
    return cudaErrorInvalidValue;
  *hostPtr = nullptr;
  *devicePtr = nullptr;
  cudaError_t status = cudaHostAlloc(hostPtr, size, cudaHostAllocMapped);
  if (status != cudaSuccess)
    return status;
  status = cudaHostGetDevicePointer(devicePtr, *hostPtr, 0);
  if (status != cudaSuccess) {
    cudaFreeHost(*hostPtr);
    *hostPtr = nullptr;
    *devicePtr = nullptr;
  }
  return status;
}

inline devStatus devFreeMappedHostImpl(device_tag::CudaTag, void* hostPtr) {
  return cudaFreeHost(hostPtr);
}

inline devStatus devCreateStreamImpl(device_tag::CudaTag, devStream* stream) {
  cudaStream_t cudaStream;
  devStatus status = cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking);
  stream->handle = static_cast<void*>(cudaStream);
  return status;
}

inline devStatus devCreateHighPriorityStreamImpl(device_tag::CudaTag, devStream* stream) {
  // Executor DMA lanes run at the greatest stream priority so that a user stream
  // parked on a completion cuStreamWaitValue cannot starve the copy-engine work
  // that actually drives the collective forward (the nccl-tests warmup hang: a
  // single parked WaitValue stalls the cold D2H on a default-priority lane).
  int least = 0, greatest = 0;
  cudaError_t range = cudaDeviceGetStreamPriorityRange(&least, &greatest);
  cudaStream_t cudaStream;
  devStatus status;
  if (range == cudaSuccess && greatest != least) {
    status = cudaStreamCreateWithPriority(&cudaStream, cudaStreamNonBlocking, greatest);
  } else {
    status = cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking);
  }
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

inline devStatus devCreateIpcEventImpl(device_tag::CudaTag, devEvent* event) {
  cudaEvent_t cudaEvent;
  devStatus status = cudaEventCreateWithFlags(&cudaEvent, cudaEventDisableTiming | cudaEventInterprocess);
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
    return cudaSuccess;
  }
  return result;
}

inline devStatus devStreamWaitEventImpl(device_tag::CudaTag, devStream stream, devEvent event) {
  cudaStream_t cudaStream = static_cast<cudaStream_t>(stream.handle);
  cudaEvent_t cudaEvent = static_cast<cudaEvent_t>(event.handle);
  return cudaStreamWaitEvent(cudaStream, cudaEvent, 0);
}

inline devStatus devGetIpcEventHandleImpl(device_tag::CudaTag, devEvent event, void* handle, size_t handle_size) {
  if (!handle || handle_size < sizeof(cudaIpcEventHandle_t))
    return cudaErrorInvalidValue;
  cudaEvent_t cudaEvent = static_cast<cudaEvent_t>(event.handle);
  cudaIpcEventHandle_t ipc_handle;
  cudaError_t status = cudaIpcGetEventHandle(&ipc_handle, cudaEvent);
  if (status == cudaSuccess)
    std::memcpy(handle, &ipc_handle, sizeof(ipc_handle));
  return status;
}

inline devStatus devOpenIpcEventHandleImpl(device_tag::CudaTag, devEvent* event, const void* handle,
                                           size_t handle_size) {
  if (!event || !handle || handle_size < sizeof(cudaIpcEventHandle_t))
    return cudaErrorInvalidValue;
  cudaIpcEventHandle_t ipc_handle;
  std::memcpy(&ipc_handle, handle, sizeof(ipc_handle));
  cudaEvent_t cudaEvent;
  cudaError_t status = cudaIpcOpenEventHandle(&cudaEvent, ipc_handle);
  if (status == cudaSuccess)
    event->handle = static_cast<void*>(cudaEvent);
  return status;
}

inline devStatus devCreateTimingEventImpl(device_tag::CudaTag, devEvent* event) {
  cudaEvent_t cudaEvent;
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
  if (kernel_name && std::strncmp(kernel_name, "PcclCollective_", 15) == 0) {
    struct {
      void* complete_flag_ptr;
      void* ready_seq_ptr;
      uint64_t work_seq;
    }* pargs = static_cast<decltype(pargs)>(args);
    return PcclCollectiveKernel_cuda(kernel_name, static_cast<uint32_t*>(pargs->complete_flag_ptr),
                                     static_cast<uint64_t*>(pargs->ready_seq_ptr), pargs->work_seq, cudaStream);
  }
  if (kernel_name && std::strcmp(kernel_name, "WaitCompletionKernel") == 0) {
    struct {
      void* flag_ptr;
    }* pargs = static_cast<decltype(pargs)>(args);
    WaitCompletionKernel_cuda(static_cast<uint32_t*>(pargs->flag_ptr), cudaStream);
    return cudaGetLastError();
  }
  return cudaErrorInvalidValue;
}

inline devStatus devLaunchHostFuncImpl(device_tag::CudaTag, devStream stream, devHostFunc fn, void* user_data) {
  if (!fn)
    return cudaErrorInvalidValue;
  cudaStream_t cudaStream = static_cast<cudaStream_t>(stream.handle);
  struct HostFuncPayload {
    devHostFunc fn;
    void* user_data;
  };
  return cudaLaunchHostFunc(cudaStream, [](void* data) {
    auto* payload = static_cast<HostFuncPayload*>(data);
    devHostFunc callback = payload->fn;
    void* callback_data = payload->user_data;
    delete payload;
    callback(callback_data);
  }, new HostFuncPayload{fn, user_data});
}

inline bool devCanUseStreamMemOps64Impl(device_tag::CudaTag) {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess)
    return false;
  CUdevice cu_device;
  if (cuDeviceGet(&cu_device, device) != CUDA_SUCCESS)
    return false;
  int supported = 0;
  CUresult status =
      cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, cu_device);
  return status == CUDA_SUCCESS && supported != 0;
}

inline bool devCanFlushRemoteWritesImpl(device_tag::CudaTag) {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess)
    return false;
  CUdevice cu_device;
  if (cuDeviceGet(&cu_device, device) != CUDA_SUCCESS)
    return false;
  int supported = 0;
  CUresult status = cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, cu_device);
  return status == CUDA_SUCCESS && supported != 0;
}

inline devStatus devStreamWriteValue64Impl(device_tag::CudaTag, devStream stream, uint64_t* dev_addr,
                                           uint64_t value) {
  if (!dev_addr)
    return cudaErrorInvalidValue;
  CUstream cu_stream = reinterpret_cast<CUstream>(stream.handle);
  CUdeviceptr addr = reinterpret_cast<CUdeviceptr>(dev_addr);
  CUresult status =
      cuStreamWriteValue64(cu_stream, addr, static_cast<cuuint64_t>(value), CU_STREAM_WRITE_VALUE_DEFAULT);
  return static_cast<devStatus>(status);
}

inline devStatus devStreamWaitValue64Impl(device_tag::CudaTag, devStream stream, uint64_t* dev_addr,
                                          uint64_t value, bool flush_remote_writes) {
  if (!dev_addr)
    return cudaErrorInvalidValue;
  CUstream cu_stream = reinterpret_cast<CUstream>(stream.handle);
  CUdeviceptr addr = reinterpret_cast<CUdeviceptr>(dev_addr);
  unsigned int flags = CU_STREAM_WAIT_VALUE_GEQ;
  if (flush_remote_writes)
    flags |= CU_STREAM_WAIT_VALUE_FLUSH;
  CUresult status = cuStreamWaitValue64(cu_stream, addr, static_cast<cuuint64_t>(value), flags);
  return static_cast<devStatus>(status);
}

inline devStatus devLaunchD2DReduceImpl(device_tag::CudaTag, void* dst, const void* src, size_t bytes,
                                        int scalar_type, int reduce_op, devStream stream) {
  if (!dst || !src)
    return cudaErrorInvalidValue;
  cudaStream_t cudaStream = static_cast<cudaStream_t>(stream.handle);
  D2DReduceKernel_cuda(dst, src, bytes, scalar_type, reduce_op, cudaStream);
  return cudaGetLastError();
}

inline devStatus devPrepareD2DReduceImpl(device_tag::CudaTag) {
  return D2DReduceKernelPrepare_cuda();
}

inline devStatus devPrepareCollectiveKernelImpl(device_tag::CudaTag) {
  return PcclCollectiveKernelPrepare_cuda();
}

inline devStatus devWarmupKernelImpl(device_tag::CudaTag) {
  return PcclWarmupKernel_cuda();
}

inline devStatus devHostRegisterImpl(device_tag::CudaTag, void* ptr, size_t size) {
  return cudaHostRegister(ptr, size, cudaHostRegisterPortable);
}

inline devStatus devHostUnregisterImpl(device_tag::CudaTag, void* ptr) {
  return cudaHostUnregister(ptr);
}

inline devStatus devGetPciBusIdImpl(device_tag::CudaTag, int device, char* bus_id, int len) {
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, device);
  if (err != cudaSuccess)
    return err;
  // Format: "0000:XX:YY.Z"
  snprintf(bus_id, len, "%04x:%02x:%02x.0", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
  return cudaSuccess;
}

inline devStatus devCanAccessPeerImpl(device_tag::CudaTag, int device, int peer_device, int* can_access) {
  return cudaDeviceCanAccessPeer(can_access, device, peer_device);
}

inline devStatus devEnablePeerAccessImpl(device_tag::CudaTag, int peer_device) {
  cudaError_t err = cudaDeviceEnablePeerAccess(peer_device, 0);
  if (err == cudaErrorPeerAccessAlreadyEnabled) {
    cudaGetLastError();
    return cudaSuccess;
  }
  return err;
}

inline devStatus devMemcpyPeerAsyncImpl(device_tag::CudaTag, void* dst, int dst_device, const void* src,
                                        int src_device, size_t size, devStream stream) {
  cudaStream_t cudaStream = static_cast<cudaStream_t>(stream.handle);
  return cudaMemcpyPeerAsync(dst, dst_device, src, src_device, size, cudaStream);
}

}  // namespace detail
}  // namespace pccl

#endif /* PCCL_CUDA_RT_IMPL_HPP */
