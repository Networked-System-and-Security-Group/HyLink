#ifndef PCCL_DEVICE_RT_HPP
#define PCCL_DEVICE_RT_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace pccl {

using devStatus = int;
using devHostFunc = void (*)(void*);

enum class MemcpyDirection { H2H, H2D, D2H, D2D };

struct devStream {
  void* handle;
  operator void*() const { return handle; }
};

struct devEvent {
  void* handle;
  operator void*() const { return handle; }
};

enum devEventStatus { devEventStatusNotReady = 0, devEventStatusComplete = 1 };

constexpr devStatus devSuccess = 0;

// Device tag dispatch
namespace device_tag {
struct CannTag {};
struct CudaTag {};
}  // namespace device_tag

#if defined(PCCL_DEVICE_CUDA)
using DeviceTag = device_tag::CudaTag;
#elif defined(PCCL_DEVICE_ASCEND)
using DeviceTag = device_tag::CannTag;
#endif

}  // namespace pccl

// Include the appropriate implementation based on device type
#if defined(PCCL_DEVICE_CUDA)
#include "impl/cuda_rt_impl.hpp"
#elif defined(PCCL_DEVICE_ASCEND)
#include "impl/cann_rt_impl.hpp"
#else
#error "No device backend selected. Define PCCL_DEVICE_CUDA or PCCL_DEVICE_ASCEND."
#endif

// Public API: wrapper functions that hide the tag parameter
namespace pccl {

inline devStatus devInit() {
  return detail::devInitImpl(DeviceTag{});
}

inline devStatus devGetDevice(int* deviceId) {
  return detail::devGetDeviceImpl(DeviceTag{}, deviceId);
}

inline devStatus devSetDevice(int deviceId) {
  return detail::devSetDeviceImpl(DeviceTag{}, deviceId);
}

inline devStatus devMalloc(void** devPtr, size_t size) {
  return detail::devMallocImpl(DeviceTag{}, devPtr, size);
}

inline devStatus devFree(void* devPtr) {
  return detail::devFreeImpl(DeviceTag{}, devPtr);
}

inline devStatus devMallocHost(void** hostPtr, size_t size) {
  return detail::devMallocHostImpl(DeviceTag{}, hostPtr, size);
}

inline devStatus devFreeHost(void* hostPtr) {
  return detail::devFreeHostImpl(DeviceTag{}, hostPtr);
}

inline devStatus devMallocMappedHost(void** hostPtr, void** devicePtr, size_t size) {
  return detail::devMallocMappedHostImpl(DeviceTag{}, hostPtr, devicePtr, size);
}

inline devStatus devFreeMappedHost(void* hostPtr) {
  return detail::devFreeMappedHostImpl(DeviceTag{}, hostPtr);
}

inline devStatus devCreateStream(devStream* stream) {
  return detail::devCreateStreamImpl(DeviceTag{}, stream);
}

inline devStatus devCreateHighPriorityStream(devStream* stream) {
  return detail::devCreateHighPriorityStreamImpl(DeviceTag{}, stream);
}

inline devStatus devSynchronizeStream(devStream stream) {
  return detail::devSynchronizeStreamImpl(DeviceTag{}, stream);
}

inline devStatus devDestroyStream(devStream stream) {
  return detail::devDestroyStreamImpl(DeviceTag{}, stream);
}

inline devStatus devMemcpyAsync(void* dst, const void* src, size_t size, MemcpyDirection dir, devStream stream) {
  return detail::devMemcpyAsyncImpl(DeviceTag{}, dst, src, size, dir, stream);
}

inline devStatus devCreateEvent(devEvent* event) {
  return detail::devCreateEventImpl(DeviceTag{}, event);
}

inline devStatus devCreateSyncEvent(devEvent* event) {
  return detail::devCreateSyncEventImpl(DeviceTag{}, event);
}

inline devStatus devCreateIpcEvent(devEvent* event) {
  return detail::devCreateIpcEventImpl(DeviceTag{}, event);
}

inline devStatus devDestroyEvent(devEvent event) {
  return detail::devDestroyEventImpl(DeviceTag{}, event);
}

inline devStatus devRecordEvent(devEvent event, devStream stream) {
  return detail::devRecordEventImpl(DeviceTag{}, event, stream);
}

inline devStatus devQueryEvent(devEvent event, devEventStatus* status) {
  return detail::devQueryEventImpl(DeviceTag{}, event, status);
}

inline devStatus devStreamWaitEvent(devStream stream, devEvent event) {
  return detail::devStreamWaitEventImpl(DeviceTag{}, stream, event);
}

inline devStatus devGetIpcEventHandle(devEvent event, void* handle, size_t handle_size) {
  return detail::devGetIpcEventHandleImpl(DeviceTag{}, event, handle, handle_size);
}

inline devStatus devOpenIpcEventHandle(devEvent* event, const void* handle, size_t handle_size) {
  return detail::devOpenIpcEventHandleImpl(DeviceTag{}, event, handle, handle_size);
}

inline devStatus devCreateTimingEvent(devEvent* event) {
  return detail::devCreateTimingEventImpl(DeviceTag{}, event);
}

inline devStatus devEventElapsedTime(float* elapsed_ms, devEvent start, devEvent end) {
  return detail::devEventElapsedTimeImpl(DeviceTag{}, elapsed_ms, start, end);
}

inline devStatus devMemcpySync(void* dst, const void* src, size_t size, MemcpyDirection dir) {
  return detail::devMemcpySyncImpl(DeviceTag{}, dst, src, size, dir);
}

inline devStatus devLaunchKernel(const char* kernel_name, uint32_t block_dim, void* args, size_t args_size,
                                 devStream stream) {
  return detail::devLaunchKernelImpl(DeviceTag{}, kernel_name, block_dim, args, args_size, stream);
}

inline devStatus devLaunchHostFunc(devStream stream, devHostFunc fn, void* user_data) {
  return detail::devLaunchHostFuncImpl(DeviceTag{}, stream, fn, user_data);
}

inline bool devCanUseStreamMemOps64() {
  return detail::devCanUseStreamMemOps64Impl(DeviceTag{});
}

inline bool devCanFlushRemoteWrites() {
  return detail::devCanFlushRemoteWritesImpl(DeviceTag{});
}

inline devStatus devStreamWriteValue64(devStream stream, uint64_t* dev_addr, uint64_t value) {
  return detail::devStreamWriteValue64Impl(DeviceTag{}, stream, dev_addr, value);
}

inline devStatus devStreamWaitValue64(devStream stream, uint64_t* dev_addr, uint64_t value,
                                      bool flush_remote_writes) {
  return detail::devStreamWaitValue64Impl(DeviceTag{}, stream, dev_addr, value, flush_remote_writes);
}

inline devStatus devLaunchD2DReduce(void* dst, const void* src, size_t bytes, int scalar_type, int reduce_op,
                                    devStream stream) {
  return detail::devLaunchD2DReduceImpl(DeviceTag{}, dst, src, bytes, scalar_type, reduce_op, stream);
}

inline devStatus devPrepareD2DReduce() {
  return detail::devPrepareD2DReduceImpl(DeviceTag{});
}

inline devStatus devPrepareCollectiveKernel() {
  return detail::devPrepareCollectiveKernelImpl(DeviceTag{});
}

inline devStatus devWarmupKernel() {
  return detail::devWarmupKernelImpl(DeviceTag{});
}

inline devStatus devHostRegister(void* ptr, size_t size) {
  return detail::devHostRegisterImpl(DeviceTag{}, ptr, size);
}

inline devStatus devHostUnregister(void* ptr) {
  return detail::devHostUnregisterImpl(DeviceTag{}, ptr);
}

inline devStatus devGetPciBusId(int device, char* bus_id, int len) {
  return detail::devGetPciBusIdImpl(DeviceTag{}, device, bus_id, len);
}

inline devStatus devCanAccessPeer(int device, int peer_device, int* can_access) {
  return detail::devCanAccessPeerImpl(DeviceTag{}, device, peer_device, can_access);
}

inline devStatus devEnablePeerAccess(int peer_device) {
  return detail::devEnablePeerAccessImpl(DeviceTag{}, peer_device);
}

inline devStatus devMemcpyPeerAsync(void* dst, int dst_device, const void* src, int src_device, size_t size,
                                    devStream stream) {
  return detail::devMemcpyPeerAsyncImpl(DeviceTag{}, dst, dst_device, src, src_device, size, stream);
}

}  // namespace pccl

#endif /* PCCL_DEVICE_RT_HPP */
