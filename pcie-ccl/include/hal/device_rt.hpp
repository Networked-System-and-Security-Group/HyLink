#ifndef DEVICE_RT_HPP
#define DEVICE_RT_HPP

#include <cstddef>
#include <cstdint>

namespace pccl {

using devStatus = int;

enum class MemcpyDirection { H2H, H2D, D2H, D2D };

struct devStream {
  void *handle;
  operator void *() const { return handle; }
};

struct devEvent {
  void *handle;
  operator void *() const { return handle; }
};

enum devEventStatus { devEventStatusNotReady = 0, devEventStatusComplete = 1 };

constexpr devStatus devSuccess = 0;

}  // namespace pccl

// Include tag definitions
#include "device_tags.hpp"

// Include the appropriate implementation based on device type
#if defined(PCCL_DEVICE_CUDA)
#include "../../src/hal/cuda_rt_impl.hpp"
#elif defined(PCCL_DEVICE_ASCEND)
#include "../../src/hal/cann_rt_impl.hpp"
#endif

// Public API: wrapper functions that hide the tag parameter
namespace pccl {

inline devStatus devInit() {
  return detail::devInitImpl(DeviceTag{});
}

inline devStatus devGetDevice(int *deviceId) {
  return detail::devGetDeviceImpl(DeviceTag{}, deviceId);
}

inline devStatus devSetDevice(int deviceId) {
  return detail::devSetDeviceImpl(DeviceTag{}, deviceId);
}

inline devStatus devMalloc(void **devPtr, size_t size) {
  return detail::devMallocImpl(DeviceTag{}, devPtr, size);
}

inline devStatus devFree(void *devPtr) {
  return detail::devFreeImpl(DeviceTag{}, devPtr);
}

inline devStatus devMallocHost(void **hostPtr, size_t size) {
  return detail::devMallocHostImpl(DeviceTag{}, hostPtr, size);
}

inline devStatus devFreeHost(void *hostPtr) {
  return detail::devFreeHostImpl(DeviceTag{}, hostPtr);
}

inline devStatus devCreateStream(devStream *stream) {
  return detail::devCreateStreamImpl(DeviceTag{}, stream);
}

inline devStatus devSynchronizeStream(devStream stream) {
  return detail::devSynchronizeStreamImpl(DeviceTag{}, stream);
}

inline devStatus devDestroyStream(devStream stream) {
  return detail::devDestroyStreamImpl(DeviceTag{}, stream);
}

inline devStatus devMemcpyAsync(void *dst, const void *src, size_t size, MemcpyDirection dir, devStream stream) {
  return detail::devMemcpyAsyncImpl(DeviceTag{}, dst, src, size, dir, stream);
}

inline devStatus devCreateEvent(devEvent *event) {
  return detail::devCreateEventImpl(DeviceTag{}, event);
}

inline devStatus devCreateSyncEvent(devEvent *event) {
  return detail::devCreateSyncEventImpl(DeviceTag{}, event);
}

inline devStatus devDestroyEvent(devEvent event) {
  return detail::devDestroyEventImpl(DeviceTag{}, event);
}

inline devStatus devRecordEvent(devEvent event, devStream stream) {
  return detail::devRecordEventImpl(DeviceTag{}, event, stream);
}

inline devStatus devQueryEvent(devEvent event, devEventStatus *status) {
  return detail::devQueryEventImpl(DeviceTag{}, event, status);
}

inline devStatus devStreamWaitEvent(devStream stream, devEvent event) {
  return detail::devStreamWaitEventImpl(DeviceTag{}, stream, event);
}

inline devStatus devCreateTimingEvent(devEvent *event) {
  return detail::devCreateTimingEventImpl(DeviceTag{}, event);
}

inline devStatus devEventElapsedTime(float *elapsed_ms, devEvent start, devEvent end) {
  return detail::devEventElapsedTimeImpl(DeviceTag{}, elapsed_ms, start, end);
}

inline devStatus devMemcpySync(void *dst, const void *src, size_t size, MemcpyDirection dir) {
  return detail::devMemcpySyncImpl(DeviceTag{}, dst, src, size, dir);
}

inline devStatus devLaunchKernel(const char *kernel_name, uint32_t block_dim, void *args, size_t args_size,
                                 devStream stream) {
  return detail::devLaunchKernelImpl(DeviceTag{}, kernel_name, block_dim, args, args_size, stream);
}

inline devStatus devHostRegister(void *ptr, size_t size) {
  return detail::devHostRegisterImpl(DeviceTag{}, ptr, size);
}

inline devStatus devHostUnregister(void *ptr) {
  return detail::devHostUnregisterImpl(DeviceTag{}, ptr);
}

}  // namespace pccl

#endif /* DEVICE_RT_HPP */
