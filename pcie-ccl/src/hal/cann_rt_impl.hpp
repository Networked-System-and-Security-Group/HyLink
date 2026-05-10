#ifndef CANN_RT_IMPL_HPP
#define CANN_RT_IMPL_HPP

#include <acl/acl.h>

#include <cstring>

#include "../../include/hal/device_rt.hpp"
#include "../../include/hal/device_tags.hpp"

// External declarations for kernel wrapper functions
extern "C" void WaitCompletionKernel_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *flag_ptr);

namespace pccl {
namespace detail {

inline devStatus devInitImpl(device_tag::CannTag) {
  return aclInit(NULL);
}

inline devStatus devGetDeviceImpl(device_tag::CannTag, int *deviceId) {
  return aclrtGetDevice(deviceId);
}

inline devStatus devSetDeviceImpl(device_tag::CannTag, int deviceId) {
  return aclrtSetDevice(deviceId);
}

inline devStatus devMallocImpl(device_tag::CannTag, void **devPtr, size_t size) {
  return aclrtMalloc(devPtr, size, ACL_MEM_MALLOC_HUGE_FIRST);
}

inline devStatus devFreeImpl(device_tag::CannTag, void *devPtr) {
  return aclrtFree(devPtr);
}

inline devStatus devMemcpyAsyncImpl(device_tag::CannTag, void *dst, const void *src, size_t size, MemcpyDirection dir,
                                    devStream stream) {
  aclrtStream aclStream = static_cast<aclrtStream>(stream);
  aclrtMemcpyKind kind;
  switch (dir) {
    case MemcpyDirection::H2H:
      kind = ACL_MEMCPY_HOST_TO_HOST;
      break;
    case MemcpyDirection::H2D:
      kind = ACL_MEMCPY_HOST_TO_DEVICE;
      break;
    case MemcpyDirection::D2H:
      kind = ACL_MEMCPY_DEVICE_TO_HOST;
      break;
    case MemcpyDirection::D2D:
      kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
      break;
    default:
      return -1;
  }
  return aclrtMemcpyAsync(dst, size, src, size, kind, aclStream);
}

inline devStatus devMallocHostImpl(device_tag::CannTag, void **hostPtr, size_t size) {
  return aclrtMallocHost(hostPtr, size);
}

inline devStatus devFreeHostImpl(device_tag::CannTag, void *hostPtr) {
  return aclrtFreeHost(hostPtr);
}

inline devStatus devCreateStreamImpl(device_tag::CannTag, devStream *stream) {
  aclrtStream aclStream;
  devStatus status = aclrtCreateStream(&aclStream);
  stream->handle = static_cast<void *>(aclStream);
  return status;
}

inline devStatus devSynchronizeStreamImpl(device_tag::CannTag, devStream stream) {
  aclrtStream aclStream = static_cast<aclrtStream>(stream);
  return aclrtSynchronizeStream(aclStream);
}

inline devStatus devDestroyStreamImpl(device_tag::CannTag, devStream stream) {
  aclrtStream aclStream = static_cast<aclrtStream>(stream);
  return aclrtDestroyStream(aclStream);
}

inline devStatus devCreateEventImpl(device_tag::CannTag, devEvent *event) {
  aclrtEvent aclEvent;
  devStatus status = aclrtCreateEventWithFlag(&aclEvent, ACL_EVENT_CAPTURE_STREAM_PROGRESS);
  event->handle = static_cast<void *>(aclEvent);
  return status;
}

inline devStatus devCreateSyncEventImpl(device_tag::CannTag, devEvent *event) {
  aclrtEvent aclEvent;
  devStatus status = aclrtCreateEventWithFlag(&aclEvent, ACL_EVENT_SYNC);
  event->handle = static_cast<void *>(aclEvent);
  return status;
}

inline devStatus devDestroyEventImpl(device_tag::CannTag, devEvent event) {
  aclrtEvent aclEvent = static_cast<aclrtEvent>(event);
  return aclrtDestroyEvent(aclEvent);
}

inline devStatus devRecordEventImpl(device_tag::CannTag, devEvent event, devStream stream) {
  aclrtEvent aclEvent = static_cast<aclrtEvent>(event);
  aclrtStream aclStream = static_cast<aclrtStream>(stream);
  return aclrtRecordEvent(aclEvent, aclStream);
}

inline devStatus devQueryEventImpl(device_tag::CannTag, devEvent event, devEventStatus *status) {
  aclrtEvent aclEvent = static_cast<aclrtEvent>(event);
  aclrtEventRecordedStatus aclStatus;
  devStatus result = aclrtQueryEventStatus(aclEvent, &aclStatus);
  if (result == 0) {
    *status = (aclStatus == ACL_EVENT_RECORDED_STATUS_COMPLETE) ? devEventStatusComplete : devEventStatusNotReady;
  }
  return result;
}

inline devStatus devStreamWaitEventImpl(device_tag::CannTag, devStream stream, devEvent event) {
  aclrtStream aclStream = static_cast<aclrtStream>(stream);
  aclrtEvent aclEvent = static_cast<aclrtEvent>(event);
  return aclrtStreamWaitEvent(aclStream, aclEvent);
}

inline devStatus devCreateTimingEventImpl(device_tag::CannTag, devEvent *event) {
  aclrtEvent aclEvent;
  devStatus status = aclrtCreateEventWithFlag(&aclEvent, ACL_EVENT_TIME_LINE);
  event->handle = static_cast<void *>(aclEvent);
  return status;
}

inline devStatus devEventElapsedTimeImpl(device_tag::CannTag, float *elapsed_ms, devEvent start, devEvent end) {
  aclrtEvent aclStart = static_cast<aclrtEvent>(start.handle);
  aclrtEvent aclEnd = static_cast<aclrtEvent>(end.handle);
  return aclrtEventElapsedTime(elapsed_ms, aclStart, aclEnd);
}

inline devStatus devMemcpySyncImpl(device_tag::CannTag, void *dst, const void *src, size_t size, MemcpyDirection dir) {
  aclrtMemcpyKind kind;
  switch (dir) {
    case MemcpyDirection::H2H:
      kind = ACL_MEMCPY_HOST_TO_HOST;
      break;
    case MemcpyDirection::H2D:
      kind = ACL_MEMCPY_HOST_TO_DEVICE;
      break;
    case MemcpyDirection::D2H:
      kind = ACL_MEMCPY_DEVICE_TO_HOST;
      break;
    case MemcpyDirection::D2D:
      kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
      break;
    default:
      return -1;
  }
  return aclrtMemcpy(dst, size, src, size, kind);
}

inline devStatus devLaunchKernelImpl(device_tag::CannTag, const char *kernel_name, uint32_t block_dim, void *args,
                                     size_t, devStream stream) {
  aclrtStream aclStream = static_cast<aclrtStream>(stream);

  if (std::strcmp(kernel_name, "WaitCompletionKernel") == 0) {
    struct {
      void *flag_ptr;
    } *pargs = static_cast<decltype(pargs)>(args);
    WaitCompletionKernel_do(block_dim, nullptr, aclStream, static_cast<uint8_t *>(pargs->flag_ptr));
    return 0;
  }

  return -1;
}

inline devStatus devHostRegisterImpl(device_tag::CannTag, void *ptr, size_t size) {
  void *devPtr;
  return aclrtHostRegister(ptr, size, ACL_HOST_REGISTER_MAPPED, &devPtr);
}

inline devStatus devHostUnregisterImpl(device_tag::CannTag, void *ptr) {
  return aclrtHostUnregister(ptr);
}

}  // namespace detail
}  // namespace pccl

#endif /* CANN_RT_IMPL_HPP */
