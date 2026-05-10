#ifndef PCCL_ASCEND_RUNTIME_HPP
#define PCCL_ASCEND_RUNTIME_HPP

#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <cstddef>
#include <cstdint>

namespace pccl {

using devStatus = aclError;
using devHostFunc = void (*)(void*);

enum class MemcpyDirection { H2D, D2H, D2D };

struct devStream {
  aclrtStream handle;
  operator aclrtStream() const { return handle; }
};

struct devEvent {
  aclrtEvent handle;
  operator aclrtEvent() const { return handle; }
};

enum devEventStatus { devEventStatusNotReady = 0, devEventStatusComplete = 1 };

constexpr devStatus devSuccess = ACL_SUCCESS;

devStatus devInit();
devStatus devFinalize();
devStatus devGetDevice(int* deviceId);
devStatus devSetDevice(int deviceId);
devStatus devMalloc(void** devPtr, size_t size);
devStatus devFree(void* devPtr);
devStatus devMallocHost(void** hostPtr, size_t size);
devStatus devFreeHost(void* hostPtr);
devStatus devCreateStream(devStream* stream);
devStatus devCreateHighPriorityStream(devStream* stream);
devStatus devSynchronizeStream(devStream stream);
devStatus devDestroyStream(devStream stream);
devStatus devMemcpyAsync(void* dst, const void* src, size_t size, MemcpyDirection dir, devStream stream);
devStatus devMemcpySync(void* dst, const void* src, size_t size, MemcpyDirection dir);
devStatus devMemsetAsync(void* dst, int value, size_t size, devStream stream);
devStatus devCreateEvent(devEvent* event);
devStatus devCreateSyncEvent(devEvent* event);
devStatus devDestroyEvent(devEvent event);
devStatus devRecordEvent(devEvent event, devStream stream);
devStatus devQueryEvent(devEvent event, devEventStatus* status);
devStatus devStreamWaitEvent(devStream stream, devEvent event);
devStatus devCreateTimingEvent(devEvent* event);
devStatus devEventElapsedTime(float* elapsed_ms, devEvent start, devEvent end);
devStatus devLaunchHostFunc(devStream stream, devHostFunc fn, void* user_data);
bool devCanUseStreamMemOps64();
devStatus devStreamWriteValue64(devStream stream, uint64_t* dev_addr, uint64_t value);
devStatus devStreamWaitValue64(devStream stream, uint64_t* dev_addr, uint64_t value);
bool devD2DReduceSupported(int scalar_type, int reduce_op);
devStatus devLaunchD2DReduce(void* dst, const void* src, size_t bytes, int scalar_type, int reduce_op,
                             devStream stream);
devStatus devPrepareD2DReduce();
devStatus devHostRegister(void* ptr, size_t size);
devStatus devHostUnregister(void* ptr);
devStatus devGetPciBusId(int device, char* bus_id, int len);

}  // namespace pccl

#endif /* PCCL_ASCEND_RUNTIME_HPP */
