#include "ascend_runtime.hpp"
#ifdef PCCL_AICORE_REDUCE_ENABLED
#include "pccl_aicore_reduce.h"
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "../ir/compiler.hpp"
#include "../util/log.hpp"

namespace pccl {

namespace {

aclrtMemcpyKind memcpyKind(MemcpyDirection dir, bool* ok = nullptr) {
  if (ok)
    *ok = true;
  switch (dir) {
    case MemcpyDirection::H2D:
      return ACL_MEMCPY_HOST_TO_DEVICE;
    case MemcpyDirection::D2H:
      return ACL_MEMCPY_DEVICE_TO_HOST;
    case MemcpyDirection::D2D:
      return ACL_MEMCPY_DEVICE_TO_DEVICE;
  }
  if (ok)
    *ok = false;
  return ACL_MEMCPY_HOST_TO_HOST;
}

bool scalarTypeToAcl(int scalar_type, aclDataType* out) {
  if (!out)
    return false;
  switch (static_cast<ScalarDataType>(scalar_type)) {
    case ScalarDataType::Int8:
      *out = ACL_INT8;
      return true;
    case ScalarDataType::Uint8:
      *out = ACL_UINT8;
      return true;
    case ScalarDataType::Int32:
      *out = ACL_INT32;
      return true;
    case ScalarDataType::Uint32:
      *out = ACL_UINT32;
      return true;
    case ScalarDataType::Int64:
      *out = ACL_INT64;
      return true;
    case ScalarDataType::Uint64:
      *out = ACL_UINT64;
      return true;
    case ScalarDataType::Float16:
      *out = ACL_FLOAT16;
      return true;
    case ScalarDataType::Float32:
      *out = ACL_FLOAT;
      return true;
    case ScalarDataType::Float64:
      *out = ACL_DOUBLE;
      return true;
    case ScalarDataType::Bfloat16:
      *out = ACL_BF16;
      return true;
  }
  return false;
}

bool reduceOpToAcl(int reduce_op, aclrtReduceKind* out) {
  if (!out)
    return false;
  switch (static_cast<ReduceOpKind>(reduce_op)) {
    case ReduceOpKind::Sum:
      *out = ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM;
      return true;
    case ReduceOpKind::Max:
      *out = ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX;
      return true;
    case ReduceOpKind::Min:
      *out = ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN;
      return true;
  }
  return false;
}

size_t scalarTypeSize(int scalar_type) {
  switch (static_cast<ScalarDataType>(scalar_type)) {
    case ScalarDataType::Int8:
    case ScalarDataType::Uint8:
      return 1;
    case ScalarDataType::Float16:
    case ScalarDataType::Bfloat16:
      return 2;
    case ScalarDataType::Int32:
    case ScalarDataType::Uint32:
    case ScalarDataType::Float32:
      return 4;
    case ScalarDataType::Int64:
    case ScalarDataType::Uint64:
    case ScalarDataType::Float64:
      return 8;
  }
  return 0;
}

std::string readFirstLine(const std::string& path) {
  std::ifstream f(path);
  std::string line;
  if (f.is_open())
    std::getline(f, line);
  return line;
}

bool parseBdfFromSysfs(int device_id, std::string* out) {
  if (!out)
    return false;
  const std::string root = "/sys/class/devdrv-svm/devices";
  for (int idx = 0; idx < 256; ++idx) {
    std::string base = root + "/devdrv" + std::to_string(idx);
    std::string devid = readFirstLine(base + "/devid");
    if (devid.empty())
      continue;
    char* end = nullptr;
    long parsed = std::strtol(devid.c_str(), &end, 0);
    if (end == devid.c_str() || parsed != device_id)
      continue;
    std::string bdf = readFirstLine(base + "/devdrv_sysfs_bdf_to_devid");
    if (!bdf.empty()) {
      *out = bdf;
      return true;
    }
  }
  return false;
}

bool parseBdfFromDevdrvPciSysfs(int device_id, std::string* out) {
  if (!out)
    return false;
  const std::filesystem::path root("/sys/devices");
  std::error_code ec;
  if (!std::filesystem::exists(root, ec))
    return false;
  for (std::filesystem::recursive_directory_iterator it(root, std::filesystem::directory_options::skip_permission_denied, ec),
       end;
       it != end && !ec;
       it.increment(ec)) {
    if (!it->is_regular_file(ec) || it->path().filename() != "devdrv_sysfs_bdf_to_devid")
      continue;
    std::ifstream f(it->path());
    std::string line;
    while (std::getline(f, line)) {
      char raw_bdf[32] = {};
      int parsed_device = -1;
      if (std::sscanf(line.c_str(), " %31s ---> %d", raw_bdf, &parsed_device) != 2)
        continue;
      if (parsed_device != device_id)
        continue;
      std::string bdf(raw_bdf);
      size_t arrow = line.find("--->");
      if (arrow == std::string::npos)
        continue;
      bdf = line.substr(0, arrow);
      bdf.erase(bdf.find_last_not_of(" \t\r\n") + 1);
      bdf.erase(0, bdf.find_first_not_of(" \t\r\n"));
      if (bdf.empty())
        continue;
      if (bdf.rfind("0000:", 0) != 0)
        bdf = "0000:" + bdf;
      *out = bdf;
      return true;
    }
  }
  return false;
}

}  // namespace

devStatus devInit() {
  return aclInit(nullptr);
}

devStatus devFinalize() {
  return aclFinalize();
}

devStatus devGetDevice(int* deviceId) {
  return aclrtGetDevice(deviceId);
}

devStatus devSetDevice(int deviceId) {
  return aclrtSetDevice(deviceId);
}

devStatus devMalloc(void** devPtr, size_t size) {
  return aclrtMalloc(devPtr, size, ACL_MEM_MALLOC_HUGE_FIRST);
}

devStatus devFree(void* devPtr) {
  return aclrtFree(devPtr);
}

devStatus devMallocHost(void** hostPtr, size_t size) {
  return aclrtMallocHost(hostPtr, size);
}

devStatus devFreeHost(void* hostPtr) {
  return aclrtFreeHost(hostPtr);
}

devStatus devCreateStream(devStream* stream) {
  if (!stream)
    return ACL_ERROR_INVALID_PARAM;
  aclrtStream acl_stream = nullptr;
  devStatus status = aclrtCreateStream(&acl_stream);
  stream->handle = acl_stream;
  return status;
}

devStatus devCreateHighPriorityStream(devStream* stream) {
  return devCreateStream(stream);
}

devStatus devSynchronizeStream(devStream stream) {
  return aclrtSynchronizeStream(stream.handle);
}

devStatus devDestroyStream(devStream stream) {
  return stream.handle ? aclrtDestroyStream(stream.handle) : ACL_SUCCESS;
}

devStatus devMemcpyAsync(void* dst, const void* src, size_t size, MemcpyDirection dir, devStream stream) {
  bool ok = false;
  aclrtMemcpyKind kind = memcpyKind(dir, &ok);
  if (!ok)
    return ACL_ERROR_INVALID_PARAM;
  return aclrtMemcpyAsync(dst, size, src, size, kind, stream.handle);
}

devStatus devMemcpySync(void* dst, const void* src, size_t size, MemcpyDirection dir) {
  bool ok = false;
  aclrtMemcpyKind kind = memcpyKind(dir, &ok);
  if (!ok)
    return ACL_ERROR_INVALID_PARAM;
  return aclrtMemcpy(dst, size, src, size, kind);
}

devStatus devMemsetAsync(void* dst, int value, size_t size, devStream stream) {
  return aclrtMemsetAsync(dst, size, value, size, stream.handle);
}

devStatus devCreateEvent(devEvent* event) {
  if (!event)
    return ACL_ERROR_INVALID_PARAM;
  aclrtEvent acl_event = nullptr;
  devStatus status = aclrtCreateEventWithFlag(&acl_event, ACL_EVENT_CAPTURE_STREAM_PROGRESS);
  event->handle = acl_event;
  return status;
}

devStatus devCreateSyncEvent(devEvent* event) {
  if (!event)
    return ACL_ERROR_INVALID_PARAM;
  aclrtEvent acl_event = nullptr;
  devStatus status = aclrtCreateEventWithFlag(&acl_event, ACL_EVENT_SYNC);
  event->handle = acl_event;
  return status;
}

devStatus devDestroyEvent(devEvent event) {
  return event.handle ? aclrtDestroyEvent(event.handle) : ACL_SUCCESS;
}

devStatus devRecordEvent(devEvent event, devStream stream) {
  return aclrtRecordEvent(event.handle, stream.handle);
}

devStatus devQueryEvent(devEvent event, devEventStatus* status) {
  if (!status)
    return ACL_ERROR_INVALID_PARAM;
  aclrtEventRecordedStatus acl_status;
  devStatus result = aclrtQueryEventStatus(event.handle, &acl_status);
  if (result == ACL_SUCCESS) {
    *status = (acl_status == ACL_EVENT_RECORDED_STATUS_COMPLETE) ? devEventStatusComplete
                                                                 : devEventStatusNotReady;
  }
  return result;
}

devStatus devStreamWaitEvent(devStream stream, devEvent event) {
  return aclrtStreamWaitEvent(stream.handle, event.handle);
}

devStatus devCreateTimingEvent(devEvent* event) {
  if (!event)
    return ACL_ERROR_INVALID_PARAM;
  aclrtEvent acl_event = nullptr;
  devStatus status = aclrtCreateEventWithFlag(&acl_event, ACL_EVENT_TIME_LINE);
  event->handle = acl_event;
  return status;
}

devStatus devEventElapsedTime(float* elapsed_ms, devEvent start, devEvent end) {
  return aclrtEventElapsedTime(elapsed_ms, start.handle, end.handle);
}

devStatus devLaunchHostFunc(devStream stream, devHostFunc fn, void* user_data) {
  if (!fn)
    return ACL_ERROR_INVALID_PARAM;
  devStatus status = devSynchronizeStream(stream);
  if (status != ACL_SUCCESS)
    return status;
  fn(user_data);
  return ACL_SUCCESS;
}

bool devCanUseStreamMemOps64() {
  return true;
}

devStatus devStreamWriteValue64(devStream stream, uint64_t* dev_addr, uint64_t value) {
  return aclrtValueWrite(dev_addr, value, 0, stream.handle);
}

devStatus devStreamWaitValue64(devStream stream, uint64_t* dev_addr, uint64_t value) {
  return aclrtValueWait(dev_addr, value, ACL_STREAM_WAIT_VALUE_GEQ, stream.handle);
}

bool devD2DReduceSupported(int scalar_type, int reduce_op) {
  aclrtReduceKind kind;
  if (!reduceOpToAcl(reduce_op, &kind))
    return false;
  switch (static_cast<ScalarDataType>(scalar_type)) {
    case ScalarDataType::Int8:
    case ScalarDataType::Int32:
    case ScalarDataType::Float16:
    case ScalarDataType::Float32:
    case ScalarDataType::Bfloat16:
      return true;
    case ScalarDataType::Uint8:
    case ScalarDataType::Uint32:
    case ScalarDataType::Int64:
    case ScalarDataType::Uint64:
    case ScalarDataType::Float64:
      return false;
  }
  return false;
}

devStatus devLaunchD2DReduce(void* dst, const void* src, size_t bytes, int scalar_type, int reduce_op,
                             devStream stream) {
  aclDataType data_type;
  aclrtReduceKind kind;
  size_t elem_size = scalarTypeSize(scalar_type);
  if (!dst || !src || bytes == 0 || elem_size == 0 || bytes % elem_size != 0 ||
      !devD2DReduceSupported(scalar_type, reduce_op) || !scalarTypeToAcl(scalar_type, &data_type) ||
      !reduceOpToAcl(reduce_op, &kind)) {
    return ACL_ERROR_INVALID_PARAM;
  }
#ifdef PCCL_AICORE_REDUCE_ENABLED
  const aclError aicore_status = pcclAicoreReduceAsync(dst, src, bytes, kind, data_type, stream.handle);
  if (aicore_status != ACL_ERROR_API_NOT_SUPPORT)
    return aicore_status;

  LOG_WARN("AICore reduce unsupported; falling back to aclrtReduceAsync dtype=%d op=%d bytes=%zu status=%d",
           scalar_type, reduce_op, bytes, static_cast<int>(aicore_status));
#endif
  return aclrtReduceAsync(dst, src, bytes, kind, data_type, stream.handle, nullptr);
}

devStatus devPrepareD2DReduce() {
  return ACL_SUCCESS;
}

devStatus devHostRegister(void* ptr, size_t size) {
  void* dev_ptr = nullptr;
  return aclrtHostRegister(ptr, size, ACL_HOST_REGISTER_MAPPED, &dev_ptr);
}

devStatus devHostUnregister(void* ptr) {
  return aclrtHostUnregister(ptr);
}

devStatus devGetPciBusId(int device, char* bus_id, int len) {
  if (!bus_id || len <= 0)
    return ACL_ERROR_INVALID_PARAM;
  std::string bdf;
  if (!parseBdfFromSysfs(device, &bdf) && !parseBdfFromDevdrvPciSysfs(device, &bdf)) {
    std::snprintf(bus_id, static_cast<size_t>(len), "0000:00:%02x.0", device & 0xff);
    return ACL_SUCCESS;
  }
  std::snprintf(bus_id, static_cast<size_t>(len), "%s", bdf.c_str());
  return ACL_SUCCESS;
}

}  // namespace pccl
