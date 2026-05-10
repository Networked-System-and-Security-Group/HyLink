#ifndef PCCL_CANN_RT_IMPL_HPP
#define PCCL_CANN_RT_IMPL_HPP

#include <acl/acl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../device_rt.hpp"

extern "C" void WaitCompletionKernel_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* flag_ptr);

namespace pccl {
namespace detail {

inline devStatus devInitImpl(device_tag::CannTag) {
  return aclInit(NULL);
}

inline devStatus devGetDeviceImpl(device_tag::CannTag, int* deviceId) {
  return aclrtGetDevice(deviceId);
}

inline devStatus devSetDeviceImpl(device_tag::CannTag, int deviceId) {
  return aclrtSetDevice(deviceId);
}

inline devStatus devMallocImpl(device_tag::CannTag, void** devPtr, size_t size) {
  return aclrtMalloc(devPtr, size, ACL_MEM_MALLOC_HUGE_FIRST);
}

inline devStatus devFreeImpl(device_tag::CannTag, void* devPtr) {
  return aclrtFree(devPtr);
}

inline devStatus devMemcpyAsyncImpl(device_tag::CannTag, void* dst, const void* src, size_t size, MemcpyDirection dir,
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

inline devStatus devMallocHostImpl(device_tag::CannTag, void** hostPtr, size_t size) {
  return aclrtMallocHost(hostPtr, size);
}

inline devStatus devFreeHostImpl(device_tag::CannTag, void* hostPtr) {
  return aclrtFreeHost(hostPtr);
}

inline devStatus devMallocMappedHostImpl(device_tag::CannTag, void** hostPtr, void** devicePtr, size_t size) {
  (void)hostPtr;
  (void)devicePtr;
  (void)size;
  return -1;
}

inline devStatus devFreeMappedHostImpl(device_tag::CannTag, void* hostPtr) {
  return aclrtFreeHost(hostPtr);
}

inline devStatus devCreateStreamImpl(device_tag::CannTag, devStream* stream) {
  aclrtStream aclStream;
  devStatus status = aclrtCreateStream(&aclStream);
  stream->handle = static_cast<void*>(aclStream);
  return status;
}

inline devStatus devCreateHighPriorityStreamImpl(device_tag::CannTag, devStream* stream) {
  // CANN has no stream-priority knob exposed here; fall back to a normal stream.
  return devCreateStreamImpl(device_tag::CannTag{}, stream);
}

inline devStatus devSynchronizeStreamImpl(device_tag::CannTag, devStream stream) {
  aclrtStream aclStream = static_cast<aclrtStream>(stream);
  return aclrtSynchronizeStream(aclStream);
}

inline devStatus devDestroyStreamImpl(device_tag::CannTag, devStream stream) {
  aclrtStream aclStream = static_cast<aclrtStream>(stream);
  return aclrtDestroyStream(aclStream);
}

inline devStatus devCreateEventImpl(device_tag::CannTag, devEvent* event) {
  aclrtEvent aclEvent;
  devStatus status = aclrtCreateEventWithFlag(&aclEvent, ACL_EVENT_CAPTURE_STREAM_PROGRESS);
  event->handle = static_cast<void*>(aclEvent);
  return status;
}

inline devStatus devCreateSyncEventImpl(device_tag::CannTag, devEvent* event) {
  aclrtEvent aclEvent;
  devStatus status = aclrtCreateEventWithFlag(&aclEvent, ACL_EVENT_SYNC);
  event->handle = static_cast<void*>(aclEvent);
  return status;
}

inline devStatus devCreateIpcEventImpl(device_tag::CannTag, devEvent*) {
  return -1;
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

inline devStatus devQueryEventImpl(device_tag::CannTag, devEvent event, devEventStatus* status) {
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

inline devStatus devGetIpcEventHandleImpl(device_tag::CannTag, devEvent, void*, size_t) {
  return -1;
}

inline devStatus devOpenIpcEventHandleImpl(device_tag::CannTag, devEvent*, const void*, size_t) {
  return -1;
}

inline devStatus devCreateTimingEventImpl(device_tag::CannTag, devEvent* event) {
  aclrtEvent aclEvent;
  devStatus status = aclrtCreateEventWithFlag(&aclEvent, ACL_EVENT_TIME_LINE);
  event->handle = static_cast<void*>(aclEvent);
  return status;
}

inline devStatus devEventElapsedTimeImpl(device_tag::CannTag, float* elapsed_ms, devEvent start, devEvent end) {
  aclrtEvent aclStart = static_cast<aclrtEvent>(start.handle);
  aclrtEvent aclEnd = static_cast<aclrtEvent>(end.handle);
  return aclrtEventElapsedTime(elapsed_ms, aclStart, aclEnd);
}

inline devStatus devMemcpySyncImpl(device_tag::CannTag, void* dst, const void* src, size_t size, MemcpyDirection dir) {
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

inline devStatus devLaunchKernelImpl(device_tag::CannTag, const char* kernel_name, uint32_t block_dim, void* args,
                                     size_t, devStream stream) {
  aclrtStream aclStream = static_cast<aclrtStream>(stream);
  if (std::strcmp(kernel_name, "WaitCompletionKernel") == 0 || std::strncmp(kernel_name, "PcclWait_", 9) == 0 ||
      std::strncmp(kernel_name, "PcclCollective_", 15) == 0) {
    struct {
      void* flag_ptr;
    }* pargs = static_cast<decltype(pargs)>(args);
    WaitCompletionKernel_do(block_dim, nullptr, aclStream, static_cast<uint8_t*>(pargs->flag_ptr));
    return 0;
  }
  return -1;
}

inline devStatus devLaunchHostFuncImpl(device_tag::CannTag, devStream stream, devHostFunc fn, void* user_data) {
  if (!fn)
    return -1;
  devStatus status = devSynchronizeStreamImpl(device_tag::CannTag{}, stream);
  if (status != 0)
    return status;
  fn(user_data);
  return 0;
}

inline bool devCanUseStreamMemOps64Impl(device_tag::CannTag) {
  return false;
}

inline bool devCanFlushRemoteWritesImpl(device_tag::CannTag) {
  return false;
}

inline devStatus devStreamWriteValue64Impl(device_tag::CannTag, devStream, uint64_t*, uint64_t) {
  return -1;
}

inline devStatus devStreamWaitValue64Impl(device_tag::CannTag, devStream, uint64_t*, uint64_t, bool) {
  return -1;
}

inline devStatus devLaunchD2DReduceImpl(device_tag::CannTag, void*, const void*, size_t, int, int, devStream) {
  return -1;
}

inline devStatus devPrepareD2DReduceImpl(device_tag::CannTag) {
  return -1;
}

inline devStatus devPrepareCollectiveKernelImpl(device_tag::CannTag) {
  return 0;
}

inline devStatus devWarmupKernelImpl(device_tag::CannTag) {
  return 0;
}

inline devStatus devHostRegisterImpl(device_tag::CannTag, void* ptr, size_t size) {
  void* devPtr;
  return aclrtHostRegister(ptr, size, ACL_HOST_REGISTER_MAPPED, &devPtr);
}

inline devStatus devHostUnregisterImpl(device_tag::CannTag, void* ptr) {
  return aclrtHostUnregister(ptr);
}

inline devStatus devGetPciBusIdImpl(device_tag::CannTag, int device_id, char* bus_id, int len) {
  // devdrv_sysfs_bdf_to_devid (present on every Ascend NPU PCI device) contains
  // the authoritative "BB:DD.F ---> N" mapping for all NPUs on the system.
  // Reading it avoids any assumption about PCI enumeration order.
  static const char* pci_root = "/sys/bus/pci/devices";

  // Resolve ACL logical device ID to physical NPU number via ASCEND_RT_VISIBLE_DEVICES.
  // ASCEND_RT_VISIBLE_DEVICES="1,2" means ACL device 0 -> NPU 1, ACL device 1 -> NPU 2.
  int physical_npu = device_id;
  const char* visible = getenv("ASCEND_RT_VISIBLE_DEVICES");
  if (visible && visible[0] != '\0') {
    std::vector<int> vdevs;
    const char* p = visible;
    while (*p) {
      char* end;
      int v = static_cast<int>(std::strtol(p, &end, 10));
      if (end == p) break;
      vdevs.push_back(v);
      p = end;
      if (*p == ',') ++p;
    }
    if (device_id < 0 || device_id >= static_cast<int>(vdevs.size())) return -1;
    physical_npu = vdevs[device_id];
  }

  // Find any NPU PCI device and read its BDF→NPU mapping.
  // All NPU devices expose the same global table; one read is sufficient.
  DIR* dir = opendir(pci_root);
  if (!dir) return -1;

  std::string bdf_map;
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_name[0] == '.') continue;
    std::string path = std::string(pci_root) + "/" + entry->d_name + "/devdrv_sysfs_bdf_to_devid";
    std::ifstream f(path);
    if (!f.is_open()) continue;
    std::ostringstream ss;
    ss << f.rdbuf();
    bdf_map = ss.str();
    break;
  }
  closedir(dir);

  if (bdf_map.empty()) return -1;

  // Parse lines of the form "BB:DD.F ---> N", find the line where N == physical_npu.
  std::istringstream iss(bdf_map);
  std::string line;
  while (std::getline(iss, line)) {
    const std::string arrow = " ---> ";
    auto pos = line.find(arrow);
    if (pos == std::string::npos) continue;
    std::string bdf_short = line.substr(0, pos);
    // trim leading/trailing whitespace
    auto b = bdf_short.find_first_not_of(" \t");
    auto e = bdf_short.find_last_not_of(" \t");
    if (b == std::string::npos) continue;
    bdf_short = bdf_short.substr(b, e - b + 1);

    int npu_num = static_cast<int>(std::strtol(line.c_str() + pos + arrow.size(), nullptr, 10));
    if (npu_num == physical_npu) {
      // bdf_short is "BB:DD.F"; sysfs uses "0000:BB:DD.F"
      std::snprintf(bus_id, static_cast<size_t>(len), "0000:%s", bdf_short.c_str());
      return 0;
    }
  }

  return -1;
}

inline devStatus devCanAccessPeerImpl(device_tag::CannTag, int, int, int* can_access) {
  if (can_access)
    *can_access = 0;
  return -1;
}

inline devStatus devEnablePeerAccessImpl(device_tag::CannTag, int) {
  return -1;
}

inline devStatus devMemcpyPeerAsyncImpl(device_tag::CannTag, void*, int, const void*, int, size_t, devStream) {
  return -1;
}

}  // namespace detail
}  // namespace pccl

#endif /* PCCL_CANN_RT_IMPL_HPP */
