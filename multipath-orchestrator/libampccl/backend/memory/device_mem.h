#ifndef AMPCCL_BACKEND_MEMORY_DEVICE_MEM_H_
#define AMPCCL_BACKEND_MEMORY_DEVICE_MEM_H_

#include <cstddef>

namespace ampccl {

void* AllocDeviceBuffer(size_t bytes);
void FreeDeviceBuffer(void* ptr);
void DeviceMemcpyD2D(void* dst, const void* src, size_t bytes);
void DeviceMemcpyD2DAsync(void* dst, const void* src, size_t bytes, void* stream);
void DeviceMemcpyD2H(void* host_dst, const void* dev_src, size_t bytes);

}  // namespace ampccl

#endif  // AMPCCL_BACKEND_MEMORY_DEVICE_MEM_H_
