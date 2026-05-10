#include "backend/memory/device_mem.h"
#include "common/log.h"
#ifdef AMPCCL_ENABLE_PCIE
#if defined(AMPCCL_USE_ACL_TIMER)
#include <acl/acl_rt.h>
#elif defined(AMPCCL_USE_CUDA_TIMER)
#include <cuda_runtime.h>
#endif
#endif

namespace ampccl {

#ifdef AMPCCL_ENABLE_PCIE
#if defined(AMPCCL_USE_ACL_TIMER)

void* AllocDeviceBuffer(size_t bytes) {
    if (bytes == 0) return nullptr;
    void* ptr = nullptr;
    if (aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
        AMPCCL_LOG(ERROR,"aclrtMalloc failed");
        return nullptr;
    }
    return ptr;
}

void FreeDeviceBuffer(void* ptr) {
    if (ptr) {
        (void)aclrtFree(ptr);
    }
}

void DeviceMemcpyD2D(void* dst, const void* src, size_t bytes) {
    if (bytes == 0 || !dst || !src) return;
    (void)aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE);
}

void DeviceMemcpyD2DAsync(void* dst, const void* src, size_t bytes, void* stream) {
    if (bytes == 0 || !dst || !src || !stream) return;
    (void)aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                          static_cast<aclrtStream>(stream));
}

void DeviceMemcpyD2H(void* host_dst, const void* dev_src, size_t bytes) {
    if (bytes == 0 || !host_dst || !dev_src) return;
    (void)aclrtMemcpy(host_dst, bytes, dev_src, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
}

#elif defined(AMPCCL_USE_CUDA_TIMER)

void* AllocDeviceBuffer(size_t bytes) {
    if (bytes == 0) return nullptr;
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
        AMPCCL_LOG(ERROR,"cudaMalloc failed");
        return nullptr;
    }
    return ptr;
}

void FreeDeviceBuffer(void* ptr) {
    if (ptr) {
        (void)cudaFree(ptr);
    }
}

void DeviceMemcpyD2D(void* dst, const void* src, size_t bytes) {
    if (bytes == 0 || !dst || !src) return;
    (void)cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
}

void DeviceMemcpyD2DAsync(void* dst, const void* src, size_t bytes, void* stream) {
    if (bytes == 0 || !dst || !src || !stream) return;
    (void)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice,
                         static_cast<cudaStream_t>(stream));
}

void DeviceMemcpyD2H(void* host_dst, const void* dev_src, size_t bytes) {
    if (bytes == 0 || !host_dst || !dev_src) return;
    (void)cudaMemcpy(host_dst, dev_src, bytes, cudaMemcpyDeviceToHost);
}

#else

void* AllocDeviceBuffer(size_t bytes) {
    (void)bytes;
    return nullptr;
}
void FreeDeviceBuffer(void* ptr) { (void)ptr; }
void DeviceMemcpyD2D(void* dst, const void* src, size_t bytes) {
    (void)dst;
    (void)src;
    (void)bytes;
}
void DeviceMemcpyD2DAsync(void* dst, const void* src, size_t bytes, void* stream) {
    (void)dst;
    (void)src;
    (void)bytes;
    (void)stream;
}
void DeviceMemcpyD2H(void* host_dst, const void* dev_src, size_t bytes) {
    (void)host_dst;
    (void)dev_src;
    (void)bytes;
}

#endif  // AMPCCL_USE_ACL_TIMER / CUDA
#else

void* AllocDeviceBuffer(size_t bytes) {
    (void)bytes;
    return nullptr;
}
void FreeDeviceBuffer(void* ptr) { (void)ptr; }
void DeviceMemcpyD2D(void* dst, const void* src, size_t bytes) {
    (void)dst;
    (void)src;
    (void)bytes;
}
void DeviceMemcpyD2DAsync(void* dst, const void* src, size_t bytes, void* stream) {
    (void)dst;
    (void)src;
    (void)bytes;
    (void)stream;
}
void DeviceMemcpyD2H(void* host_dst, const void* dev_src, size_t bytes) {
    (void)host_dst;
    (void)dev_src;
    (void)bytes;
}

#endif  // AMPCCL_ENABLE_PCIE

}  // namespace ampccl
