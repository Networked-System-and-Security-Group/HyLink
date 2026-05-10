// LD_PRELOAD interposers for the host-blocking CUDA runtime calls that a user
// thread (e.g. PyTorch's .item() / tensor.cpu() / explicit synchronize) issues
// while a PCCL collective is still in flight.
//
// Such a call parks inside the CUDA driver -- holding the driver lock -- behind the
// collective-completion wait that PCCL enqueued on the user stream. PCCL's executor
// thread then cannot acquire the driver lock to advance (and ultimately complete)
// the collective, so the user stream's wait never clears: deadlock, with both
// threads spinning at ~100% CPU.
//
// The fix: before entering the real (blocking) CUDA call, drain any in-flight PCCL
// collective via the executor's CPU condition variable (inflightDrainAll(), which
// never touches the driver lock). With the user thread parked on a condvar instead
// of inside the driver, the executor finishes the collective freely and the
// subsequent real call completes normally. These are host-blocking sync points, so
// draining here does not cost any compute/communication overlap.
//
// Only compiled into libpccl.so for CUDA builds (see CMakeLists.txt).

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>

#include "../../util/inflight_drain.hpp"

namespace {

template <typename Fn>
Fn realSym(const char* name) {
  return reinterpret_cast<Fn>(dlsym(RTLD_NEXT, name));
}

// Diagnostic: log allocation interposition when PCCL_DEBUG_BLOCK_SHIM=1.
bool blockShimDebug() {
  static int cached = -1;
  if (cached < 0) {
    const char* v = std::getenv("PCCL_DEBUG_BLOCK_SHIM");
    cached = (v && v[0] == '1') ? 1 : 0;
  }
  return cached == 1;
}

// True when dst is host memory that is NOT page-locked. A device->pageable-host
// cudaMemcpyAsync is documented to be synchronous w.r.t. the host, which is exactly
// the call that deadlocks; a copy to pinned host memory stays asynchronous and is
// left untouched so normal overlap is preserved.
bool isPageableHostDest(const void* ptr) {
  cudaPointerAttributes attr;
  cudaError_t st = cudaPointerGetAttributes(&attr, ptr);
  if (st != cudaSuccess) {
    // Older runtimes return an error for unregistered (pageable) host pointers.
    cudaGetLastError();
    return true;
  }
  return attr.type == cudaMemoryTypeUnregistered;
}

}  // namespace

extern "C" {

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  using Fn = cudaError_t (*)(cudaStream_t);
  static Fn real = realSym<Fn>("cudaStreamSynchronize");
  if (!pccl::inflightThreadIsInternal())
    pccl::inflightDrainAll();
  return real(stream);
}

cudaError_t cudaDeviceSynchronize(void) {
  using Fn = cudaError_t (*)(void);
  static Fn real = realSym<Fn>("cudaDeviceSynchronize");
  if (!pccl::inflightThreadIsInternal())
    pccl::inflightDrainAll();
  return real();
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  using Fn = cudaError_t (*)(cudaEvent_t);
  static Fn real = realSym<Fn>("cudaEventSynchronize");
  if (!pccl::inflightThreadIsInternal())
    pccl::inflightDrainAll();
  return real(event);
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
  using Fn = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind);
  static Fn real = realSym<Fn>("cudaMemcpy");
  // Plain cudaMemcpy is synchronous w.r.t. the host for every direction.
  if (!pccl::inflightThreadIsInternal())
    pccl::inflightDrainAll();
  return real(dst, src, count, kind);
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind,
                            cudaStream_t stream) {
  using Fn = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
  static Fn real = realSym<Fn>("cudaMemcpyAsync");
  if (!pccl::inflightThreadIsInternal()) {
    const bool d2h = (kind == cudaMemcpyDeviceToHost) || (kind == cudaMemcpyDefault);
    if (d2h && isPageableHostDest(dst))
      pccl::inflightDrainAll();
  }
  return real(dst, src, count, kind, stream);
}

// cudaMalloc and cudaFree are device-synchronizing: inside the driver lock they
// wait for previously issued GPU work to drain before (re)arranging the address
// space. When a user thread (PyTorch caching allocator growing/trimming its pool,
// TE workspace allocation, etc.) calls these while a PCCL collective is still in
// flight, the driver-internal sync parks on the collective's pending user-stream
// wait while holding the driver lock -- so the executor thread cannot acquire the
// lock to advance the collective. Deadlock, identical in shape to the explicit
// synchronize case above, just reached through allocation instead of a sync call.
// (The async variants cudaMallocAsync/cudaFreeAsync are stream-ordered and do NOT
// synchronize the device, so they are intentionally left untouched to preserve
// overlap.) Draining first -- on a CPU condvar, never the driver lock -- lets the
// executor finish the collective before we enter the real allocation call.
cudaError_t cudaMalloc(void** ptr, size_t size) {
  using Fn = cudaError_t (*)(void**, size_t);
  static Fn real = realSym<Fn>("cudaMalloc");
  if (!pccl::inflightThreadIsInternal()) {
    if (blockShimDebug())
      std::fprintf(stderr, "[block_shim] cudaMalloc size=%zu draining inflight\n", size);
    pccl::inflightDrainAll();
  }
  return real(ptr, size);
}

cudaError_t cudaFree(void* ptr) {
  using Fn = cudaError_t (*)(void*);
  static Fn real = realSym<Fn>("cudaFree");
  if (!pccl::inflightThreadIsInternal()) {
    if (blockShimDebug())
      std::fprintf(stderr, "[block_shim] cudaFree ptr=%p draining inflight\n", ptr);
    pccl::inflightDrainAll();
  }
  return real(ptr);
}

// The remaining implicitly-synchronizing host calls (CUDA C Programming Guide,
// "Implicit Synchronization"): page-locked host allocation/free and the
// synchronous cudaMemset. Each parks inside the driver lock waiting for prior GPU
// work, so a user thread issuing one while a PCCL collective is in flight produces
// the same driver-lock deadlock against the executor. Drain first. (cudaMemsetAsync
// is stream-ordered and intentionally NOT intercepted.)
cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
  using Fn = cudaError_t (*)(void*, int, size_t);
  static Fn real = realSym<Fn>("cudaMemset");
  if (!pccl::inflightThreadIsInternal()) {
    if (blockShimDebug())
      std::fprintf(stderr, "[block_shim] cudaMemset count=%zu draining inflight\n", count);
    pccl::inflightDrainAll();
  }
  return real(devPtr, value, count);
}

cudaError_t cudaMallocHost(void** ptr, size_t size) {
  using Fn = cudaError_t (*)(void**, size_t);
  static Fn real = realSym<Fn>("cudaMallocHost");
  if (!pccl::inflightThreadIsInternal()) {
    if (blockShimDebug())
      std::fprintf(stderr, "[block_shim] cudaMallocHost size=%zu draining inflight\n", size);
    pccl::inflightDrainAll();
  }
  return real(ptr, size);
}

cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
  using Fn = cudaError_t (*)(void**, size_t, unsigned int);
  static Fn real = realSym<Fn>("cudaHostAlloc");
  if (!pccl::inflightThreadIsInternal()) {
    if (blockShimDebug())
      std::fprintf(stderr, "[block_shim] cudaHostAlloc size=%zu draining inflight\n", size);
    pccl::inflightDrainAll();
  }
  return real(ptr, size, flags);
}

cudaError_t cudaFreeHost(void* ptr) {
  using Fn = cudaError_t (*)(void*);
  static Fn real = realSym<Fn>("cudaFreeHost");
  if (!pccl::inflightThreadIsInternal()) {
    if (blockShimDebug())
      std::fprintf(stderr, "[block_shim] cudaFreeHost ptr=%p draining inflight\n", ptr);
    pccl::inflightDrainAll();
  }
  return real(ptr);
}

}  // extern "C"
