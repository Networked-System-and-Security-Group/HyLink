#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

__device__ __forceinline__ void WaitCompletionLoop(volatile uint32_t* flag_ptr) {
  while (*flag_ptr == 0) {
    // Spin until flag is set by CPU
  }
}

__device__ __forceinline__ void CollectiveDoorbellLoop(volatile uint32_t* complete_flag,
                                                       volatile uint64_t* ready_seq,
                                                       uint64_t work_seq) {
  *ready_seq = work_seq;
  __threadfence_system();
  while (*complete_flag == 0) {
    // Spin until the executor signals collective completion.
  }
}

__global__ void WaitCompletionKernelImpl(volatile uint32_t* flag_ptr) {
  WaitCompletionLoop(flag_ptr);
}

__global__ void CollectiveDoorbellKernelImpl(volatile uint32_t* complete_flag, volatile uint64_t* ready_seq,
                                             uint64_t work_seq) {
  CollectiveDoorbellLoop(complete_flag, ready_seq, work_seq);
}

extern "C" void WaitCompletionKernel_cuda(uint32_t* flag_ptr, cudaStream_t stream) {
  WaitCompletionKernelImpl<<<1, 1, 0, stream>>>(flag_ptr);
}

#define DEFINE_COLLECTIVE_KERNEL(OP, RANK, SLOT)                                                   \
  extern "C" __global__ void PcclCollective_##OP##_R##RANK##_S##SLOT(                              \
      volatile uint32_t* complete_flag, volatile uint64_t* ready_seq, uint64_t work_seq) {          \
    CollectiveDoorbellLoop(complete_flag, ready_seq, work_seq);                                     \
  }

#define DEFINE_COLLECTIVE_KERNEL_RX(OP, SLOT)                                                      \
  extern "C" __global__ void PcclCollective_##OP##_RX_S##SLOT(                                     \
      volatile uint32_t* complete_flag, volatile uint64_t* ready_seq, uint64_t work_seq) {          \
    CollectiveDoorbellLoop(complete_flag, ready_seq, work_seq);                                     \
  }

#define DEFINE_COLLECTIVE_KERNEL_RANKS(OP, SLOT)                                                  \
  DEFINE_COLLECTIVE_KERNEL(OP, 0, SLOT)                                                           \
  DEFINE_COLLECTIVE_KERNEL(OP, 1, SLOT)                                                           \
  DEFINE_COLLECTIVE_KERNEL(OP, 2, SLOT)                                                           \
  DEFINE_COLLECTIVE_KERNEL(OP, 3, SLOT)                                                           \
  DEFINE_COLLECTIVE_KERNEL(OP, 4, SLOT)                                                           \
  DEFINE_COLLECTIVE_KERNEL(OP, 5, SLOT)                                                           \
  DEFINE_COLLECTIVE_KERNEL(OP, 6, SLOT)                                                           \
  DEFINE_COLLECTIVE_KERNEL(OP, 7, SLOT)                                                           \
  DEFINE_COLLECTIVE_KERNEL_RX(OP, SLOT)

#define DEFINE_COLLECTIVE_KERNEL_SLOTS(OP)                                                        \
  DEFINE_COLLECTIVE_KERNEL_RANKS(OP, 00)                                                          \
  DEFINE_COLLECTIVE_KERNEL_RANKS(OP, 01)                                                          \
  DEFINE_COLLECTIVE_KERNEL_RANKS(OP, 02)                                                          \
  DEFINE_COLLECTIVE_KERNEL_RANKS(OP, 03)                                                          \
  DEFINE_COLLECTIVE_KERNEL_RANKS(OP, 04)                                                          \
  DEFINE_COLLECTIVE_KERNEL_RANKS(OP, 05)                                                          \
  DEFINE_COLLECTIVE_KERNEL_RANKS(OP, 06)                                                          \
  DEFINE_COLLECTIVE_KERNEL_RANKS(OP, 07)                                                          \
  DEFINE_COLLECTIVE_KERNEL_RANKS(OP, 08)                                                          \
  DEFINE_COLLECTIVE_KERNEL_RANKS(OP, 09)                                                          \
  DEFINE_COLLECTIVE_KERNEL_RANKS(OP, 10)                                                          \
  DEFINE_COLLECTIVE_KERNEL_RANKS(OP, 11)                                                          \
  DEFINE_COLLECTIVE_KERNEL_RANKS(OP, 12)                                                          \
  DEFINE_COLLECTIVE_KERNEL_RANKS(OP, 13)                                                          \
  DEFINE_COLLECTIVE_KERNEL_RANKS(OP, 14)                                                          \
  DEFINE_COLLECTIVE_KERNEL_RANKS(OP, 15)

DEFINE_COLLECTIVE_KERNEL_SLOTS(AllReduce)
DEFINE_COLLECTIVE_KERNEL_SLOTS(AllGather)
DEFINE_COLLECTIVE_KERNEL_SLOTS(ReduceScatter)
DEFINE_COLLECTIVE_KERNEL_SLOTS(Broadcast)
DEFINE_COLLECTIVE_KERNEL_SLOTS(Unknown)

using CollectiveKernelFn = void (*)(volatile uint32_t*, volatile uint64_t*, uint64_t);

#define COLLECTIVE_FN(OP, RANK, SLOT) PcclCollective_##OP##_R##RANK##_S##SLOT
#define COLLECTIVE_FN_RX(OP, SLOT) PcclCollective_##OP##_RX_S##SLOT

#define COLLECTIVE_RANK_LIST(OP, SLOT)                                                            \
  COLLECTIVE_FN(OP, 0, SLOT), COLLECTIVE_FN(OP, 1, SLOT), COLLECTIVE_FN(OP, 2, SLOT),             \
      COLLECTIVE_FN(OP, 3, SLOT), COLLECTIVE_FN(OP, 4, SLOT), COLLECTIVE_FN(OP, 5, SLOT),         \
      COLLECTIVE_FN(OP, 6, SLOT), COLLECTIVE_FN(OP, 7, SLOT), COLLECTIVE_FN_RX(OP, SLOT)

#define COLLECTIVE_SLOT_LIST(OP)                                                                  \
  {COLLECTIVE_RANK_LIST(OP, 00)}, {COLLECTIVE_RANK_LIST(OP, 01)},                                 \
      {COLLECTIVE_RANK_LIST(OP, 02)}, {COLLECTIVE_RANK_LIST(OP, 03)},                             \
      {COLLECTIVE_RANK_LIST(OP, 04)}, {COLLECTIVE_RANK_LIST(OP, 05)},                             \
      {COLLECTIVE_RANK_LIST(OP, 06)}, {COLLECTIVE_RANK_LIST(OP, 07)},                             \
      {COLLECTIVE_RANK_LIST(OP, 08)}, {COLLECTIVE_RANK_LIST(OP, 09)},                             \
      {COLLECTIVE_RANK_LIST(OP, 10)}, {COLLECTIVE_RANK_LIST(OP, 11)},                             \
      {COLLECTIVE_RANK_LIST(OP, 12)}, {COLLECTIVE_RANK_LIST(OP, 13)},                             \
      {COLLECTIVE_RANK_LIST(OP, 14)}, {COLLECTIVE_RANK_LIST(OP, 15)}

static CollectiveKernelFn kCollectiveKernels[5][16][9] = {
    {COLLECTIVE_SLOT_LIST(AllReduce)},
    {COLLECTIVE_SLOT_LIST(AllGather)},
    {COLLECTIVE_SLOT_LIST(ReduceScatter)},
    {COLLECTIVE_SLOT_LIST(Broadcast)},
    {COLLECTIVE_SLOT_LIST(Unknown)},
};

static int parseCollectiveOp(const char* op) {
  if (std::strcmp(op, "AllReduce") == 0)
    return 0;
  if (std::strcmp(op, "AllGather") == 0)
    return 1;
  if (std::strcmp(op, "ReduceScatter") == 0)
    return 2;
  if (std::strcmp(op, "Broadcast") == 0)
    return 3;
  if (std::strcmp(op, "Unknown") == 0)
    return 4;
  return 4;
}

extern "C" cudaError_t PcclCollectiveKernel_cuda(const char* kernel_name, uint32_t* complete_flag,
                                                 uint64_t* ready_seq, uint64_t work_seq,
                                                 cudaStream_t stream) {
  constexpr const char prefix[] = "PcclCollective_";
  if (!kernel_name || std::strncmp(kernel_name, prefix, sizeof(prefix) - 1) != 0) {
    CollectiveDoorbellKernelImpl<<<1, 1, 0, stream>>>(complete_flag, ready_seq, work_seq);
    return cudaGetLastError();
  }

  const char* op_begin = kernel_name + sizeof(prefix) - 1;
  const char* rank_marker = std::strstr(op_begin, "_R");
  if (!rank_marker) {
    CollectiveDoorbellKernelImpl<<<1, 1, 0, stream>>>(complete_flag, ready_seq, work_seq);
    return cudaGetLastError();
  }
  char op_name[32] = {};
  size_t op_len = static_cast<size_t>(rank_marker - op_begin);
  if (op_len >= sizeof(op_name))
    op_len = sizeof(op_name) - 1;
  std::memcpy(op_name, op_begin, op_len);
  int op = parseCollectiveOp(op_name);

  const char* rank_begin = rank_marker + 2;
  const char* slot_marker = std::strstr(rank_begin, "_S");
  if (!slot_marker) {
    CollectiveDoorbellKernelImpl<<<1, 1, 0, stream>>>(complete_flag, ready_seq, work_seq);
    return cudaGetLastError();
  }
  int rank = 8;
  if (*rank_begin >= '0' && *rank_begin <= '7')
    rank = *rank_begin - '0';

  const char* slot_begin = slot_marker + 2;
  int slot = -1;
  if (slot_begin[0] >= '0' && slot_begin[0] <= '9' && slot_begin[1] >= '0' && slot_begin[1] <= '9')
    slot = (slot_begin[0] - '0') * 10 + (slot_begin[1] - '0');
  if (slot < 0 || slot >= 16) {
    CollectiveDoorbellKernelImpl<<<1, 1, 0, stream>>>(complete_flag, ready_seq, work_seq);
    return cudaGetLastError();
  }

  void* args[] = {&complete_flag, &ready_seq, &work_seq};
  cudaLaunchKernel(reinterpret_cast<const void*>(kCollectiveKernels[op][slot][rank]), dim3(1), dim3(1), args,
                   0, stream);
  return cudaGetLastError();
}

// One-shot context warm-up. A freshly created CUDA context whose first GPU work
// is a pure-DMA collective (no compute kernel has ever executed + no device sync
// has occurred) deadlocks: the copy engine schedules a couple of subchunk DMAs
// then permanently stops, and the recorded events never complete. Executing one
// real compute kernel followed by a full cudaDeviceSynchronize() activates the
// context's execution path, after which arbitrarily deep DMA-only collective
// bursts run fine. This mirrors exactly what nccl-tests' initData() does
// (prepareInput kernel on the default stream + cudaDeviceSynchronize) before its
// first benchmarked collective. Diagnostic/fix probe gated by PCCL_WARMUP_KERNEL.
__global__ void PcclWarmupKernelImpl(volatile int* p) {
  if (p)
    *p = 1;
}

extern "C" cudaError_t PcclWarmupKernel_cuda() {
  int dev = -1;
  cudaGetDevice(&dev);
  constexpr int kMaxWarmupDevices = 64;
  static int* d_scratch[kMaxWarmupDevices] = {};
  if (dev < 0 || dev >= kMaxWarmupDevices)
    return cudaErrorInvalidDevice;
  if (d_scratch[dev] == nullptr) {
    cudaError_t e = cudaMalloc(&d_scratch[dev], sizeof(int));
    if (e != cudaSuccess) {
      std::fprintf(stderr, "PcclWarmupKernel_cuda cudaMalloc failed device=%d error=%d %s\n", dev,
                   static_cast<int>(e), cudaGetErrorString(e));
      return e;
    }
  }
  PcclWarmupKernelImpl<<<1, 1>>>(d_scratch[dev]);  // default stream
  cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess) {
    std::fprintf(stderr, "PcclWarmupKernel_cuda launch failed device=%d error=%d %s\n", dev,
                 static_cast<int>(le), cudaGetErrorString(le));
    return le;
  }
  cudaError_t se = cudaDeviceSynchronize();
  if (se != cudaSuccess) {
    std::fprintf(stderr, "PcclWarmupKernel_cuda synchronize failed device=%d error=%d %s\n", dev,
                 static_cast<int>(se), cudaGetErrorString(se));
  }
  return se;
}

extern "C" cudaError_t PcclCollectiveKernelPrepare_cuda() {
  cudaFuncAttributes attrs;
  cudaError_t status = cudaFuncGetAttributes(&attrs, reinterpret_cast<const void*>(CollectiveDoorbellKernelImpl));
  if (status != cudaSuccess)
    return status;
  status = cudaFuncGetAttributes(&attrs, reinterpret_cast<const void*>(WaitCompletionKernelImpl));
  if (status != cudaSuccess)
    return status;
  for (int op = 0; op < 5; ++op) {
    for (int slot = 0; slot < 16; ++slot) {
      for (int rank = 0; rank < 9; ++rank) {
        status = cudaFuncGetAttributes(&attrs, reinterpret_cast<const void*>(kCollectiveKernels[op][slot][rank]));
        if (status != cudaSuccess)
          return status;
      }
    }
  }
  return cudaSuccess;
}

enum KernelScalarDataType {
  KernelInt8 = 0,
  KernelUint8 = 1,
  KernelInt32 = 2,
  KernelUint32 = 3,
  KernelInt64 = 4,
  KernelUint64 = 5,
  KernelFloat16 = 6,
  KernelFloat32 = 7,
  KernelFloat64 = 8,
  KernelBfloat16 = 9,
};

enum KernelReduceOp {
  KernelSum = 0,
  KernelMax = 1,
  KernelMin = 2,
};

template <typename T>
__device__ T reduceValue(T a, T b, int op) {
  if (op == KernelMax)
    return a > b ? a : b;
  if (op == KernelMin)
    return a < b ? a : b;
  return static_cast<T>(a + b);
}

template <>
__device__ __half reduceValue<__half>(__half a, __half b, int op) {
  float af = __half2float(a);
  float bf = __half2float(b);
  if (op == KernelMax)
    return __float2half(af > bf ? af : bf);
  if (op == KernelMin)
    return __float2half(af < bf ? af : bf);
  return __float2half(af + bf);
}

template <>
__device__ __nv_bfloat16 reduceValue<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b, int op) {
  float af = __bfloat162float(a);
  float bf = __bfloat162float(b);
  if (op == KernelMax)
    return __float2bfloat16(af > bf ? af : bf);
  if (op == KernelMin)
    return __float2bfloat16(af < bf ? af : bf);
  return __float2bfloat16(af + bf);
}

template <typename T>
__global__ void D2DReduceTypedKernel(T* dst, const T* src, size_t count, int reduce_op) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t i = idx; i < count; i += stride) {
    dst[i] = reduceValue<T>(dst[i], src[i], reduce_op);
  }
}

__global__ void D2DReduceHalfRawKernel(uint16_t* dst, const uint16_t* src, size_t count, int reduce_op) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t i = idx; i < count; i += stride) {
    float a = __half2float(__ushort_as_half(dst[i]));
    float b = __half2float(__ushort_as_half(src[i]));
    float out = a + b;
    if (reduce_op == KernelMax)
      out = a > b ? a : b;
    else if (reduce_op == KernelMin)
      out = a < b ? a : b;
    dst[i] = __half_as_ushort(__float2half(out));
  }
}

__global__ void D2DReduceBfloat16RawKernel(uint16_t* dst, const uint16_t* src, size_t count, int reduce_op) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t i = idx; i < count; i += stride) {
    float a = __bfloat162float(__ushort_as_bfloat16(dst[i]));
    float b = __bfloat162float(__ushort_as_bfloat16(src[i]));
    float out = a + b;
    if (reduce_op == KernelMax)
      out = a > b ? a : b;
    else if (reduce_op == KernelMin)
      out = a < b ? a : b;
    dst[i] = __bfloat16_as_ushort(__float2bfloat16(out));
  }
}

template <typename T>
static void launchD2DReduceTyped(void* dst, const void* src, size_t bytes, int reduce_op, cudaStream_t stream) {
  size_t count = bytes / sizeof(T);
  if (count == 0)
    return;
  int block = 256;
  int grid = static_cast<int>((count + block - 1) / block);
  if (grid > 1024)
    grid = 1024;
  D2DReduceTypedKernel<T><<<grid, block, 0, stream>>>(static_cast<T*>(dst), static_cast<const T*>(src), count,
                                                      reduce_op);
}

static void launchD2DReduceHalfRaw(void* dst, const void* src, size_t bytes, int reduce_op, cudaStream_t stream) {
  size_t count = bytes / sizeof(uint16_t);
  if (count == 0)
    return;
  int block = 256;
  int grid = static_cast<int>((count + block - 1) / block);
  if (grid > 1024)
    grid = 1024;
  D2DReduceHalfRawKernel<<<grid, block, 0, stream>>>(static_cast<uint16_t*>(dst),
                                                     static_cast<const uint16_t*>(src), count, reduce_op);
}

static void launchD2DReduceBfloat16Raw(void* dst, const void* src, size_t bytes, int reduce_op, cudaStream_t stream) {
  size_t count = bytes / sizeof(uint16_t);
  if (count == 0)
    return;
  int block = 256;
  int grid = static_cast<int>((count + block - 1) / block);
  if (grid > 1024)
    grid = 1024;
  D2DReduceBfloat16RawKernel<<<grid, block, 0, stream>>>(static_cast<uint16_t*>(dst),
                                                         static_cast<const uint16_t*>(src), count, reduce_op);
}

extern "C" void D2DReduceKernel_cuda(void* dst, const void* src, size_t bytes, int scalar_type, int reduce_op,
                                     cudaStream_t stream) {
  switch (scalar_type) {
    case KernelInt8:
      launchD2DReduceTyped<int8_t>(dst, src, bytes, reduce_op, stream);
      break;
    case KernelUint8:
      launchD2DReduceTyped<uint8_t>(dst, src, bytes, reduce_op, stream);
      break;
    case KernelInt32:
      launchD2DReduceTyped<int32_t>(dst, src, bytes, reduce_op, stream);
      break;
    case KernelUint32:
      launchD2DReduceTyped<uint32_t>(dst, src, bytes, reduce_op, stream);
      break;
    case KernelInt64:
      launchD2DReduceTyped<int64_t>(dst, src, bytes, reduce_op, stream);
      break;
    case KernelUint64:
      launchD2DReduceTyped<uint64_t>(dst, src, bytes, reduce_op, stream);
      break;
    case KernelFloat16:
      launchD2DReduceHalfRaw(dst, src, bytes, reduce_op, stream);
      break;
    case KernelFloat32:
      launchD2DReduceTyped<float>(dst, src, bytes, reduce_op, stream);
      break;
    case KernelFloat64:
      launchD2DReduceTyped<double>(dst, src, bytes, reduce_op, stream);
      break;
    case KernelBfloat16:
      launchD2DReduceBfloat16Raw(dst, src, bytes, reduce_op, stream);
      break;
    default:
      break;
  }
}

// Resolve EVERY D2DReduce kernel variant the dispatcher (D2DReduceKernel_cuda)
// can launch, in the current context, at init time.
//
// Why: CUDA 12 defaults to CUDA_MODULE_LOADING=LAZY, so a kernel's module/function
// is resolved on its FIRST launch inside cuModuleGetFunction, under a driver lock.
// PCCL launches the reduce kernel from several executor threads (one per
// communicator/device) at the same instant; if the kernel is still unresolved,
// those concurrent first launches deadlock on the driver's module-load mutex
// (all threads stuck in cudaLaunchKernel -> cuModuleGetFunction -> pthread_mutex_lock,
// GPU idle). Previously only the fp32 kernel was warmed, so fp16/bf16/integer
// collectives hung. cudaFuncGetAttributes forces resolution here, on the single
// init thread, so every later launch finds the function already loaded.
extern "C" cudaError_t D2DReduceKernelPrepare_cuda() {
  cudaFuncAttributes attrs;
  const void* kernels[] = {
      reinterpret_cast<const void*>(D2DReduceTypedKernel<int8_t>),
      reinterpret_cast<const void*>(D2DReduceTypedKernel<uint8_t>),
      reinterpret_cast<const void*>(D2DReduceTypedKernel<int32_t>),
      reinterpret_cast<const void*>(D2DReduceTypedKernel<uint32_t>),
      reinterpret_cast<const void*>(D2DReduceTypedKernel<int64_t>),
      reinterpret_cast<const void*>(D2DReduceTypedKernel<uint64_t>),
      reinterpret_cast<const void*>(D2DReduceTypedKernel<float>),
      reinterpret_cast<const void*>(D2DReduceTypedKernel<double>),
      reinterpret_cast<const void*>(D2DReduceHalfRawKernel),
      reinterpret_cast<const void*>(D2DReduceBfloat16RawKernel),
  };
  for (const void* fn : kernels) {
    cudaError_t status = cudaFuncGetAttributes(&attrs, fn);
    if (status != cudaSuccess)
      return status;
  }
  return cudaSuccess;
}
