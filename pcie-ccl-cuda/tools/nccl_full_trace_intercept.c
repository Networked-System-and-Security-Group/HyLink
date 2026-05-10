#define _GNU_SOURCE

#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <nccl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

typedef ncclResult_t (*fn_mem_alloc_t)(void **, size_t);
typedef ncclResult_t (*fn_mem_free_t)(void *);
typedef ncclResult_t (*fn_get_version_t)(int *);
typedef ncclResult_t (*fn_get_unique_id_t)(ncclUniqueId *);
typedef ncclResult_t (*fn_comm_init_rank_config_t)(ncclComm_t *, int, ncclUniqueId, int, ncclConfig_t *);
typedef ncclResult_t (*fn_comm_init_rank_t)(ncclComm_t *, int, ncclUniqueId, int);
typedef ncclResult_t (*fn_comm_init_all_t)(ncclComm_t *, int, const int *);
typedef ncclResult_t (*fn_comm_finalize_t)(ncclComm_t);
typedef ncclResult_t (*fn_comm_destroy_t)(ncclComm_t);
typedef ncclResult_t (*fn_comm_abort_t)(ncclComm_t);
typedef ncclResult_t (*fn_comm_split_t)(ncclComm_t, int, int, ncclComm_t *, ncclConfig_t *);
typedef ncclResult_t (*fn_comm_init_rank_scalable_t)(ncclComm_t *, int, int, int, ncclUniqueId *, ncclConfig_t *);
typedef const char *(*fn_get_error_string_t)(ncclResult_t);
typedef const char *(*fn_get_last_error_t)(ncclComm_t);
typedef ncclResult_t (*fn_comm_get_async_error_t)(ncclComm_t, ncclResult_t *);
typedef ncclResult_t (*fn_comm_count_t)(const ncclComm_t, int *);
typedef ncclResult_t (*fn_comm_device_t)(const ncclComm_t, int *);
typedef ncclResult_t (*fn_comm_user_rank_t)(const ncclComm_t, int *);
typedef ncclResult_t (*fn_comm_register_t)(const ncclComm_t, void *, size_t, void **);
typedef ncclResult_t (*fn_comm_deregister_t)(const ncclComm_t, void *);
typedef ncclResult_t (*fn_red_op_create_t)(ncclRedOp_t *, void *, ncclDataType_t,
                                           ncclScalarResidence_t, ncclComm_t);
typedef ncclResult_t (*fn_red_op_destroy_t)(ncclRedOp_t, ncclComm_t);
typedef ncclResult_t (*fn_group_start_t)(void);
typedef ncclResult_t (*fn_group_end_t)(void);
typedef ncclResult_t (*fn_group_simulate_end_t)(ncclSimInfo_t *);
typedef ncclResult_t (*fn_reduce_t)(const void *, void *, size_t, ncclDataType_t,
                                    ncclRedOp_t, int, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*fn_bcast_t)(void *, size_t, ncclDataType_t, int, ncclComm_t,
                                   cudaStream_t);
typedef ncclResult_t (*fn_broadcast_t)(const void *, void *, size_t, ncclDataType_t, int,
                                       ncclComm_t, cudaStream_t);
typedef ncclResult_t (*fn_all_reduce_t)(const void *, void *, size_t, ncclDataType_t,
                                        ncclRedOp_t, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*fn_reduce_scatter_t)(const void *, void *, size_t, ncclDataType_t,
                                            ncclRedOp_t, ncclComm_t, cudaStream_t);
typedef ncclResult_t (*fn_all_gather_t)(const void *, void *, size_t, ncclDataType_t,
                                        ncclComm_t, cudaStream_t);
typedef ncclResult_t (*fn_send_t)(const void *, size_t, ncclDataType_t, int, ncclComm_t,
                                  cudaStream_t);
typedef ncclResult_t (*fn_recv_t)(void *, size_t, ncclDataType_t, int, ncclComm_t,
                                  cudaStream_t);

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static FILE *g_log = NULL;
static uint64_t g_seq = 0;
static uint64_t g_group_seq = 0;
static __thread int t_group_depth = 0;
static __thread uint64_t t_group_stack[16];

static void *sym(const char *name) {
  void *p = dlsym(RTLD_NEXT, name);
  if (!p) {
    static void *nccl = NULL;
    if (!nccl) {
      nccl = dlopen("libnccl.so.2", RTLD_LAZY | RTLD_LOCAL);
      if (!nccl) nccl = dlopen("libnccl.so", RTLD_LAZY | RTLD_LOCAL);
    }
    if (nccl) p = dlsym(nccl, name);
  }
  if (!p) {
    fprintf(stderr, "nccl_full_trace_intercept: missing symbol %s: %s\n", name, dlerror());
  }
  return p;
}

static void *must_sym(const char *name) {
  void *p = sym(name);
  if (!p) _exit(127);
  return p;
}

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static long tid(void) { return syscall(SYS_gettid); }

static uint64_t next_seq(void) {
  pthread_mutex_lock(&g_lock);
  uint64_t s = ++g_seq;
  pthread_mutex_unlock(&g_lock);
  return s;
}

static void ensure_log(void) {
  if (g_log) return;
  pthread_mutex_lock(&g_lock);
  if (!g_log) {
    const char *dir = getenv("NCCL_TRACE_LOG_DIR");
    if (!dir || !dir[0]) dir = "/tmp";
    char path[1024];
    snprintf(path, sizeof(path), "%s/nccl_full_trace_pid%d.jsonl", dir, getpid());
    g_log = fopen(path, "a");
    if (!g_log) g_log = stderr;
    setvbuf(g_log, NULL, _IOLBF, 0);
  }
  pthread_mutex_unlock(&g_lock);
}

static void log_line(const char *fmt, ...) {
  ensure_log();
  pthread_mutex_lock(&g_lock);
  va_list ap;
  va_start(ap, fmt);
  vfprintf(g_log, fmt, ap);
  va_end(ap);
  fputc('\n', g_log);
  fflush(g_log);
  pthread_mutex_unlock(&g_lock);
}

static uint64_t hash_unique_id(const ncclUniqueId *id) {
  if (!id) return 0;
  uint64_t h = 1469598103934665603ull;
  const unsigned char *p = (const unsigned char *)id->internal;
  for (size_t i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
    h ^= (uint64_t)p[i];
    h *= 1099511628211ull;
  }
  return h;
}

static const char *dtype_name(ncclDataType_t dt) {
  switch (dt) {
    case ncclInt8: return "int8";
    case ncclUint8: return "uint8";
    case ncclInt32: return "int32";
    case ncclUint32: return "uint32";
    case ncclInt64: return "int64";
    case ncclUint64: return "uint64";
    case ncclFloat16: return "float16";
    case ncclFloat32: return "float32";
    case ncclFloat64: return "float64";
#ifdef ncclBfloat16
    case ncclBfloat16: return "bfloat16";
#endif
    default: return "unknown";
  }
}

static size_t dtype_size(ncclDataType_t dt) {
  switch (dt) {
    case ncclInt8:
    case ncclUint8: return 1;
    case ncclFloat16: return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32: return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64: return 8;
#ifdef ncclBfloat16
    case ncclBfloat16: return 2;
#endif
    default: return 0;
  }
}

static const char *op_name(ncclRedOp_t op) {
  switch (op) {
    case ncclSum: return "sum";
    case ncclProd: return "prod";
    case ncclMax: return "max";
    case ncclMin: return "min";
    case ncclAvg: return "avg";
    default: return "custom";
  }
}

static const char *capture_name(enum cudaStreamCaptureStatus st) {
  switch (st) {
    case cudaStreamCaptureStatusNone: return "none";
    case cudaStreamCaptureStatusActive: return "active";
    case cudaStreamCaptureStatusInvalidated: return "invalidated";
    default: return "unknown";
  }
}

static const char *stream_capture(cudaStream_t stream, int *cuda_status) {
  enum cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  cudaError_t err = cudaStreamIsCapturing(stream, &st);
  *cuda_status = (int)err;
  if (err != cudaSuccess) return "query_error";
  return capture_name(st);
}

static uint64_t current_group_id(void) {
  if (t_group_depth <= 0) return 0;
  int idx = t_group_depth - 1;
  if (idx >= 16) idx = 15;
  return t_group_stack[idx];
}

static void comm_info(ncclComm_t comm, int *rank, int *nranks, int *device) {
  *rank = -1;
  *nranks = -1;
  *device = -1;
  fn_comm_user_rank_t user_rank = (fn_comm_user_rank_t)sym("pncclCommUserRank");
  fn_comm_count_t count = (fn_comm_count_t)sym("pncclCommCount");
  fn_comm_device_t cu_device = (fn_comm_device_t)sym("pncclCommCuDevice");
  if (user_rank) user_rank(comm, rank);
  if (count) count(comm, nranks);
  if (cu_device) cu_device(comm, device);
}

static void log_call_ret(const char *api, ncclResult_t ret, const char *args_fmt, ...) {
  ensure_log();
  uint64_t seq = next_seq();
  pthread_mutex_lock(&g_lock);
  fprintf(g_log,
          "{\"ts_ns\":%llu,\"pid\":%d,\"tid\":%ld,\"seq\":%llu,"
          "\"event\":\"call\",\"api\":\"%s\"",
          (unsigned long long)now_ns(), getpid(), tid(), (unsigned long long)seq, api);
  if (args_fmt && args_fmt[0]) {
    fputc(',', g_log);
    va_list ap;
    va_start(ap, args_fmt);
    vfprintf(g_log, args_fmt, ap);
    va_end(ap);
  }
  fprintf(g_log, ",\"group_depth\":%d,\"group_id\":%llu,\"ret\":%d}\n",
          t_group_depth, (unsigned long long)current_group_id(), (int)ret);
  fflush(g_log);
  pthread_mutex_unlock(&g_lock);
}

static void log_collective(const char *api, const void *sendbuff, void *recvbuff, size_t count,
                           size_t send_bytes, size_t recv_bytes, ncclDataType_t datatype,
                           int has_op, ncclRedOp_t op, int root_or_peer, ncclComm_t comm,
                           cudaStream_t stream, int inplace, ncclResult_t ret) {
  int rank, nranks, device, cuda_status;
  comm_info(comm, &rank, &nranks, &device);
  const char *capture = stream_capture(stream, &cuda_status);
  log_line("{\"ts_ns\":%llu,\"pid\":%d,\"tid\":%ld,\"seq\":%llu,"
           "\"event\":\"collective\",\"api\":\"%s\",\"rank\":%d,\"nranks\":%d,"
           "\"device\":%d,\"count\":%zu,\"datatype\":\"%s\",\"dtype_size\":%zu,"
           "\"send_bytes\":%zu,\"recv_bytes\":%zu,\"op\":\"%s\","
           "\"root_or_peer\":%d,\"sendbuff\":\"%p\",\"recvbuff\":\"%p\","
           "\"inplace\":%d,\"stream\":\"%p\",\"capture\":\"%s\","
           "\"cuda_status\":%d,\"group_depth\":%d,\"group_id\":%llu,\"ret\":%d}",
           (unsigned long long)now_ns(), getpid(), tid(), (unsigned long long)next_seq(),
           api, rank, nranks, device, count, dtype_name(datatype), dtype_size(datatype),
           send_bytes, recv_bytes, has_op ? op_name(op) : "none", root_or_peer,
           sendbuff, recvbuff, inplace, stream, capture, cuda_status, t_group_depth,
           (unsigned long long)current_group_id(), (int)ret);
}

ncclResult_t ncclMemAlloc(void **ptr, size_t size) {
  static fn_mem_alloc_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_mem_alloc_t)must_sym("pncclMemAlloc");
  ncclResult_t ret = real_fn(ptr, size);
  log_call_ret("MemAlloc", ret, "\"ptr\":\"%p\",\"size\":%zu", ptr ? *ptr : NULL, size);
  return ret;
}

ncclResult_t ncclMemFree(void *ptr) {
  static fn_mem_free_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_mem_free_t)must_sym("pncclMemFree");
  ncclResult_t ret = real_fn(ptr);
  log_call_ret("MemFree", ret, "\"ptr\":\"%p\"", ptr);
  return ret;
}

ncclResult_t ncclGetVersion(int *version) {
  static fn_get_version_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_get_version_t)must_sym("pncclGetVersion");
  ncclResult_t ret = real_fn(version);
  log_call_ret("GetVersion", ret, "\"version\":%d", version ? *version : -1);
  return ret;
}

ncclResult_t ncclGetUniqueId(ncclUniqueId *uniqueId) {
  static fn_get_unique_id_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_get_unique_id_t)must_sym("pncclGetUniqueId");
  ncclResult_t ret = real_fn(uniqueId);
  log_call_ret("GetUniqueId", ret, "\"unique_id_hash\":\"%016llx\"",
               (unsigned long long)hash_unique_id(uniqueId));
  return ret;
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank,
                                    ncclConfig_t *config) {
  static fn_comm_init_rank_config_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_comm_init_rank_config_t)must_sym("pncclCommInitRankConfig");
  ncclResult_t ret = real_fn(comm, nranks, commId, rank, config);
  int out_rank = -1, out_nranks = -1, out_device = -1;
  if (ret == ncclSuccess && comm && *comm) comm_info(*comm, &out_rank, &out_nranks, &out_device);
  log_call_ret("CommInitRankConfig", ret,
               "\"comm\":\"%p\",\"nranks\":%d,\"rank\":%d,\"unique_id_hash\":\"%016llx\","
               "\"config\":\"%p\",\"out_rank\":%d,\"out_nranks\":%d,\"out_device\":%d",
               comm ? *comm : NULL, nranks, rank, (unsigned long long)hash_unique_id(&commId),
               config, out_rank, out_nranks, out_device);
  return ret;
}

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank) {
  static fn_comm_init_rank_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_comm_init_rank_t)must_sym("pncclCommInitRank");
  ncclResult_t ret = real_fn(comm, nranks, commId, rank);
  int out_rank = -1, out_nranks = -1, out_device = -1;
  if (ret == ncclSuccess && comm && *comm) comm_info(*comm, &out_rank, &out_nranks, &out_device);
  log_call_ret("CommInitRank", ret,
               "\"comm\":\"%p\",\"nranks\":%d,\"rank\":%d,\"unique_id_hash\":\"%016llx\","
               "\"out_rank\":%d,\"out_nranks\":%d,\"out_device\":%d",
               comm ? *comm : NULL, nranks, rank, (unsigned long long)hash_unique_id(&commId),
               out_rank, out_nranks, out_device);
  return ret;
}

ncclResult_t ncclCommInitAll(ncclComm_t *comm, int ndev, const int *devlist) {
  static fn_comm_init_all_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_comm_init_all_t)must_sym("pncclCommInitAll");
  ncclResult_t ret = real_fn(comm, ndev, devlist);
  log_call_ret("CommInitAll", ret, "\"comm\":\"%p\",\"ndev\":%d,\"devlist\":\"%p\"",
               comm ? (void *)comm : NULL, ndev, devlist);
  return ret;
}

ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  static fn_comm_finalize_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_comm_finalize_t)must_sym("pncclCommFinalize");
  ncclResult_t ret = real_fn(comm);
  log_call_ret("CommFinalize", ret, "\"comm\":\"%p\"", comm);
  return ret;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  static fn_comm_destroy_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_comm_destroy_t)must_sym("pncclCommDestroy");
  ncclResult_t ret = real_fn(comm);
  log_call_ret("CommDestroy", ret, "\"comm\":\"%p\"", comm);
  return ret;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) {
  static fn_comm_abort_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_comm_abort_t)must_sym("pncclCommAbort");
  ncclResult_t ret = real_fn(comm);
  log_call_ret("CommAbort", ret, "\"comm\":\"%p\"", comm);
  return ret;
}

ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  static fn_comm_split_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_comm_split_t)must_sym("pncclCommSplit");
  ncclResult_t ret = real_fn(comm, color, key, newcomm, config);
  log_call_ret("CommSplit", ret,
               "\"comm\":\"%p\",\"color\":%d,\"key\":%d,\"newcomm\":\"%p\",\"config\":\"%p\"",
               comm, color, key, newcomm ? *newcomm : NULL, config);
  return ret;
}

ncclResult_t ncclCommInitRankScalable(ncclComm_t *newcomm, int nranks, int myrank, int nId,
                                      ncclUniqueId *commIds, ncclConfig_t *config) {
  static fn_comm_init_rank_scalable_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_comm_init_rank_scalable_t)must_sym("pncclCommInitRankScalable");
  ncclResult_t ret = real_fn(newcomm, nranks, myrank, nId, commIds, config);
  log_call_ret("CommInitRankScalable", ret,
               "\"comm\":\"%p\",\"nranks\":%d,\"rank\":%d,\"n_id\":%d,\"config\":\"%p\"",
               newcomm ? *newcomm : NULL, nranks, myrank, nId, config);
  return ret;
}

const char *ncclGetErrorString(ncclResult_t result) {
  static fn_get_error_string_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_get_error_string_t)must_sym("pncclGetErrorString");
  const char *ret = real_fn(result);
  log_call_ret("GetErrorString", ncclSuccess, "\"result\":%d,\"result_string\":\"%p\"", result, ret);
  return ret;
}

const char *ncclGetLastError(ncclComm_t comm) {
  static fn_get_last_error_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_get_last_error_t)must_sym("pncclGetLastError");
  const char *ret = real_fn(comm);
  log_call_ret("GetLastError", ncclSuccess, "\"comm\":\"%p\",\"result_string\":\"%p\"", comm, ret);
  return ret;
}

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  static fn_comm_get_async_error_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_comm_get_async_error_t)must_sym("pncclCommGetAsyncError");
  ncclResult_t ret = real_fn(comm, asyncError);
  log_call_ret("CommGetAsyncError", ret, "\"comm\":\"%p\",\"async_error\":%d",
               comm, asyncError ? (int)*asyncError : -1);
  return ret;
}

ncclResult_t ncclCommCount(const ncclComm_t comm, int *count) {
  static fn_comm_count_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_comm_count_t)must_sym("pncclCommCount");
  ncclResult_t ret = real_fn(comm, count);
  log_call_ret("CommCount", ret, "\"comm\":\"%p\",\"count\":%d", comm, count ? *count : -1);
  return ret;
}

ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int *device) {
  static fn_comm_device_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_comm_device_t)must_sym("pncclCommCuDevice");
  ncclResult_t ret = real_fn(comm, device);
  log_call_ret("CommCuDevice", ret, "\"comm\":\"%p\",\"device\":%d", comm, device ? *device : -1);
  return ret;
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *rank) {
  static fn_comm_user_rank_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_comm_user_rank_t)must_sym("pncclCommUserRank");
  ncclResult_t ret = real_fn(comm, rank);
  log_call_ret("CommUserRank", ret, "\"comm\":\"%p\",\"rank\":%d", comm, rank ? *rank : -1);
  return ret;
}

ncclResult_t ncclCommRegister(const ncclComm_t comm, void *buff, size_t size, void **handle) {
  static fn_comm_register_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_comm_register_t)must_sym("pncclCommRegister");
  ncclResult_t ret = real_fn(comm, buff, size, handle);
  log_call_ret("CommRegister", ret,
               "\"comm\":\"%p\",\"buff\":\"%p\",\"size\":%zu,\"handle\":\"%p\"",
               comm, buff, size, handle ? *handle : NULL);
  return ret;
}

ncclResult_t ncclCommDeregister(const ncclComm_t comm, void *handle) {
  static fn_comm_deregister_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_comm_deregister_t)must_sym("pncclCommDeregister");
  ncclResult_t ret = real_fn(comm, handle);
  log_call_ret("CommDeregister", ret, "\"comm\":\"%p\",\"handle\":\"%p\"", comm, handle);
  return ret;
}

ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype,
                                      ncclScalarResidence_t residence, ncclComm_t comm) {
  static fn_red_op_create_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_red_op_create_t)must_sym("pncclRedOpCreatePreMulSum");
  ncclResult_t ret = real_fn(op, scalar, datatype, residence, comm);
  log_call_ret("RedOpCreatePreMulSum", ret,
               "\"comm\":\"%p\",\"op\":%d,\"scalar\":\"%p\",\"datatype\":\"%s\","
               "\"datatype_id\":%d,\"residence\":%d",
               comm, op ? (int)*op : -1, scalar, dtype_name(datatype), (int)datatype, (int)residence);
  return ret;
}

ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  static fn_red_op_destroy_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_red_op_destroy_t)must_sym("pncclRedOpDestroy");
  ncclResult_t ret = real_fn(op, comm);
  log_call_ret("RedOpDestroy", ret, "\"comm\":\"%p\",\"op\":%d", comm, (int)op);
  return ret;
}

ncclResult_t ncclGroupStart(void) {
  static fn_group_start_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_group_start_t)must_sym("pncclGroupStart");
  pthread_mutex_lock(&g_lock);
  uint64_t gid = ++g_group_seq;
  pthread_mutex_unlock(&g_lock);
  if (t_group_depth < 16) t_group_stack[t_group_depth] = gid;
  t_group_depth++;
  ncclResult_t ret = real_fn();
  log_call_ret("GroupStart", ret, "\"event_kind\":\"group_start\"");
  return ret;
}

ncclResult_t ncclGroupEnd(void) {
  static fn_group_end_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_group_end_t)must_sym("pncclGroupEnd");
  uint64_t gid = current_group_id();
  ncclResult_t ret = real_fn();
  log_call_ret("GroupEnd", ret, "\"event_kind\":\"group_end\",\"closing_group_id\":%llu",
               (unsigned long long)gid);
  if (t_group_depth > 0) t_group_depth--;
  return ret;
}

ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t *simInfo) {
  static fn_group_simulate_end_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_group_simulate_end_t)must_sym("pncclGroupSimulateEnd");
  ncclResult_t ret = real_fn(simInfo);
  log_call_ret("GroupSimulateEnd", ret, "\"sim_info\":\"%p\"", simInfo);
  return ret;
}

ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm,
                        cudaStream_t stream) {
  static fn_reduce_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_reduce_t)must_sym("pncclReduce");
  ncclResult_t ret = real_fn(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
  size_t bytes = count * dtype_size(datatype);
  log_collective("Reduce", sendbuff, recvbuff, count, bytes, bytes, datatype, 1, op, root,
                 comm, stream, sendbuff == recvbuff, ret);
  return ret;
}

ncclResult_t ncclBcast(void *buff, size_t count, ncclDataType_t datatype, int root,
                       ncclComm_t comm, cudaStream_t stream) {
  static fn_bcast_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_bcast_t)must_sym("pncclBcast");
  ncclResult_t ret = real_fn(buff, count, datatype, root, comm, stream);
  size_t bytes = count * dtype_size(datatype);
  log_collective("Bcast", buff, buff, count, bytes, bytes, datatype, 0, ncclSum, root, comm,
                 stream, 1, ret);
  return ret;
}

ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, int root, ncclComm_t comm,
                           cudaStream_t stream) {
  static fn_broadcast_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_broadcast_t)must_sym("pncclBroadcast");
  ncclResult_t ret = real_fn(sendbuff, recvbuff, count, datatype, root, comm, stream);
  size_t bytes = count * dtype_size(datatype);
  log_collective("Broadcast", sendbuff, recvbuff, count, bytes, bytes, datatype, 0, ncclSum,
                 root, comm, stream, sendbuff == recvbuff, ret);
  return ret;
}

ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                           cudaStream_t stream) {
  static fn_all_reduce_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_all_reduce_t)must_sym("pncclAllReduce");
  ncclResult_t ret = real_fn(sendbuff, recvbuff, count, datatype, op, comm, stream);
  size_t bytes = count * dtype_size(datatype);
  log_collective("AllReduce", sendbuff, recvbuff, count, bytes, bytes, datatype, 1, op, -1,
                 comm, stream, sendbuff == recvbuff, ret);
  return ret;
}

ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                               ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                               cudaStream_t stream) {
  static fn_reduce_scatter_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_reduce_scatter_t)must_sym("pncclReduceScatter");
  ncclResult_t ret = real_fn(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
  int rank, nranks, device;
  comm_info(comm, &rank, &nranks, &device);
  size_t elem = dtype_size(datatype);
  int inplace = 0;
  if (rank >= 0 && elem > 0)
    inplace = ((uintptr_t)recvbuff == (uintptr_t)sendbuff + (uintptr_t)rank * recvcount * elem);
  size_t recv_bytes = recvcount * elem;
  size_t send_bytes = nranks > 0 ? recv_bytes * (size_t)nranks : 0;
  log_collective("ReduceScatter", sendbuff, recvbuff, recvcount, send_bytes, recv_bytes,
                 datatype, 1, op, -1, comm, stream, inplace, ret);
  return ret;
}

ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff, size_t sendcount,
                           ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  static fn_all_gather_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_all_gather_t)must_sym("pncclAllGather");
  ncclResult_t ret = real_fn(sendbuff, recvbuff, sendcount, datatype, comm, stream);
  int rank, nranks, device;
  comm_info(comm, &rank, &nranks, &device);
  size_t elem = dtype_size(datatype);
  int inplace = 0;
  if (rank >= 0 && elem > 0)
    inplace = ((uintptr_t)sendbuff == (uintptr_t)recvbuff + (uintptr_t)rank * sendcount * elem);
  size_t send_bytes = sendcount * elem;
  size_t recv_bytes = nranks > 0 ? send_bytes * (size_t)nranks : 0;
  log_collective("AllGather", sendbuff, recvbuff, sendcount, send_bytes, recv_bytes, datatype,
                 0, ncclSum, -1, comm, stream, inplace, ret);
  return ret;
}

ncclResult_t ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype, int peer,
                      ncclComm_t comm, cudaStream_t stream) {
  static fn_send_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_send_t)must_sym("pncclSend");
  ncclResult_t ret = real_fn(sendbuff, count, datatype, peer, comm, stream);
  size_t bytes = count * dtype_size(datatype);
  log_collective("Send", sendbuff, NULL, count, bytes, 0, datatype, 0, ncclSum, peer, comm,
                 stream, 0, ret);
  return ret;
}

ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype, int peer,
                      ncclComm_t comm, cudaStream_t stream) {
  static fn_recv_t real_fn = NULL;
  if (!real_fn) real_fn = (fn_recv_t)must_sym("pncclRecv");
  ncclResult_t ret = real_fn(recvbuff, count, datatype, peer, comm, stream);
  size_t bytes = count * dtype_size(datatype);
  log_collective("Recv", NULL, recvbuff, count, 0, bytes, datatype, 0, ncclSum, peer, comm,
                 stream, 0, ret);
  return ret;
}
