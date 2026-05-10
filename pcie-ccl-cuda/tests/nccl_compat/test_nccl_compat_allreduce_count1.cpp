#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

extern "C" {

typedef struct {
  char internal[128];
} ncclUniqueId;

typedef enum {
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2,
  ncclInternalError = 3,
  ncclInvalidArgument = 4,
  ncclInvalidUsage = 5,
  ncclRemoteError = 6,
  ncclInProgress = 7,
  ncclNumResults = 8
} ncclResult_t;

typedef void* ncclComm_t;
typedef enum { ncclSum = 0, ncclProd = 1, ncclMax = 2, ncclMin = 3, ncclAvg = 4 } ncclRedOp_t;
typedef enum {
  ncclInt8 = 0,
  ncclUint8 = 1,
  ncclInt32 = 2,
  ncclUint32 = 3,
  ncclInt64 = 4,
  ncclUint64 = 5,
  ncclFloat16 = 6,
  ncclFloat32 = 7,
  ncclFloat64 = 8,
  ncclBfloat16 = 9
} ncclDataType_t;

ncclResult_t ncclGetVersion(int* version);
ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
ncclResult_t ncclCommDestroy(ncclComm_t comm);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                           ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
const char* ncclGetErrorString(ncclResult_t result);
}

namespace {

#define CHECK_CUDA(call)                                                                 \
  do {                                                                                   \
    cudaError_t _e = (call);                                                             \
    if (_e != cudaSuccess) {                                                             \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                   cudaGetErrorString(_e));                                              \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

#define CHECK_NCCL(call)                                                                 \
  do {                                                                                   \
    ncclResult_t _r = (call);                                                            \
    if (_r != ncclSuccess) {                                                             \
      std::fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__,             \
                   ncclGetErrorString(_r));                                              \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

class Barrier {
 public:
  explicit Barrier(int count) : total_(count), arrived_(0), generation_(0) {}

  void wait() {
    std::unique_lock<std::mutex> lock(mu_);
    int gen = generation_;
    if (++arrived_ == total_) {
      arrived_ = 0;
      ++generation_;
      cv_.notify_all();
      return;
    }
    cv_.wait(lock, [&] { return generation_ != gen; });
  }

 private:
  int total_;
  int arrived_;
  int generation_;
  std::mutex mu_;
  std::condition_variable cv_;
};

struct Args {
  std::string mode = "mp";
  int nranks = 4;
  size_t count = 1;
  bool in_place = true;
  int iters = 1;
};

void usage(const char* prog) {
  std::fprintf(stderr,
               "Usage: %s --mode mp|sp_threads [--nranks N] [--count N] [--in-place 0|1] "
               "[--iters N]\n",
               prog);
}

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    auto needValue = [&](const char* opt) -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", opt);
        usage(argv[0]);
        std::exit(2);
      }
      return argv[++i];
    };
    if (std::strcmp(argv[i], "--mode") == 0) {
      args.mode = needValue(argv[i]);
    } else if (std::strcmp(argv[i], "--nranks") == 0) {
      args.nranks = std::atoi(needValue(argv[i]));
    } else if (std::strcmp(argv[i], "--count") == 0) {
      args.count = static_cast<size_t>(std::strtoull(needValue(argv[i]), nullptr, 10));
    } else if (std::strcmp(argv[i], "--in-place") == 0) {
      args.in_place = std::atoi(needValue(argv[i])) != 0;
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      args.iters = std::atoi(needValue(argv[i]));
    } else {
      std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
      usage(argv[0]);
      std::exit(2);
    }
  }
  if (args.nranks <= 0 || args.count == 0 || args.iters <= 0) {
    usage(argv[0]);
    std::exit(2);
  }
  if (args.mode != "mp" && args.mode != "sp_threads") {
    usage(argv[0]);
    std::exit(2);
  }
  return args;
}

void fillDevice(float* ptr, size_t count, float value, cudaStream_t stream) {
  std::vector<float> host(count, value);
  CHECK_CUDA(cudaMemcpyAsync(ptr, host.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream));
}

bool verifyDevice(int rank, const char* tag, const float* ptr, size_t count, float expected, cudaStream_t stream) {
  std::vector<float> host(count);
  CHECK_CUDA(cudaMemcpyAsync(host.data(), ptr, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  for (size_t i = 0; i < count; ++i) {
    if (std::fabs(host[i] - expected) > 1e-5f) {
      std::fprintf(stderr, "FAIL %s rank=%d idx=%zu got=%f expected=%f\n", tag, rank, i, host[i], expected);
      return false;
    }
  }
  std::printf("PASS %s rank=%d first=%f expected=%f count=%zu\n", tag, rank, host[0], expected, count);
  return true;
}

void checkDeviceCount(int nranks) {
  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count < nranks) {
    std::fprintf(stderr, "need %d CUDA devices, found %d\n", nranks, device_count);
    std::exit(1);
  }
}

struct RankResources {
  ncclComm_t comm = nullptr;
  cudaStream_t stream = nullptr;
  float* send = nullptr;
  float* recv = nullptr;
};

void initRankResources(int rank, const Args& args, ncclUniqueId id, RankResources* res) {
  CHECK_CUDA(cudaSetDevice(rank));

  CHECK_NCCL(ncclCommInitRank(&res->comm, args.nranks, id, rank));
  CHECK_CUDA(cudaStreamCreate(&res->stream));

  CHECK_CUDA(cudaMalloc(&res->send, args.count * sizeof(float)));
  if (args.in_place) {
    res->recv = res->send;
  } else {
    CHECK_CUDA(cudaMalloc(&res->recv, args.count * sizeof(float)));
  }
}

bool runCollectivesForRank(int rank, const Args& args, RankResources* res, Barrier* submit_barrier) {
  CHECK_CUDA(cudaSetDevice(rank));
  const float expected = static_cast<float>(args.nranks) * static_cast<float>(args.nranks + 1) / 2.0f;
  bool ok = true;
  for (int iter = 0; iter < args.iters; ++iter) {
    fillDevice(res->send, args.count, static_cast<float>(rank + 1), res->stream);
    if (!args.in_place)
      fillDevice(res->recv, args.count, -1.0f, res->stream);
    CHECK_CUDA(cudaStreamSynchronize(res->stream));

    if (submit_barrier)
      submit_barrier->wait();

    CHECK_NCCL(ncclAllReduce(res->send, res->recv, args.count, ncclFloat32, ncclSum, res->comm, res->stream));
    CHECK_CUDA(cudaStreamSynchronize(res->stream));

    char tag[128];
    std::snprintf(tag, sizeof(tag), "%s iter=%d %s", args.mode.c_str(), iter,
                  args.in_place ? "in_place" : "out_of_place");
    ok = verifyDevice(rank, tag, res->recv, args.count, expected, res->stream) && ok;
  }
  return ok;
}

void destroyRankResources(const Args& args, RankResources* res) {
  if (!args.in_place)
    CHECK_CUDA(cudaFree(res->recv));
  CHECK_CUDA(cudaFree(res->send));
  CHECK_CUDA(cudaStreamDestroy(res->stream));
  CHECK_NCCL(ncclCommDestroy(res->comm));
}

int runRank(int rank, const Args& args, ncclUniqueId id) {
  checkDeviceCount(args.nranks);
  RankResources res;
  initRankResources(rank, args, id, &res);
  bool ok = runCollectivesForRank(rank, args, &res, nullptr);
  destroyRankResources(args, &res);
  return ok ? 0 : 1;
}

int runMultiProcess(const Args& args) {
  setenv("PCCL_NCCL_COMPAT_MODE", "mp", 1);
  ncclUniqueId id {};
  CHECK_NCCL(ncclGetUniqueId(&id));
  std::fflush(stdout);

  std::vector<pid_t> children;
  children.reserve(args.nranks);
  for (int r = 0; r < args.nranks; ++r) {
    pid_t pid = fork();
    if (pid < 0) {
      perror("fork");
      return 1;
    }
    if (pid == 0) {
      _exit(runRank(r, args, id));
    }
    children.push_back(pid);
  }

  bool ok = true;
  for (pid_t pid : children) {
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
      perror("waitpid");
      ok = false;
      continue;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      std::fprintf(stderr, "child pid=%d failed status=0x%x\n", static_cast<int>(pid), status);
      ok = false;
    }
  }
  return ok ? 0 : 1;
}

int runSingleProcessThreads(const Args& args) {
  setenv("PCCL_NCCL_COMPAT_MODE", "sp", 1);
  checkDeviceCount(args.nranks);
  ncclUniqueId id {};
  CHECK_NCCL(ncclGetUniqueId(&id));

  Barrier submit_barrier(args.nranks);
  std::atomic<int> failures{0};
  std::vector<RankResources> resources(args.nranks);
  for (int r = 0; r < args.nranks; ++r)
    initRankResources(r, args, id, &resources[r]);

  std::vector<std::thread> threads;
  threads.reserve(args.nranks);
  for (int r = 0; r < args.nranks; ++r) {
    threads.emplace_back([&, r] {
      if (!runCollectivesForRank(r, args, &resources[r], &submit_barrier))
        failures.fetch_add(1, std::memory_order_relaxed);
    });
  }
  for (auto& t : threads)
    t.join();

  for (int r = 0; r < args.nranks; ++r) {
    CHECK_CUDA(cudaSetDevice(r));
    destroyRankResources(args, &resources[r]);
  }
  return failures.load(std::memory_order_relaxed) == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  Args args = parseArgs(argc, argv);
  int version = 0;
  CHECK_NCCL(ncclGetVersion(&version));
  std::printf("NCCL compat allreduce count test: version=%d mode=%s nranks=%d count=%zu in_place=%d iters=%d\n",
              version, args.mode.c_str(), args.nranks, args.count, args.in_place ? 1 : 0, args.iters);

  int rc = args.mode == "mp" ? runMultiProcess(args) : runSingleProcessThreads(args);
  std::printf("RESULT mode=%s nranks=%d count=%zu in_place=%d: %s\n",
              args.mode.c_str(), args.nranks, args.count, args.in_place ? 1 : 0, rc == 0 ? "PASS" : "FAIL");
  return rc;
}
