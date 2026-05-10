#include "memory/completion_flag.hpp"
#include <acl/acl.h>
#include <dlfcn.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

static std::atomic<int> value_waits{0};
extern "C" aclError aclrtValueWait(void* address, uint64_t value, uint32_t flag, aclrtStream stream) {
  using Fn = aclError (*)(void*, uint64_t, uint32_t, aclrtStream);
  static auto real = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "aclrtValueWait"));
  if (!real) std::abort();
  ++value_waits;
  return real(address, value, flag, stream);
}

#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s line=%d\n", #x, __LINE__); return 1; } } while (0)
int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "event";
  setenv("PCCL_COMPLETION_WAIT_MODE", mode, 1);
  setenv("PCCL_EXECUTOR_QUEUE_DEPTH", "2", 1);
  REQUIRE(aclInit(nullptr) == ACL_SUCCESS);
  REQUIRE(aclrtSetDevice(0) == ACL_SUCCESS);
  {
    pccl::CompletionFlag completion;
    int init = completion.initialize();
    if (std::strcmp(mode, "invalid") == 0) {
      REQUIRE(init != 0);
      std::puts("PASS invalid mode rejected");
    } else {
      REQUIRE(init == 0);
      aclrtStream user{};
      aclrtEvent tail{};
      REQUIRE(aclrtCreateStream(&user) == ACL_SUCCESS);
      REQUIRE(aclrtCreateEventWithFlag(&tail, ACL_EVENT_CAPTURE_STREAM_PROGRESS) == ACL_SUCCESS);
      for (uint64_t seq = 1; seq <= 12; ++seq) {
        int slot = (seq - 1) % 2;
        REQUIRE(completion.prepareUserWait(slot, {user}, pccl::CollectiveDebugOp::AllGather,
                                           0, seq, 1024, 0, nullptr, user) == 0);
        REQUIRE(aclrtRecordEvent(tail, user) == ACL_SUCCESS);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        aclrtEventRecordedStatus status;
        REQUIRE(aclrtQueryEventStatus(tail, &status) == ACL_SUCCESS);
        REQUIRE(status == ACL_EVENT_RECORDED_STATUS_NOT_READY);
        REQUIRE(completion.signalComplete(slot, seq) == 0);
        REQUIRE(aclrtSynchronizeStream(user) == ACL_SUCCESS);
      }
      // The experimental contract changes the actual runtime wait primitive,
      // not only its label, while preserving delayed completion and slot reuse.
      REQUIRE(value_waits == (std::strcmp(mode, "event") == 0 ? 0 : 12));
      REQUIRE(aclrtDestroyEvent(tail) == ACL_SUCCESS);
      REQUIRE(aclrtDestroyStream(user) == ACL_SUCCESS);
      std::printf("PASS mode=%s value_waits=%d, 12 rounds with 2 slots\n", mode, value_waits.load());
    }
  }
  REQUIRE(aclrtResetDevice(0) == ACL_SUCCESS);
  REQUIRE(aclFinalize() == ACL_SUCCESS);
}
