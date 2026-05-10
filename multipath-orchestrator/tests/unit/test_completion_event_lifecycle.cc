#include <acl/acl.h>
#include <chrono>
#include <cstdio>
#include <thread>

#define CHECK(call) do { aclError rc = (call); if (rc != ACL_SUCCESS) { \
  std::fprintf(stderr, "%s failed: %d line=%d\n", #call, rc, __LINE__); return 1; } } while (0)

// Device test for the proposed experimental completion notification only.
// No library/default behavior is changed by this executable.
int main() {
  CHECK(aclInit(nullptr));
  CHECK(aclrtSetDevice(0));
  aclrtStream users[2]{}, signals[2]{};
  aclrtEvent completions[2]{}, tails[2]{};
  for (int group = 0; group < 2; ++group) {
    CHECK(aclrtCreateStream(&users[group]));
    CHECK(aclrtCreateStream(&signals[group]));
    CHECK(aclrtCreateEventWithFlag(&completions[group], ACL_EVENT_SYNC));
    CHECK(aclrtCreateEventWithFlag(&tails[group], ACL_EVENT_CAPTURE_STREAM_PROGRESS));
  }
  for (int round = 0; round < 100; ++round) {
    for (int group = 0; group < 2; ++group) {
      CHECK(aclrtStreamWaitEvent(users[group], completions[group]));
      CHECK(aclrtResetEvent(completions[group], users[group]));
      CHECK(aclrtRecordEvent(tails[group], users[group]));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    for (int group = 0; group < 2; ++group) {
      aclrtEventRecordedStatus status;
      CHECK(aclrtQueryEventStatus(tails[group], &status));
      if (status != ACL_EVENT_RECORDED_STATUS_NOT_READY) {
        std::fprintf(stderr, "premature completion group=%d round=%d status=%d\n", group, round, status);
        return 2;
      }
    }
    CHECK(aclrtRecordEvent(completions[0], signals[0]));
    CHECK(aclrtSynchronizeStream(users[0]));
    aclrtEventRecordedStatus untouched;
    CHECK(aclrtQueryEventStatus(tails[1], &untouched));
    if (untouched != ACL_EVENT_RECORDED_STATUS_NOT_READY) {
      std::fprintf(stderr, "cross-instance completion round=%d\n", round);
      return 3;
    }
    CHECK(aclrtRecordEvent(completions[1], signals[1]));
    CHECK(aclrtSynchronizeStream(users[1]));
  }
  for (int group = 0; group < 2; ++group) {
    CHECK(aclrtDestroyEvent(tails[group]));
    CHECK(aclrtDestroyEvent(completions[group]));
    CHECK(aclrtDestroyStream(users[group]));
    CHECK(aclrtDestroyStream(signals[group]));
  }
  CHECK(aclrtResetDevice(0));
  CHECK(aclFinalize());
  std::puts("PASS: 100 reuse rounds, delayed Record, two independent instances");
}
