// Deterministic scheduler intervention, entirely in the test executable.
// A completed copy must be consumable after its queue mutex has been released,
// even if the producer is descheduled immediately after that release.
#include <atomic>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <execinfo.h>
#include <iostream>
#include <pthread.h>
#include <thread>
#include <vector>

#include "executor/h2h_engine.hpp"
#include "memory/shared_state.hpp"

namespace {
using Unlock = int (*)(pthread_mutex_t*);
Unlock real_unlock = nullptr;
std::atomic<bool> armed{false}, paused{false}, resume_producer{false};
thread_local bool in_hook = false;
}

extern "C" int pthread_mutex_unlock(pthread_mutex_t* mutex) {
  if (!real_unlock)
    real_unlock = reinterpret_cast<Unlock>(dlsym(RTLD_NEXT, "pthread_mutex_unlock"));
  const int result = real_unlock(mutex);
  if (!in_hook && armed.load(std::memory_order_acquire)) {
    in_hook = true;
    void* frames[16];
    int count = backtrace(frames, 16);
    bool completion_release = false;
    for (int i = 0; i < count; ++i) {
      Dl_info info{};
      if (dladdr(frames[i], &info) && info.dli_sname &&
          std::strstr(info.dli_sname, "pushCompletion"))
        completion_release = true;
    }
    if (completion_release && armed.exchange(false, std::memory_order_acq_rel)) {
      paused.store(true, std::memory_order_release);
      while (!resume_producer.load(std::memory_order_acquire))
        std::this_thread::yield();
    }
    in_hook = false;
  }
  return result;
}

int main() {
  real_unlock = reinterpret_cast<Unlock>(dlsym(RTLD_NEXT, "pthread_mutex_unlock"));
  setenv("PCCL_H2H_WORKERS_PER_NUMA", "1", 1);
  pccl::SharedStateManager state(1);
  pccl::H2HEngine engine(&state, nullptr);
  if (engine.initialize() != 0)
    return 2;
  std::vector<char> src(65536, 42), dst(65536, 0);
  pccl::H2HEngine::CopyRequest request;
  request.owner = &engine;
  request.work_seq = 1;
  request.src = src.data();
  request.dst = dst.data();
  request.bytes = src.size();
  request.temporal_write = true;
  armed.store(true, std::memory_order_release);
  int submitted = engine.submitCopy(request);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!paused.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  bool observed = paused.load(std::memory_order_acquire);
  std::vector<pccl::H2HEngine::Completion> completions;
  size_t consumed = observed ? engine.pollCompletions(&engine, 1, completions) : 0;
  resume_producer.store(true, std::memory_order_release);
  armed.store(false, std::memory_order_release);
  engine.shutdown();
  if (submitted || !observed) {
    std::cerr << "Invalid reproduction: producer was not paused at completion mutex release\n";
    return 2;
  }
  if (consumed != 1 || completions.size() != 1 || dst != src) {
    std::cerr << "FAIL: completed copy invisible after mutex release; consumed=" << consumed << '\n';
    return 1;
  }
  std::cout << "PASS: completion visible while producer paused after mutex release\n";
}
