#include "inflight_drain.hpp"

#include <mutex>
#include <unordered_set>
#include <vector>

#include "../executor/graph_executor.hpp"

namespace pccl {

namespace {

std::mutex& registryMutex() {
  static std::mutex m;
  return m;
}

std::unordered_set<GraphExecutor*>& registry() {
  static std::unordered_set<GraphExecutor*> s;
  return s;
}

// Threads that belong to PCCL itself (executor) must never drain.
thread_local bool t_internal = false;
// Guard against re-entrancy (a drain path should not recursively trigger a drain).
thread_local bool t_draining = false;

}  // namespace

void inflightRegister(GraphExecutor* exec) {
  std::lock_guard<std::mutex> lk(registryMutex());
  registry().insert(exec);
}

void inflightUnregister(GraphExecutor* exec) {
  std::lock_guard<std::mutex> lk(registryMutex());
  registry().erase(exec);
}

void inflightMarkThreadInternal() { t_internal = true; }

bool inflightThreadIsInternal() { return t_internal; }

void inflightDrainAll() {
  if (t_internal || t_draining)
    return;
  t_draining = true;

  // Snapshot the registry so we don't hold its lock while draining (drain() blocks
  // on the executor's condition variable, which can take arbitrarily long).
  std::vector<GraphExecutor*> execs;
  {
    std::lock_guard<std::mutex> lk(registryMutex());
    execs.assign(registry().begin(), registry().end());
  }
  for (GraphExecutor* e : execs) {
    if (e->hasInflight())
      e->drain();
  }

  t_draining = false;
}

}  // namespace pccl
