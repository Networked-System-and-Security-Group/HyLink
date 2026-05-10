#ifndef PCCL_INFLIGHT_DRAIN_HPP
#define PCCL_INFLIGHT_DRAIN_HPP

// Registry of live GraphExecutors, used to drain CPU-progress-driven PCCL
// collectives before a user thread enters a host-blocking device runtime call.
//
// PCCL collectives are advanced by a dedicated executor thread that keeps issuing
// device runtime calls until completion. Draining waits on the executor's CPU
// condition variable and does not issue device runtime calls itself.
//
namespace pccl {

class GraphExecutor;

// Register / unregister an executor. Called from GraphExecutor's ctor/dtor.
void inflightRegister(GraphExecutor* exec);
void inflightUnregister(GraphExecutor* exec);

// Drain every registered executor that currently has work in flight. Cheap no-op
// when everything is idle. Must only be called from non-internal (user) threads.
void inflightDrainAll();

// Mark the calling thread as a PCCL-internal thread (executor) so the
// interposers never attempt to drain from within PCCL itself (which would make the
// executor thread wait for its own work and self-deadlock).
void inflightMarkThreadInternal();
bool inflightThreadIsInternal();

}  // namespace pccl

#endif /* PCCL_INFLIGHT_DRAIN_HPP */
