#ifndef PCCL_INFLIGHT_DRAIN_HPP
#define PCCL_INFLIGHT_DRAIN_HPP

// Registry of live GraphExecutors, used to break the CUDA driver-lock deadlock
// between a CPU-progress-driven PCCL collective and a host-blocking user-thread
// CUDA call (e.g. a pageable D2H from torch's .item() / cudaStreamSynchronize).
//
// PCCL collectives are advanced by a dedicated executor thread that keeps issuing
// CUDA driver calls until completion. If the user thread enters a host-blocking
// CUDA runtime call while a collective is still in flight, that call parks inside
// the driver holding the driver lock, starving the executor thread -> deadlock
// (both threads spin at ~100% CPU).
//
// The interposers in cuda_block_shim.cpp call inflightDrainAll() *before* entering
// the real (blocking) CUDA call. Draining waits on the executor's CPU condition
// variable -- it never touches the driver lock -- so the executor is free to finish
// the collective. Because these are host-blocking sync points anyway, draining here
// costs no compute/communication overlap.

namespace pccl {

class GraphExecutor;

// Register / unregister an executor. Called from GraphExecutor's ctor/dtor.
void inflightRegister(GraphExecutor* exec);
void inflightUnregister(GraphExecutor* exec);

// Drain every registered executor that currently has work in flight. Cheap no-op
// when everything is idle. Must only be called from non-internal (user) threads.
void inflightDrainAll();

// Mark the calling thread as a PCCL-internal thread (executor / H2H worker) so the
// interposers never attempt to drain from within PCCL itself (which would make the
// executor thread wait for its own work and self-deadlock).
void inflightMarkThreadInternal();
bool inflightThreadIsInternal();

}  // namespace pccl

#endif /* PCCL_INFLIGHT_DRAIN_HPP */
