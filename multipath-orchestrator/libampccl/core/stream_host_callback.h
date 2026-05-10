#ifndef AMPCCL_CORE_STREAM_HOST_CALLBACK_H_
#define AMPCCL_CORE_STREAM_HOST_CALLBACK_H_

namespace ampccl {

// Enqueue a host function on `stream` that runs after all prior work on that stream,
// including cross-stream waits (e.g. StreamWaitEvent for PCIe). Must be called after
// RegisterStreamPending with the same `void* stream` handle.
//
// CANN: aclrtLaunchHostFunc(aclrtStream stream, aclrtHostFunc fn, void *args) with
// typedef void (*aclrtHostFunc)(void *args). We pass `stream` as `args`, so the
// callback receives the user stream pointer and OnStreamSynchronized uses it for
// DomainManager::TakeStreamPending(stream). CUDA: cudaLaunchHostFunc is analogous.
void EnqueueCollectiveCompletionCallback(void* stream);

}  // namespace ampccl

#endif  // AMPCCL_CORE_STREAM_HOST_CALLBACK_H_
