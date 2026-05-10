#ifndef AMPCCL_CORE_STREAM_SYNC_H_
#define AMPCCL_CORE_STREAM_SYNC_H_

namespace ampccl {

// Called from a host callback enqueued on the user's stream after each collective
// (cudaLaunchHostFunc / aclrtLaunchHostFunc). If this stream had a pending
// collective, merges buffers when needed, and reads timer elapsed values. By
// default it relies on stream ordering: timer_fast is recorded before the
// callback on the user stream, and timer_pcie is waited by the user stream
// before callback registration when PCIe is used. If timer event synchronization
// is enabled, the callback waits for a normal host worker thread to synchronize
// those events; the runtime callback thread itself does not execute blocking
// event synchronize calls. Never SyncDeviceStreamRaw(user_stream) here
// (LaunchHostFunc deadlock). AllGather merge uses sync D2D.
void OnStreamSynchronized(void* stream);

void SyncDeviceStreamRaw(void* stream);

}  // namespace ampccl

#endif  // AMPCCL_CORE_STREAM_SYNC_H_
