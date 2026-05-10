#ifndef AMPCCL_CORE_STREAM_SYNC_H_
#define AMPCCL_CORE_STREAM_SYNC_H_

namespace ampccl {

// Called from hooked aclrtSynchronizeStream / cudaStreamSynchronize after the
// original sync. If this stream had a pending collective, syncs PCIe stream
// and domain timers, builds ExecStat, and updates the controller.
void OnStreamSynchronized(void* stream);

void SyncDeviceStreamRaw(void* stream);

}  // namespace ampccl

#endif  // AMPCCL_CORE_STREAM_SYNC_H_
