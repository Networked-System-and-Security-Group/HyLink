#include "stream_host_callback.h"
#include "stream_sync.h"
#include "common/log.h"

#if defined(AMPCCL_USE_CUDA_TIMER)
#include <cuda_runtime.h>
#elif defined(AMPCCL_USE_ACL_TIMER)
#include <acl/acl_rt.h>
#endif

namespace ampccl {

namespace {

// `user_data` is the third argument to aclrtLaunchHostFunc / cudaLaunchHostFunc (same as `stream`).
static void HostCompletionFn(void* user_data) {
    // AMPCCL_LOG(DEBUG, "HostCompletionFn begin");
    OnStreamSynchronized(user_data);
    // AMPCCL_LOG(DEBUG, "HostCompletionFn end");
}

}  // namespace

void EnqueueCollectiveCompletionCallback(void* stream) {
    if (!stream) {
        return;
    }
#if defined(AMPCCL_USE_CUDA_TIMER)
    cudaError_t st = cudaLaunchHostFunc(static_cast<cudaStream_t>(stream), HostCompletionFn, stream);
    if (st != cudaSuccess) {
        AMPCCL_LOG(ERROR, "cudaLaunchHostFunc failed: %d", static_cast<int>(st));
    }
#elif defined(AMPCCL_USE_ACL_TIMER)
    aclError st = aclrtLaunchHostFunc(static_cast<aclrtStream>(stream), HostCompletionFn, stream);
    if (st != ACL_SUCCESS) {
        AMPCCL_LOG(ERROR, "aclrtLaunchHostFunc failed: %d", static_cast<int>(st));
    }
#else
    (void)stream;
    AMPCCL_LOG(WARN, "EnqueueCollectiveCompletionCallback: no CUDA/ACL timer backend; skipping host callback");
#endif
}

}  // namespace ampccl
