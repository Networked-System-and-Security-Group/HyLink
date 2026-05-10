#include "nccl_compat_internal.hpp"

// The NCCL-compatible default path is stream-ordered PCCL graph submission.
// Host collectives were intentionally removed from that path because they
// synchronize the user stream on the host and do not match NCCL enqueue
// semantics.
