#include "core/value_sync.h"

#include "backend/memory/device_mem.h"
#include "common/log.h"

#if defined(AMPCCL_USE_ACL_TIMER)
#include <acl/acl_rt.h>
#endif

#include <limits>

namespace ampccl {

bool ValueSyncContext::Initialize() {
#if defined(AMPCCL_USE_ACL_TIMER)
    const size_t token_bytes = 2 * kRingSize * sizeof(uint64_t);
    tokens_ = AllocDeviceBuffer(token_bytes);
    if (!tokens_) {
        AMPCCL_LOG(ERROR, "ValueSync: token allocation failed bytes=%zu", token_bytes);
        return false;
    }
    aclError ret = aclrtMemset(tokens_, token_bytes, 0, token_bytes);
    if (ret != ACL_SUCCESS) {
        AMPCCL_LOG(ERROR, "ValueSync: token initialization failed ret=%d", static_cast<int>(ret));
        FreeDeviceBuffer(tokens_);
        tokens_ = nullptr;
        return false;
    }
    return true;
#else
    return false;
#endif
}

ValueSyncTicket ValueSyncContext::NextTicket() {
    if (!tokens_ || next_sequence_ == std::numeric_limits<uint64_t>::max()) {
        return {};
    }
    ValueSyncTicket ticket;
    ticket.sequence = next_sequence_++;
    const size_t slot = ticket.sequence % kRingSize;
    ticket.input_ready_address = static_cast<uint64_t*>(tokens_) + slot;
    ticket.completion_address = static_cast<uint64_t*>(tokens_) + kRingSize + slot;
    return ticket;
}

ValueSyncContext::~ValueSyncContext() {
    FreeDeviceBuffer(tokens_);
    tokens_ = nullptr;
}

ValueSyncSubmission::ValueSyncSubmission(std::shared_ptr<ValueSyncContext> context,
                                         std::unique_lock<std::mutex> lock,
                                         ValueSyncTicket ticket)
    : context_(std::move(context)), lock_(std::move(lock)), ticket_(ticket) {}

bool ValueSyncSubmission::EnqueueInputReady(void* user_stream) const {
#if defined(AMPCCL_USE_ACL_TIMER)
    if (!ticket_.input_ready_address || ticket_.sequence == 0 || !user_stream) {
        return false;
    }
    aclError ret = aclrtValueWrite(ticket_.input_ready_address, ticket_.sequence, 0,
                                   static_cast<aclrtStream>(user_stream));
    if (ret != ACL_SUCCESS) {
        AMPCCL_LOG(ERROR, "ValueSync: input-ready write failed ret=%d seq=%llu slot=%zu addr=%p user_stream=%p",
                   static_cast<int>(ret), static_cast<unsigned long long>(ticket_.sequence),
                   static_cast<size_t>(ticket_.sequence % ValueSyncContext::kRingSize),
                   ticket_.input_ready_address, user_stream);
        return false;
    }
    return true;
#else
    (void)user_stream;
    return false;
#endif
}

bool ValueSyncSubmission::EnqueueInputWait(void* pcie_stream) const {
#if defined(AMPCCL_USE_ACL_TIMER)
    if (!ticket_.input_ready_address || ticket_.sequence == 0 || !pcie_stream) {
        return false;
    }
    aclError ret = aclrtValueWait(ticket_.input_ready_address, ticket_.sequence,
                                  ACL_STREAM_WAIT_VALUE_GEQ,
                                  static_cast<aclrtStream>(pcie_stream));
    if (ret != ACL_SUCCESS) {
        AMPCCL_LOG(ERROR, "ValueSync: input-ready wait failed ret=%d seq=%llu slot=%zu addr=%p pcie_stream=%p",
                   static_cast<int>(ret), static_cast<unsigned long long>(ticket_.sequence),
                   static_cast<size_t>(ticket_.sequence % ValueSyncContext::kRingSize),
                   ticket_.input_ready_address, pcie_stream);
        return false;
    }
    return true;
#else
    (void)pcie_stream;
    return false;
#endif
}

bool ValueSyncSubmission::EnqueueWrite(void* pcie_stream) const {
#if defined(AMPCCL_USE_ACL_TIMER)
    if (!ticket_.completion_address || ticket_.sequence == 0 || !pcie_stream) {
        return false;
    }
    aclError ret = aclrtValueWrite(ticket_.completion_address, ticket_.sequence, 0,
                                   static_cast<aclrtStream>(pcie_stream));
    if (ret != ACL_SUCCESS) {
        AMPCCL_LOG(ERROR, "ValueSync: write failed ret=%d seq=%llu slot=%zu addr=%p pcie_stream=%p",
                   static_cast<int>(ret), static_cast<unsigned long long>(ticket_.sequence),
                   static_cast<size_t>(ticket_.sequence % ValueSyncContext::kRingSize),
                   ticket_.completion_address, pcie_stream);
        return false;
    }
    return true;
#else
    (void)pcie_stream;
    return false;
#endif
}

bool ValueSyncSubmission::EnqueueWait(void* user_stream) const {
#if defined(AMPCCL_USE_ACL_TIMER)
    if (!ticket_.completion_address || ticket_.sequence == 0 || !user_stream) {
        return false;
    }
    aclError ret = aclrtValueWait(ticket_.completion_address, ticket_.sequence,
                                  ACL_STREAM_WAIT_VALUE_GEQ,
                                  static_cast<aclrtStream>(user_stream));
    if (ret != ACL_SUCCESS) {
        AMPCCL_LOG(ERROR, "ValueSync: wait failed ret=%d seq=%llu slot=%zu addr=%p user_stream=%p",
                   static_cast<int>(ret), static_cast<unsigned long long>(ticket_.sequence),
                   static_cast<size_t>(ticket_.sequence % ValueSyncContext::kRingSize),
                   ticket_.completion_address, user_stream);
        return false;
    }
    return true;
#else
    (void)user_stream;
    return false;
#endif
}

}  // namespace ampccl
