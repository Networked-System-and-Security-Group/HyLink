#include "core/domain.h"
#include "backend/memory/device_mem.h"

namespace ampccl {

CommDomain::~CommDomain() {
    // Comm destruction is required to happen after all user-stream work that
    // may wait on these device tokens has completed.
    value_sync_contexts_.clear();
    FreeDeviceBuffer(v_meta_dev_);
    v_meta_dev_ = nullptr;
    v_meta_cap_ = 0;
}

std::unique_ptr<ValueSyncSubmission> CommDomain::BeginValueSyncSubmission(void* user_stream) {
#if defined(AMPCCL_USE_ACL_TIMER)
    if (!user_stream || !pcie_stream_) {
        return nullptr;
    }
    std::shared_ptr<ValueSyncContext> context;
    {
        std::lock_guard<std::mutex> registry_lock(value_sync_contexts_mutex_);
        auto it = value_sync_contexts_.find(user_stream);
        if (it == value_sync_contexts_.end()) {
            auto created = std::make_shared<ValueSyncContext>();
            if (!created->Initialize()) {
                return nullptr;
            }
            it = value_sync_contexts_.emplace(user_stream, std::move(created)).first;
        }
        context = it->second;
    }
    std::unique_lock<std::mutex> submission_lock(context->mutex());
    ValueSyncTicket ticket = context->NextTicket();
    if (!ticket.input_ready_address || !ticket.completion_address || ticket.sequence == 0) {
        return nullptr;
    }
    return std::make_unique<ValueSyncSubmission>(std::move(context),
                                                  std::move(submission_lock), ticket);
#else
    (void)user_stream;
    return nullptr;
#endif
}

void CommDomain::EnsureVMetaDevCapacity(size_t bytes) {
    if (bytes == 0 || v_meta_cap_ >= bytes) {
        return;
    }
    FreeDeviceBuffer(v_meta_dev_);
    v_meta_dev_ = AllocDeviceBuffer(bytes);
    v_meta_cap_ = v_meta_dev_ ? bytes : 0;
}

bool CommDomain::UploadVMetaHost64(const uint64_t* host, size_t num_uint64) {
    if (!host || num_uint64 == 0) {
        return false;
    }
    const size_t bytes = num_uint64 * sizeof(uint64_t);
    EnsureVMetaDevCapacity(bytes);
    if (!v_meta_dev_) {
        return false;
    }
    DeviceMemcpyH2D(v_meta_dev_, host, bytes);
    return true;
}

}  // namespace ampccl
