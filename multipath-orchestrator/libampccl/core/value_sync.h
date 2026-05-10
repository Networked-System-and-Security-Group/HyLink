#ifndef AMPCCL_CORE_VALUE_SYNC_H_
#define AMPCCL_CORE_VALUE_SYNC_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace ampccl {

struct ValueSyncTicket {
    void* input_ready_address = nullptr;
    void* completion_address = nullptr;
    uint64_t sequence = 0;
};

class ValueSyncContext {
public:
    static constexpr size_t kRingSize = 64;

    bool Initialize();
    ValueSyncTicket NextTicket();
    std::mutex& mutex() { return mutex_; }
    ~ValueSyncContext();

private:
    void* tokens_ = nullptr;
    uint64_t next_sequence_ = 1;
    std::mutex mutex_;
};

class ValueSyncSubmission {
public:
    ValueSyncSubmission(std::shared_ptr<ValueSyncContext> context,
                        std::unique_lock<std::mutex> lock, ValueSyncTicket ticket);
    ValueSyncSubmission(ValueSyncSubmission&&) noexcept = default;
    ValueSyncSubmission& operator=(ValueSyncSubmission&&) noexcept = default;
    ValueSyncSubmission(const ValueSyncSubmission&) = delete;
    ValueSyncSubmission& operator=(const ValueSyncSubmission&) = delete;

    bool EnqueueInputReady(void* user_stream) const;
    bool EnqueueInputWait(void* pcie_stream) const;
    bool EnqueueWrite(void* pcie_stream) const;
    bool EnqueueWait(void* user_stream) const;
    const ValueSyncTicket& ticket() const { return ticket_; }

private:
    std::shared_ptr<ValueSyncContext> context_;
    std::unique_lock<std::mutex> lock_;
    ValueSyncTicket ticket_;
};

}  // namespace ampccl

#endif  // AMPCCL_CORE_VALUE_SYNC_H_
