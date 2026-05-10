#ifndef PCCL_PROCESS_STATE_HPP
#define PCCL_PROCESS_STATE_HPP

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <unordered_map>

#include "comm_memory.hpp"
#include "limits.hpp"

namespace pccl {

class Config;
class SharedStateManager;
class HostMemoryPool;
class H2HEngine;
struct LocalTopology;

class ProcessState {
 public:
  static ProcessState& instance();

  ~ProcessState();

  // Initialize process-level resources
  int initialize(const Config& config);

  // Finalize and release resources
  void finalize();

  bool isInitialized() const { return initialized_; }

  // Accessors
  SharedStateManager* sharedState() { return shared_state_.get(); }
  HostMemoryPool* hostPool() { return host_pool_.get(); }
  H2HEngine* h2hEngine() { return h2h_engine_.get(); }
  const Config* config() const { return config_; }
  bool isMultiProcess() const;

  bool registerDeviceBuffers(uint64_t comm_hash, int comm_id, int rank, int device_id,
                             const void* input, void* output);
  const void* getInputBufferForComm(uint64_t comm_hash, int comm_id, int rank, int device_id) const;
  void* getOutputBufferForComm(uint64_t comm_hash, int comm_id, int rank, int device_id) const;

  int initializeInternalBuffer(uint64_t comm_hash, int comm_id, int rank, int device_id);
  int ensureInternalBuffer(uint64_t comm_hash, int comm_id, int rank, int device_id, size_t bytes);
  size_t getInternalBufferSize(uint64_t comm_hash, int comm_id, int rank, int device_id) const;
  void* getInternalBufferForComm(uint64_t comm_hash, int comm_id, int rank, int device_id) const;

 private:
  ProcessState();

  bool initialized_;
  const Config* config_;

  std::unique_ptr<SharedStateManager> shared_state_;
  std::unique_ptr<HostMemoryPool> host_pool_;
  std::unique_ptr<H2HEngine> h2h_engine_;

  std::mutex init_mutex_;
  mutable std::mutex buffer_mutex_;
  struct CommBuffers {
    const void* input = nullptr;
    void* output = nullptr;
    void* internal = nullptr;
    size_t internal_size = 0;
  };
  std::unordered_map<CommMemoryKey, CommBuffers, CommMemoryKeyHash> comm_buffers_;
};

}  // namespace pccl

#endif /* PCCL_PROCESS_STATE_HPP */
