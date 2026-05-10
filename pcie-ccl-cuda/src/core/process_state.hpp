#ifndef PCCL_PROCESS_STATE_HPP
#define PCCL_PROCESS_STATE_HPP

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "limits.hpp"

namespace pccl {

class Config;
class Transport;
class SharedStateManager;
class HostMemoryPool;
class H2HEngine;
struct LocalTopology;

class ProcessState {
 public:
  enum DeviceBufferIpcKind {
    IpcInput = 0,
    IpcOutput = 1,
    IpcInternal = 2,
  };

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
  Transport* transport() { return transport_.get(); }
  const Config* config() const { return config_; }
  bool isMultiProcess() const;
  bool deviceIpcEnabled() const { return device_ipc_needed_; }

  bool registerDeviceBuffers(int comm_id, int rank, int device_id, const void* input, void* output);
  const void* getInputBufferForRank(int rank) const;
  void* getOutputBufferForRank(int rank) const;

  int initializeInternalBuffer(int comm_id, int rank, int device_id);
  int ensureInternalBuffer(int rank, int device_id, size_t bytes);
  void* getInternalBufferForRank(int rank) const;

  bool isRemoteDeviceBufferReady(int comm_id, int local_rank, int kind, int rank) const;
  void* getRemoteDeviceBuffer(int comm_id, int local_rank, int local_device_id, int kind, int rank) const;

 private:
  struct LocalDeviceBufferIpc {
    const void* ptr = nullptr;
    uint64_t base_addr = 0;
    uint64_t offset = 0;
    uint64_t size = 0;
    bool published = false;
    int device_id = -1;
  };

  struct RemoteDeviceBufferIpc {
    void* mapped_base = nullptr;
    void* user_ptr = nullptr;
    uint64_t base_addr = 0;
    uint64_t offset = 0;
    uint64_t size = 0;
    uint64_t seq = 0;
    int local_device_id = -1;
  };

  ProcessState();
  bool publishDeviceBufferLocked(int comm_id, int kind, int rank, const void* ptr, int device_id);
  static void releaseRemoteDeviceBuffer(RemoteDeviceBufferIpc& remote, bool skip_cuda_close = false);

  bool initialized_;
  const Config* config_;

  // True only when the topology can ever select a direct (peer-D2D) IR, i.e. some
  // GPU pair allows direct peer access. On a pure-relay topology (no PCIe-switch
  // P2P), cross-rank transfer always goes through HOST staging and the per-buffer
  // CUDA IPC handles are never opened by any peer -- so we skip publishing them
  // entirely and avoid the heavy per-collective cudaIpcGetMemHandle calls.
  bool device_ipc_needed_ = false;

  std::unique_ptr<SharedStateManager> shared_state_;
  std::unique_ptr<HostMemoryPool> host_pool_;
  std::unique_ptr<H2HEngine> h2h_engine_;
  std::unique_ptr<Transport> transport_;

  std::mutex init_mutex_;
  mutable std::mutex buffer_mutex_;
  std::vector<const void*> rank_input_buffers_;
  std::vector<void*> rank_output_buffers_;
  std::vector<void*> rank_internal_buffers_;
  std::vector<size_t> rank_internal_sizes_;
  std::vector<LocalDeviceBufferIpc> local_ipc_[PCCL_MAX_CONCURRENT_COMMS][3];
  mutable std::vector<RemoteDeviceBufferIpc> remote_ipc_[PCCL_MAX_CONCURRENT_COMMS][3];
};

}  // namespace pccl

#endif /* PCCL_PROCESS_STATE_HPP */
