#include "include/comm.hpp"

#include <memory>
#include <mutex>
#include <vector>

#include "include/error.hpp"
#include "include/hal/device.hpp"
#include "include/host_memory_pool.hpp"
#include "include/ir.hpp"
#include "include/log.hpp"
#include "include/mem_op_manager.hpp"
#include "include/numa_topology.hpp"
#include "include/scheduler.hpp"
#include "include/state_manager.hpp"
#include "include/sync_flags.hpp"

namespace pccl {

// Internal Comm class implementation (hidden from users)
class Comm {
 private:
  int my_rank_;
  int world_size_;
  std::unique_ptr<StateManager> state_manager_;
  std::unique_ptr<HostMemoryPool> host_mem_pool_;
  // Device HAL now uses free functions
  std::unique_ptr<SyncFlagManager> sync_flag_manager_;
  std::unique_ptr<MemoryOperationManager> mem_op_manager_;
  std::unique_ptr<Scheduler> scheduler_;
  static constexpr size_t HOST_BUFFER_SIZE_PER_RANK = 1024ULL * 1024 * 1024;  // 1GB

 public:
  Comm(int rank, int world_size) : my_rank_(rank), world_size_(world_size) {
    LOG_DEBUG("Creating Comm object for rank %d/%d", rank, world_size);
  }

  // Delete copy/move constructors and assignment operators
  Comm(const Comm&) = delete;
  Comm& operator=(const Comm&) = delete;
  Comm(Comm&&) = delete;
  Comm& operator=(Comm&&) = delete;

  ~Comm() {
    LOG_DEBUG("Destroying Comm object for rank %d", my_rank_);

    // Destroy scheduler first (stops executor thread)
    scheduler_.reset();

    // Shutdown memory operation manager
    if (mem_op_manager_) {
      mem_op_manager_->shutdown();
    }
    mem_op_manager_.reset();

    // Destroy sync flag manager
    sync_flag_manager_.reset();

    // Destroy host memory pool
    host_mem_pool_.reset();

    // Destroy state manager
    state_manager_.reset();
  }

  pcclResult_t initStateManager() {
    LOG_DEBUG("Initializing StateManager for rank %d/%d", my_rank_, world_size_);
    try {
      state_manager_.reset(new StateManager(my_rank_, world_size_));
      return pcclSuccess;
    } catch (...) {
      LOG_ERROR("Failed to create StateManager");
      return pcclInternalError;
    }
  }

  pcclResult_t initHostMemoryPool() {
    LOG_DEBUG("Initializing HostMemoryPool for rank %d/%d", my_rank_, world_size_);
    try {
      host_mem_pool_.reset(new HostMemoryPool(my_rank_, world_size_, HOST_BUFFER_SIZE_PER_RANK));
      return pcclSuccess;
    } catch (...) {
      LOG_ERROR("Failed to create HostMemoryPool");
      return pcclInternalError;
    }
  }

  pcclResult_t initialize() {
    LOG_INFO("Initializing Comm for rank %d/%d", my_rank_, world_size_);

    // Validate state manager
    if (!state_manager_ || !state_manager_->isValid()) {
      LOG_ERROR("State manager is null or invalid");
      return pcclInternalError;
    }

    // Validate host memory pool
    if (!host_mem_pool_ || !host_mem_pool_->isValid()) {
      LOG_ERROR("Host memory pool is null or invalid");
      return pcclInternalError;
    }

    // Note: Device initialization (devInit/devSetDevice) is done in pcclInit()
    // before HostMemoryPool creation to ensure device context is available
    // for pinned memory registration

    // Create sync flag manager
    LOG_DEBUG("Creating sync flag manager...");
    sync_flag_manager_.reset(new SyncFlagManager());
    if (!sync_flag_manager_) {
      LOG_ERROR("Failed to allocate sync flag manager");
      return pcclInternalError;
    }
    if (sync_flag_manager_->initialize() != 0) {
      LOG_ERROR("Failed to initialize sync flag manager");
      sync_flag_manager_.reset();
      return pcclInternalError;
    }
    LOG_INFO("Sync flag manager created successfully");

    // Create memory operation manager
    LOG_DEBUG("Creating memory operation manager...");
    mem_op_manager_.reset(new MemoryOperationManager(state_manager_.get(), sync_flag_manager_.get(), my_rank_));
    if (!mem_op_manager_) {
      LOG_ERROR("Failed to allocate memory operation manager");
      return pcclInternalError;
    }

    int init_result = mem_op_manager_->initialize();
    if (init_result != 0) {
      LOG_ERROR("Failed to initialize memory operation manager");
      mem_op_manager_.reset();
      return pcclInternalError;
    }
    LOG_INFO("Memory operation manager created successfully");

    // Create scheduler
    LOG_DEBUG("Creating scheduler...");
    int my_numa_node = NumaTopology::getRankNUMANode(my_rank_);
    scheduler_.reset(new Scheduler(my_rank_, world_size_, my_numa_node, state_manager_.get(), mem_op_manager_.get(),
                                   host_mem_pool_.get(), sync_flag_manager_.get()));
    if (!scheduler_) {
      LOG_ERROR("Failed to allocate scheduler");
      return pcclInternalError;
    }
    LOG_INFO("Scheduler created successfully");

    LOG_INFO("Comm initialization completed successfully for rank %d", my_rank_);
    return pcclSuccess;
  }

  // Getter for StateManager
  StateManager* getStateManager() const { return state_manager_.get(); }

  // Getter for rank and world size
  int getRank() const { return my_rank_; }
  int getWorldSize() const { return world_size_; }

  // Submit IR program
  pcclResult_t submit(IRProgram program, void* dev_sendbuff, void* dev_recvbuff, size_t count, devStream user_stream) {
    return scheduler_->submit(program, dev_sendbuff, dev_recvbuff, count, user_stream);
  }

  // Synchronize internal D2H/H2D streams
  void synchronizeInternalStreams() {
    if (scheduler_) {
      scheduler_->synchronizeInternalStreams();
    }
  }

#if LOG_LEVEL <= 0
  // Print D2H/H2D profiling statistics (call after stream sync)
  void printProfilingStats() {
    if (scheduler_) {
      scheduler_->printProfilingStats();
    }
  }

  // Clear profiling records for next submission
  void clearProfilingRecords() {
    if (scheduler_) {
      scheduler_->clearProfilingRecords();
    }
  }
#endif
};

}  // namespace pccl

// Define pcclComm as pccl::Comm for C++ linkage
struct pcclComm : public pccl::Comm {
  using pccl::Comm::Comm;
};

// C API implementation
extern "C" {

pcclResult_t pcclInit(int rank, int nranks, pcclComm_t* comm) {
  if (!comm) {
    LOG_ERROR("pcclInit: comm parameter is null");
    return pcclInvalidArgument;
  }

  if (rank < 0 || rank >= nranks) {
    LOG_ERROR("pcclInit: invalid rank %d (nranks=%d)", rank, nranks);
    return pcclInvalidArgument;
  }

  if (nranks <= 0 || nranks > MAX_RANKS) {
    LOG_ERROR("pcclInit: invalid nranks %d (max=%d)", nranks, MAX_RANKS);
    return pcclInvalidArgument;
  }

  LOG_INFO("pcclInit called for rank %d/%d", rank, nranks);

  // Create Comm object
  pcclComm_t new_comm = nullptr;
  try {
    new_comm = new pcclComm(rank, nranks);
  } catch (...) {
    LOG_ERROR("Failed to allocate Comm object");
    return pcclInternalError;
  }

  // Initialize device context BEFORE StateManager and HostMemoryPool
  // Single-process mode: StateManager uses devMallocHost (needs active context)
  // Multi-process mode: HostMemoryPool uses devHostRegister (needs active context)
  LOG_DEBUG("Pre-initializing device context for rank %d...", rank);
  static std::once_flag s_dev_init_flag;
  pccl::devStatus dev_init_ret = pccl::devSuccess;
  std::call_once(s_dev_init_flag, [&dev_init_ret]() {
    dev_init_ret = pccl::devInit();
  });
  if (dev_init_ret != pccl::devSuccess) {
    LOG_ERROR("Failed to initialize device: error code %d", dev_init_ret);
    delete new_comm;
    return pcclSystemError;
  }
  pccl::devStatus dev_ret = pccl::devSetDevice(rank);
  if (dev_ret != pccl::devSuccess) {
    LOG_ERROR("Failed to set device %d: error code %d", rank, dev_ret);
    delete new_comm;
    return pcclSystemError;
  }

  // Create StateManager (single-process: devMallocHost; multi-process: shared memory)
  pcclResult_t result = new_comm->initStateManager();
  if (result != pcclSuccess) {
    LOG_ERROR("Failed to initialize StateManager");
    delete new_comm;
    return result;
  }

  // Create HostMemoryPool (single-process: devMallocHost; multi-process: shared memory + devHostRegister)
  result = new_comm->initHostMemoryPool();
  if (result != pcclSuccess) {
    LOG_ERROR("Failed to initialize HostMemoryPool");
    delete new_comm;
    return result;
  }

  // Initialize the communicator (device is already initialized, will skip redundant init)
  result = new_comm->initialize();
  if (result != pcclSuccess) {
    LOG_ERROR("Failed to initialize communicator");
    delete new_comm;
    return result;
  }

  // Wait for all ranks to initialize
  new_comm->getStateManager()->barrierInit();

  *comm = new_comm;
  LOG_INFO("pcclInit completed successfully for rank %d", rank);
  return pcclSuccess;
}

pcclResult_t pcclDestroy(pcclComm_t comm) {
  if (!comm) {
    LOG_ERROR("pcclDestroy: comm parameter is null");
    return pcclInvalidArgument;
  }

  LOG_INFO("pcclDestroy called for comm %p", (void*)comm);

  delete comm;

  LOG_INFO("pcclDestroy completed successfully");
  return pcclSuccess;
}

pcclResult_t pcclCommRank(pcclComm_t comm, int* rank) {
  if (!comm || !rank) {
    return pcclInvalidArgument;
  }
  *rank = comm->getRank();
  return pcclSuccess;
}

pcclResult_t pcclCommSize(pcclComm_t comm, int* size) {
  if (!comm || !size) {
    return pcclInvalidArgument;
  }
  *size = comm->getWorldSize();
  return pcclSuccess;
}

void pcclSynchronizeInternalStreams(pcclComm_t comm) {
  if (comm) {
    comm->synchronizeInternalStreams();
  }
}

}  // extern "C"

// C++ API implementation
pcclResult_t pcclSubmit(pcclComm_t comm, pccl::IRProgram program, void* dev_sendbuff, void* dev_recvbuff, size_t count,
                        pcclStream_t stream) {
  if (!comm) {
    return pcclInvalidArgument;
  }
  pccl::devStream dev_stream = {stream};
  return comm->submit(program, dev_sendbuff, dev_recvbuff, count, dev_stream);
}

#if LOG_LEVEL <= 0
void pcclPrintProfilingStats(pcclComm_t comm) {
  if (comm) {
    comm->printProfilingStats();
  }
}

void pcclClearProfilingRecords(pcclComm_t comm) {
  if (comm) {
    comm->clearProfilingRecords();
  }
}
#endif
