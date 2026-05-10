#ifndef PCCL_SHM_ALLOCATOR_HPP
#define PCCL_SHM_ALLOCATOR_HPP

#include <atomic>
#include <cstddef>
#include <string>

#include "../ir/ir.hpp"

namespace pccl {

// Bootstrap shared memory structure for IPC fd exchange between creator and joiners.
// Placed in a small POSIX shm region (shm_open).
// Used by both CUDA (cuMem export fds) and Ascend (memfd fds) paths.
struct BootstrapShm {
  pid_t creator_pid;
  std::atomic<int> ready;              // 1 = creator finished setup
  std::atomic<int> joined_count;       // number of joiners that have attached
  int expected_joiners;                // nranks - 1
  int host_fds[MAX_NUMA_NODES];        // IPC fd numbers per NUMA node (-1 = unused)
  int host_fd_count[MAX_NUMA_NODES];   // number of fds per NUMA node
  size_t host_sizes[MAX_NUMA_NODES];   // allocation size per NUMA node
};

class ShmAllocator {
 public:
#ifdef PCCL_DEVICE_ASCEND
  // Allocate shared hugepage memory via memfd_create(MFD_HUGETLB).
  // name: logical name for the memfd (e.g. "pccl_host_numa6")
  // size: bytes to allocate
  // physical_numa: OS NUMA node for mbind (-1 = no binding)
  // out_fd: if non-null, receives the memfd fd (caller must close when done)
  // Returns mapped pointer, or nullptr on failure.
  static void* allocHugepage(const std::string& name, size_t size, int physical_numa, int* out_fd = nullptr);

  // Unmap shared hugepage memory.
  static void freeHugepage(void* ptr, size_t size);
#endif

  // Create the bootstrap shm region. Called by creator (rank 0).
  // Returns pointer to BootstrapShm.
  static BootstrapShm* createBootstrap(const std::string& unique_id, int nranks);

  // Open existing bootstrap shm region. Called by joiners.
  // Spins until creator sets ready = 1.
  static BootstrapShm* openBootstrap(const std::string& unique_id);

  // Destroy bootstrap shm (called by creator after all ranks done).
  static void destroyBootstrap(const std::string& unique_id, BootstrapShm* bs);

  // Share a memfd fd via bootstrap shm.
  // Creator stores its fd number; joiners open via /proc/<pid>/fd/<fd>.
  // NOTE: Only suitable for memfd/regular fds (Ascend path).
  //       For CUDA cuMem fds, use the UDS-based exchange below.
  static int shareFd(BootstrapShm* bs, int numa_node, int fd_index, int fd);

  // Retrieve a shared fd from another process via /proc/<pid>/fd/<fd>.
  // NOTE: Only suitable for memfd/regular fds (Ascend path).
  static int openSharedFd(BootstrapShm* bs, int numa_node, int fd_index);

  // UDS-based fd exchange via SCM_RIGHTS (required for CUDA cuMem VMM fds).
  // cuMem fds must be duplicated (not reopened) — SCM_RIGHTS achieves this
  // across process boundaries, matching the NCCL IPC pattern.
  //
  // Creator: createFdServer() → [set ready] → serveFds() per joiner → close()
  // Joiner:  [wait ready]     → receiveFds()
  static int createFdServer(const std::string& unique_id, int backlog);
  static int serveFds(int server_fd, const int* fds, int num_fds);
  static int receiveFds(const std::string& unique_id, int* out_fds, int num_fds);

 private:
  static std::string bootstrapName(const std::string& unique_id);
};

}  // namespace pccl

#endif /* PCCL_SHM_ALLOCATOR_HPP */
