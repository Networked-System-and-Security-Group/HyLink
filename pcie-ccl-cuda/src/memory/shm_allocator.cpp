#include "shm_allocator.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#ifdef PCCL_DEVICE_ASCEND
#include <numaif.h>
#include <sys/syscall.h>
#endif

#include "../util/log.hpp"

#ifdef PCCL_DEVICE_ASCEND
// memfd_create flags — define if not provided by headers
#ifndef MFD_HUGETLB
#define MFD_HUGETLB 0x0004U
#endif
#ifndef MFD_HUGE_2MB
#define MFD_HUGE_2MB (21U << 26)
#endif

static int pccl_memfd_create(const char* name, unsigned int flags) {
  return static_cast<int>(syscall(SYS_memfd_create, name, flags));
}
#endif

namespace pccl {

#ifdef PCCL_DEVICE_ASCEND
void* ShmAllocator::allocHugepage(const std::string& name, size_t size, int physical_numa, int* out_fd) {
  long need = static_cast<long>((size + 2 * 1024 * 1024 - 1) / (2 * 1024 * 1024));

  int fd = pccl_memfd_create(name.c_str(), MFD_HUGETLB | MFD_HUGE_2MB);
  if (fd < 0) {
    LOG_ERROR("ShmAllocator: memfd_create(%s, MFD_HUGETLB) failed: %s\n"
              "  Ensure hugepages are reserved on NUMA node %d:\n"
              "    echo %ld > /sys/devices/system/node/node%d/hugepages/hugepages-2048kB/nr_hugepages",
              name.c_str(), strerror(errno), physical_numa, need, physical_numa);
    return nullptr;
  }

  if (ftruncate(fd, static_cast<off_t>(size)) < 0) {
    LOG_ERROR("ShmAllocator: ftruncate(%s, %zu) failed: %s", name.c_str(), size, strerror(errno));
    close(fd);
    return nullptr;
  }

  void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED) {
    LOG_ERROR("ShmAllocator: mmap(%s, %zu) failed: %s\n"
              "  Need %ld free 2MB hugepages on NUMA node %d",
              name.c_str(), size, strerror(errno), need, physical_numa);
    close(fd);
    return nullptr;
  }

  // NUMA bind
  if (physical_numa >= 0) {
    unsigned long nodemask = 1UL << physical_numa;
    if (mbind(p, size, MPOL_BIND, &nodemask, sizeof(nodemask) * 8,
              MPOL_MF_MOVE | MPOL_MF_STRICT) < 0) {
      LOG_ERROR("ShmAllocator: mbind(%s, NUMA %d) failed: %s", name.c_str(), physical_numa, strerror(errno));
      munmap(p, size);
      close(fd);
      return nullptr;
    }
  }

  // First-touch to allocate physical hugepages
  std::memset(p, 0, size);

  if (out_fd) {
    *out_fd = fd;  // Caller takes ownership; must close when done
  } else {
    close(fd);     // Mapping persists after close
  }

  LOG_INFO("ShmAllocator: Allocated %zu bytes (%ld x 2MB) on NUMA %d via memfd '%s' (fd=%d)",
           size, need, physical_numa, name.c_str(), fd);
  return p;
}

void ShmAllocator::freeHugepage(void* ptr, size_t size) {
  if (ptr && ptr != MAP_FAILED) {
    munmap(ptr, size);
  }
}
#endif  // PCCL_DEVICE_ASCEND

std::string ShmAllocator::bootstrapName(const std::string& unique_id) {
  return "/pccl_" + unique_id + "_bootstrap";
}

BootstrapShm* ShmAllocator::createBootstrap(const std::string& unique_id, int nranks) {
  std::string name = bootstrapName(unique_id);

  // Remove stale shm if exists
  shm_unlink(name.c_str());

  int fd = shm_open(name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0666);
  if (fd < 0) {
    LOG_ERROR("ShmAllocator: shm_open(%s, O_CREAT) failed: %s", name.c_str(), strerror(errno));
    return nullptr;
  }

  size_t size = sizeof(BootstrapShm);
  if (ftruncate(fd, static_cast<off_t>(size)) < 0) {
    LOG_ERROR("ShmAllocator: ftruncate bootstrap failed: %s", strerror(errno));
    close(fd);
    shm_unlink(name.c_str());
    return nullptr;
  }

  void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED) {
    LOG_ERROR("ShmAllocator: mmap bootstrap failed: %s", strerror(errno));
    shm_unlink(name.c_str());
    return nullptr;
  }

  auto* bs = static_cast<BootstrapShm*>(p);
  std::memset(bs, 0, sizeof(BootstrapShm));
  bs->creator_pid = getpid();
  bs->expected_joiners = nranks - 1;
  for (int i = 0; i < MAX_NUMA_NODES; i++) {
    bs->host_fds[i] = -1;
    bs->host_fd_count[i] = 0;
    bs->host_sizes[i] = 0;
  }
  // ready is set to 1 by the creator after all setup is done

  LOG_INFO("ShmAllocator: Created bootstrap shm '%s' (pid=%d, nranks=%d)", name.c_str(), bs->creator_pid, nranks);
  return bs;
}

BootstrapShm* ShmAllocator::openBootstrap(const std::string& unique_id) {
  std::string name = bootstrapName(unique_id);

  // Spin until the shm exists and creator marks it ready
  int fd = -1;
  for (int attempt = 0; attempt < 10000; attempt++) {
    fd = shm_open(name.c_str(), O_RDWR, 0666);
    if (fd >= 0) break;
    usleep(1000);  // 1ms
  }
  if (fd < 0) {
    LOG_ERROR("ShmAllocator: shm_open(%s) timed out: %s", name.c_str(), strerror(errno));
    return nullptr;
  }

  size_t size = sizeof(BootstrapShm);
  void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED) {
    LOG_ERROR("ShmAllocator: mmap bootstrap failed: %s", strerror(errno));
    return nullptr;
  }

  auto* bs = static_cast<BootstrapShm*>(p);

  // Wait for creator to finish setup
  while (bs->ready.load(std::memory_order_acquire) == 0) {
    usleep(100);
  }

  LOG_INFO("ShmAllocator: Opened bootstrap shm '%s' (creator_pid=%d)", name.c_str(), bs->creator_pid);
  return bs;
}

void ShmAllocator::destroyBootstrap(const std::string& unique_id, BootstrapShm* bs) {
  if (bs) {
    munmap(bs, sizeof(BootstrapShm));
  }
  std::string name = bootstrapName(unique_id);
  shm_unlink(name.c_str());
  LOG_INFO("ShmAllocator: Destroyed bootstrap shm '%s'", name.c_str());
}

int ShmAllocator::shareFd(BootstrapShm* bs, int numa_node, int fd_index, int fd) {
  // Store the fd number in bootstrap shm — joiners will open via /proc/<pid>/fd/<fd>
  // We encode multiple fds per NUMA by using fd_index as an offset into a flat array.
  // For simplicity, store the first fd per NUMA in host_fds[numa_node].
  // For multiple fds per NUMA, extend BootstrapShm as needed.
  (void)fd_index;
  bs->host_fds[numa_node] = fd;
  return 0;
}

int ShmAllocator::openSharedFd(BootstrapShm* bs, int numa_node, int fd_index) {
  (void)fd_index;
  int remote_fd = bs->host_fds[numa_node];
  if (remote_fd < 0) {
    LOG_ERROR("ShmAllocator: No fd for NUMA %d in bootstrap", numa_node);
    return -1;
  }

  char path[128];
  std::snprintf(path, sizeof(path), "/proc/%d/fd/%d", bs->creator_pid, remote_fd);

  int local_fd = open(path, O_RDWR);
  if (local_fd < 0) {
    LOG_ERROR("ShmAllocator: open(%s) failed: %s", path, strerror(errno));
    return -1;
  }

  LOG_INFO("ShmAllocator: Opened shared fd via %s -> local fd %d", path, local_fd);
  return local_fd;
}

// ---------------------------------------------------------------------------
//  UDS-based fd exchange (SCM_RIGHTS) for CUDA cuMem VMM IPC
// ---------------------------------------------------------------------------
//
// cuMemExportToShareableHandle produces a special fd that must be duplicated
// (not reopened) into the receiving process.  SCM_RIGHTS over a Unix-domain
// socket achieves this — it is the mechanism NCCL uses for the same purpose.
//
// Protocol:
//   Creator:  createFdServer → [set bootstrap ready] → serveFds per joiner
//   Joiner:   [wait bootstrap ready] → receiveFds

static struct sockaddr_un udsAddr(const std::string& unique_id, socklen_t* len) {
  struct sockaddr_un addr = {};
  addr.sun_family = AF_UNIX;
  // Abstract socket (sun_path[0] == '\0') — auto-cleaned, no filesystem entry
  std::string name = "pccl_" + unique_id + "_fd";
  if (name.size() > sizeof(addr.sun_path) - 2)
    name.resize(sizeof(addr.sun_path) - 2);
  std::memcpy(addr.sun_path + 1, name.c_str(), name.size());
  *len = static_cast<socklen_t>(offsetof(struct sockaddr_un, sun_path) + 1 + name.size());
  return addr;
}

int ShmAllocator::createFdServer(const std::string& unique_id, int backlog) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    LOG_ERROR("ShmAllocator: socket(AF_UNIX) failed: %s", strerror(errno));
    return -1;
  }

  socklen_t addrlen;
  struct sockaddr_un addr = udsAddr(unique_id, &addrlen);

  if (bind(fd, reinterpret_cast<struct sockaddr*>(&addr), addrlen) < 0) {
    LOG_ERROR("ShmAllocator: bind(UDS) failed: %s", strerror(errno));
    close(fd);
    return -1;
  }
  if (listen(fd, backlog) < 0) {
    LOG_ERROR("ShmAllocator: listen(UDS) failed: %s", strerror(errno));
    close(fd);
    return -1;
  }

  LOG_INFO("ShmAllocator: UDS fd-server listening (backlog=%d)", backlog);
  return fd;
}

int ShmAllocator::serveFds(int server_fd, const int* fds, int num_fds) {
  int client = accept(server_fd, nullptr, nullptr);
  if (client < 0) {
    LOG_ERROR("ShmAllocator: accept(UDS) failed: %s", strerror(errno));
    return -1;
  }

  // Build sendmsg with SCM_RIGHTS carrying num_fds file descriptors
  char dummy = '\0';
  struct iovec iov = {};
  iov.iov_base = &dummy;
  iov.iov_len = 1;

  size_t cmsg_space = CMSG_SPACE(static_cast<size_t>(num_fds) * sizeof(int));
  std::vector<char> cbuf(cmsg_space, 0);

  struct msghdr msg = {};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = cbuf.data();
  msg.msg_controllen = static_cast<socklen_t>(cmsg_space);

  struct cmsghdr* cm = CMSG_FIRSTHDR(&msg);
  cm->cmsg_level = SOL_SOCKET;
  cm->cmsg_type = SCM_RIGHTS;
  cm->cmsg_len = CMSG_LEN(static_cast<size_t>(num_fds) * sizeof(int));
  std::memcpy(CMSG_DATA(cm), fds, static_cast<size_t>(num_fds) * sizeof(int));

  ssize_t ret = sendmsg(client, &msg, 0);
  close(client);
  if (ret < 0) {
    LOG_ERROR("ShmAllocator: sendmsg(SCM_RIGHTS) failed: %s", strerror(errno));
    return -1;
  }

  LOG_INFO("ShmAllocator: Served %d fd(s) to joiner via UDS", num_fds);
  return 0;
}

int ShmAllocator::receiveFds(const std::string& unique_id, int* out_fds, int num_fds) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    LOG_ERROR("ShmAllocator: socket(AF_UNIX) failed: %s", strerror(errno));
    return -1;
  }

  socklen_t addrlen;
  struct sockaddr_un addr = udsAddr(unique_id, &addrlen);

  // Retry connect — the server may not be listening the instant ready=1 is visible
  for (int attempt = 0; attempt < 10000; attempt++) {
    if (connect(fd, reinterpret_cast<struct sockaddr*>(&addr), addrlen) == 0)
      break;
    if (attempt == 9999) {
      LOG_ERROR("ShmAllocator: connect(UDS) timed out: %s", strerror(errno));
      close(fd);
      return -1;
    }
    usleep(1000);  // 1 ms
  }

  // Receive fds via SCM_RIGHTS
  char dummy;
  struct iovec iov = {};
  iov.iov_base = &dummy;
  iov.iov_len = 1;

  size_t cmsg_space = CMSG_SPACE(static_cast<size_t>(num_fds) * sizeof(int));
  std::vector<char> cbuf(cmsg_space, 0);

  struct msghdr msg = {};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = cbuf.data();
  msg.msg_controllen = static_cast<socklen_t>(cmsg_space);

  ssize_t ret = recvmsg(fd, &msg, 0);
  close(fd);
  if (ret <= 0) {
    LOG_ERROR("ShmAllocator: recvmsg(SCM_RIGHTS) failed: %s", strerror(errno));
    return -1;
  }

  struct cmsghdr* cm = CMSG_FIRSTHDR(&msg);
  if (!cm || cm->cmsg_level != SOL_SOCKET || cm->cmsg_type != SCM_RIGHTS) {
    LOG_ERROR("ShmAllocator: No SCM_RIGHTS in received message");
    return -1;
  }

  std::memcpy(out_fds, CMSG_DATA(cm), static_cast<size_t>(num_fds) * sizeof(int));

  LOG_INFO("ShmAllocator: Received %d fd(s) from creator via UDS", num_fds);
  return 0;
}

}  // namespace pccl
