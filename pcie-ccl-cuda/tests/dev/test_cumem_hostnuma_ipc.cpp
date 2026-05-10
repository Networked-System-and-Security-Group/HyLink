#include <cuda.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

std::string cuerr(CUresult r) {
  const char* name = nullptr;
  const char* str = nullptr;
  cuGetErrorName(r, &name);
  cuGetErrorString(r, &str);
  std::ostringstream os;
  os << (name ? name : "?") << " / " << (str ? str : "?") << " (" << static_cast<int>(r) << ")";
  return os.str();
}

bool check(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return true;
  std::cerr << what << " failed: " << cuerr(r) << "\n";
  return false;
}

size_t parseSize(const char* s) {
  char* end = nullptr;
  double value = std::strtod(s, &end);
  size_t mult = 1;
  if (end && *end) {
    if (!strcasecmp(end, "k") || !strcasecmp(end, "kb") || !strcasecmp(end, "kib")) mult = 1024ULL;
    else if (!strcasecmp(end, "m") || !strcasecmp(end, "mb") || !strcasecmp(end, "mib")) mult = 1024ULL * 1024ULL;
    else if (!strcasecmp(end, "g") || !strcasecmp(end, "gb") || !strcasecmp(end, "gib")) mult = 1024ULL * 1024ULL * 1024ULL;
  }
  return static_cast<size_t>(value * static_cast<double>(mult));
}

std::vector<int> parseDevices(const char* s) {
  std::vector<int> out;
  std::stringstream ss(s ? s : "0,1,2,3");
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(std::atoi(item.c_str()));
  }
  return out;
}

std::string selfExe() {
  char buf[4096];
  ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n < 0) {
    perror("readlink(/proc/self/exe)");
    std::exit(2);
  }
  buf[n] = '\0';
  return std::string(buf);
}

int makeServer(const std::string& path, int backlog) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    perror("socket");
    return -1;
  }
  sockaddr_un addr {};
  addr.sun_family = AF_UNIX;
  std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path.c_str());
  unlink(path.c_str());
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    perror("bind");
    close(fd);
    return -1;
  }
  if (listen(fd, backlog) < 0) {
    perror("listen");
    close(fd);
    return -1;
  }
  return fd;
}

int connectSocket(const std::string& path) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    perror("socket");
    return -1;
  }
  sockaddr_un addr {};
  addr.sun_family = AF_UNIX;
  std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path.c_str());
  for (int i = 0; i < 5000; ++i) {
    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) return fd;
    usleep(1000);
  }
  perror("connect");
  close(fd);
  return -1;
}

bool sendFd(int sock, int fd_to_send) {
  char byte = 'x';
  iovec iov {&byte, 1};
  char ctrl[CMSG_SPACE(sizeof(int))];
  std::memset(ctrl, 0, sizeof(ctrl));

  msghdr msg {};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = ctrl;
  msg.msg_controllen = sizeof(ctrl);

  cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(cmsg), &fd_to_send, sizeof(int));

  if (sendmsg(sock, &msg, 0) < 0) {
    perror("sendmsg");
    return false;
  }
  return true;
}

int recvFd(int sock) {
  char byte = 0;
  iovec iov {&byte, 1};
  char ctrl[CMSG_SPACE(sizeof(int))];
  std::memset(ctrl, 0, sizeof(ctrl));

  msghdr msg {};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = ctrl;
  msg.msg_controllen = sizeof(ctrl);

  if (recvmsg(sock, &msg, 0) < 0) {
    perror("recvmsg");
    return -1;
  }
  cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  if (!cmsg || cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) return -1;
  int fd = -1;
  std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
  return fd;
}

void printDeviceAttrs(int device) {
  CUdevice dev;
  if (!check(cuDeviceGet(&dev, device), "cuDeviceGet")) return;
  int vmm = 0, fd = 0, host_numa = -2, fabric = 0;
  cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev);
  cuDeviceGetAttribute(&fd, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, dev);
  cuDeviceGetAttribute(&host_numa, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev);
#ifdef CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED
  cuDeviceGetAttribute(&fabric, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, dev);
#endif
  std::cout << "device=" << device << " attrs:"
            << " vmm=" << vmm
            << " posix_fd=" << fd
            << " host_numa=" << host_numa
            << " fabric=" << fabric << "\n";
}

int joinerMain(const char* sock_path, int device, size_t size, int host_numa) {
  if (!check(cuInit(0), "cuInit")) return 10;
  printDeviceAttrs(device);

  CUdevice dev;
  if (!check(cuDeviceGet(&dev, device), "cuDeviceGet")) return 11;
  CUcontext ctx;
  if (!check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain")) return 12;
  if (!check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")) return 13;

  int sock = connectSocket(sock_path);
  if (sock < 0) return 14;
  int fd = recvFd(sock);
  close(sock);
  if (fd < 0) {
    std::cerr << "joiner device=" << device << " failed to receive fd\n";
    return 15;
  }

  CUmemGenericAllocationHandle handle;
  CUresult r = cuMemImportFromShareableHandle(
      &handle, reinterpret_cast<void*>(static_cast<uintptr_t>(fd)),
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  close(fd);
  if (r != CUDA_SUCCESS) {
    std::cout << "RESULT device=" << device << " import=FAIL " << cuerr(r) << "\n";
    return 20;
  }

  CUdeviceptr dptr = 0;
  r = cuMemAddressReserve(&dptr, size, 0, 0, 0);
  if (r != CUDA_SUCCESS) {
    std::cout << "RESULT device=" << device << " reserve=FAIL " << cuerr(r) << "\n";
    cuMemRelease(handle);
    return 21;
  }

  r = cuMemMap(dptr, size, 0, handle, 0);
  if (r != CUDA_SUCCESS) {
    std::cout << "RESULT device=" << device << " map=FAIL " << cuerr(r) << "\n";
    cuMemAddressFree(dptr, size);
    cuMemRelease(handle);
    return 22;
  }

  CUmemAccessDesc access {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  r = cuMemSetAccess(dptr, size, &access, 1);
  if (r != CUDA_SUCCESS) {
    std::cout << "RESULT device=" << device << " set_device_access=FAIL " << cuerr(r) << "\n";
    cuMemUnmap(dptr, size);
    cuMemAddressFree(dptr, size);
    cuMemRelease(handle);
    return 23;
  }

  access.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  access.location.id = host_numa;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  r = cuMemSetAccess(dptr, size, &access, 1);
  if (r != CUDA_SUCCESS) {
    std::cout << "RESULT device=" << device << " set_host_access=FAIL " << cuerr(r) << "\n";
    cuMemUnmap(dptr, size);
    cuMemAddressFree(dptr, size);
    cuMemRelease(handle);
    return 24;
  }

  std::cout << "RESULT device=" << device << " import+map+access=OK ptr=0x"
            << std::hex << static_cast<unsigned long long>(dptr) << std::dec << "\n";
  cuMemUnmap(dptr, size);
  cuMemAddressFree(dptr, size);
  cuMemRelease(handle);
  return 0;
}

int creatorMain(int creator_device, const std::vector<int>& joiner_devices, size_t requested_size, int host_numa_arg) {
  if (!check(cuInit(0), "cuInit")) return 1;
  for (int d : joiner_devices) printDeviceAttrs(d);

  CUdevice dev;
  if (!check(cuDeviceGet(&dev, creator_device), "cuDeviceGet")) return 2;
  CUcontext ctx;
  if (!check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain")) return 3;
  if (!check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent")) return 4;

  int creator_host_numa = 0;
  cuDeviceGetAttribute(&creator_host_numa, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev);
  if (creator_host_numa < 0) creator_host_numa = 0;
  int host_numa = host_numa_arg >= 0 ? host_numa_arg : creator_host_numa;

  CUmemAllocationProp prop {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.location.id = host_numa;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  size_t granularity = 0;
  if (!check(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
             "cuMemGetAllocationGranularity")) return 5;
  size_t size = (requested_size + granularity - 1) / granularity * granularity;
  std::cout << "creator_device=" << creator_device << " host_numa=" << host_numa
            << " requested_size=" << requested_size << " rounded_size=" << size
            << " granularity=" << granularity << "\n";

  CUmemGenericAllocationHandle handle;
  if (!check(cuMemCreate(&handle, size, &prop, 0), "cuMemCreate")) return 6;

  CUdeviceptr dptr = 0;
  if (!check(cuMemAddressReserve(&dptr, size, granularity, 0, 0), "cuMemAddressReserve")) return 7;
  if (!check(cuMemMap(dptr, size, 0, handle, 0), "cuMemMap creator")) return 8;

  CUmemAccessDesc access {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = creator_device;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if (!check(cuMemSetAccess(dptr, size, &access, 1), "cuMemSetAccess creator device")) return 9;
  access.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  access.location.id = host_numa;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if (!check(cuMemSetAccess(dptr, size, &access, 1), "cuMemSetAccess creator host")) return 10;

  int share_fd = -1;
  if (!check(cuMemExportToShareableHandle(&share_fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0),
             "cuMemExportToShareableHandle")) return 11;

  std::string sock = "/tmp/pccl_cumem_hostnuma_ipc_" + std::to_string(getpid()) + ".sock";
  int server = makeServer(sock, static_cast<int>(joiner_devices.size()));
  if (server < 0) return 12;

  std::vector<pid_t> children;
  std::string exe = selfExe();
  std::string size_arg = std::to_string(size);
  std::string numa_arg = std::to_string(host_numa);
  for (int d : joiner_devices) {
    pid_t pid = fork();
    if (pid == 0) {
      std::string dev_arg = std::to_string(d);
      execl(exe.c_str(), exe.c_str(), "--joiner", sock.c_str(), dev_arg.c_str(),
            size_arg.c_str(), numa_arg.c_str(), static_cast<char*>(nullptr));
      perror("execl");
      _exit(127);
    }
    if (pid > 0) children.push_back(pid);
  }

  for (size_t i = 0; i < children.size(); ++i) {
    int client = accept(server, nullptr, nullptr);
    if (client < 0) {
      perror("accept");
      continue;
    }
    sendFd(client, share_fd);
    close(client);
  }

  int failures = 0;
  for (pid_t pid : children) {
    int st = 0;
    waitpid(pid, &st, 0);
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) ++failures;
  }

  close(server);
  close(share_fd);
  unlink(sock.c_str());
  cuMemUnmap(dptr, size);
  cuMemAddressFree(dptr, size);
  cuMemRelease(handle);
  std::cout << "SUMMARY failures=" << failures << " joiners=" << children.size() << "\n";
  return failures == 0 ? 0 : 30 + failures;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && std::string(argv[1]) == "--joiner") {
    if (argc != 6) {
      std::cerr << "usage: " << argv[0] << " --joiner <sock> <device> <size> <host_numa>\n";
      return 2;
    }
    return joinerMain(argv[2], std::atoi(argv[3]), parseSize(argv[4]), std::atoi(argv[5]));
  }

  int creator = 0;
  std::vector<int> devices = parseDevices("0,1,2,3");
  size_t size = parseSize("64M");
  int host_numa = -1;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--creator" && i + 1 < argc) creator = std::atoi(argv[++i]);
    else if (a == "--devices" && i + 1 < argc) devices = parseDevices(argv[++i]);
    else if (a == "--size" && i + 1 < argc) size = parseSize(argv[++i]);
    else if (a == "--host-numa" && i + 1 < argc) host_numa = std::atoi(argv[++i]);
    else {
      std::cerr << "usage: " << argv[0]
                << " [--creator DEV] [--devices 0,1,2,3] [--size 64M] [--host-numa N]\n";
      return 2;
    }
  }
  return creatorMain(creator, devices, size, host_numa);
}
