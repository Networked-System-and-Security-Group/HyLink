#include <cuda_runtime.h>

#include <cerrno>
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

struct HandleMessage {
  cudaIpcMemHandle_t handle;
  int creator_device;
  size_t bytes;
};

size_t parseSize(const char* s) {
  char* end = nullptr;
  double value = std::strtod(s, &end);
  size_t mult = 1;
  if (end && *end) {
    if (!strcasecmp(end, "k") || !strcasecmp(end, "kb") || !strcasecmp(end, "kib"))
      mult = 1024ULL;
    else if (!strcasecmp(end, "m") || !strcasecmp(end, "mb") || !strcasecmp(end, "mib"))
      mult = 1024ULL * 1024ULL;
    else if (!strcasecmp(end, "g") || !strcasecmp(end, "gb") || !strcasecmp(end, "gib"))
      mult = 1024ULL * 1024ULL * 1024ULL;
  }
  return static_cast<size_t>(value * static_cast<double>(mult));
}

std::vector<int> parseDevices(const char* s) {
  std::vector<int> out;
  std::stringstream ss(s ? s : "0,1,2,3");
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty())
      out.push_back(std::atoi(item.c_str()));
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
    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0)
      return fd;
    usleep(1000);
  }
  perror("connect");
  close(fd);
  return -1;
}

bool writeAll(int fd, const void* data, size_t bytes) {
  const char* p = static_cast<const char*>(data);
  while (bytes > 0) {
    ssize_t n = write(fd, p, bytes);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      perror("write");
      return false;
    }
    p += n;
    bytes -= static_cast<size_t>(n);
  }
  return true;
}

bool readAll(int fd, void* data, size_t bytes) {
  char* p = static_cast<char*>(data);
  while (bytes > 0) {
    ssize_t n = read(fd, p, bytes);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      perror("read");
      return false;
    }
    if (n == 0)
      return false;
    p += n;
    bytes -= static_cast<size_t>(n);
  }
  return true;
}

void printPairAccess(int dst, int src) {
  int can = 0;
  cudaError_t err = cudaDeviceCanAccessPeer(&can, dst, src);
  std::cout << "PAIR dst=" << dst << " src=" << src
            << " cudaDeviceCanAccessPeer=" << (err == cudaSuccess ? can : -1);
  if (err != cudaSuccess)
    std::cout << " err=" << cudaGetErrorString(err);
  std::cout << "\n";
}

int joinerMain(const char* sock_path, int open_device) {
  cudaError_t err = cudaSetDevice(open_device);
  if (err != cudaSuccess) {
    std::cerr << "cudaSetDevice(" << open_device << ") failed: " << cudaGetErrorString(err) << "\n";
    return 10;
  }

  int sock = connectSocket(sock_path);
  if (sock < 0)
    return 11;

  HandleMessage msg {};
  if (!readAll(sock, &msg, sizeof(msg))) {
    std::cerr << "joiner device=" << open_device << " failed to receive handle\n";
    close(sock);
    return 12;
  }
  close(sock);

  printPairAccess(open_device, msg.creator_device);

  void* remote = nullptr;
  err = cudaIpcOpenMemHandle(&remote, msg.handle, cudaIpcMemLazyEnablePeerAccess);
  if (err != cudaSuccess) {
    std::cout << "RESULT creator=" << msg.creator_device << " opener=" << open_device
              << " open=FAIL " << cudaGetErrorString(err) << "\n";
    cudaGetLastError();
    return 20;
  }

  void* local = nullptr;
  err = cudaMalloc(&local, msg.bytes);
  if (err != cudaSuccess) {
    std::cout << "RESULT creator=" << msg.creator_device << " opener=" << open_device
              << " local_malloc=FAIL " << cudaGetErrorString(err) << "\n";
    cudaIpcCloseMemHandle(remote);
    return 21;
  }

  err = cudaMemcpy(local, remote, msg.bytes, cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess) {
    std::cout << "RESULT creator=" << msg.creator_device << " opener=" << open_device
              << " d2d_copy=FAIL " << cudaGetErrorString(err) << "\n";
    cudaFree(local);
    cudaIpcCloseMemHandle(remote);
    cudaGetLastError();
    return 22;
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cout << "RESULT creator=" << msg.creator_device << " opener=" << open_device
              << " sync=FAIL " << cudaGetErrorString(err) << "\n";
    cudaFree(local);
    cudaIpcCloseMemHandle(remote);
    cudaGetLastError();
    return 23;
  }

  std::cout << "RESULT creator=" << msg.creator_device << " opener=" << open_device
            << " open+d2d=OK remote=" << remote << "\n";
  cudaFree(local);
  cudaIpcCloseMemHandle(remote);
  return 0;
}

int creatorMain(int creator_device, const std::vector<int>& opener_devices, size_t bytes) {
  cudaError_t err = cudaSetDevice(creator_device);
  if (err != cudaSuccess) {
    std::cerr << "cudaSetDevice(" << creator_device << ") failed: " << cudaGetErrorString(err) << "\n";
    return 1;
  }

  void* ptr = nullptr;
  err = cudaMalloc(&ptr, bytes);
  if (err != cudaSuccess) {
    std::cerr << "cudaMalloc(" << bytes << ") failed: " << cudaGetErrorString(err) << "\n";
    return 2;
  }
  err = cudaMemset(ptr, 0x5a, bytes);
  if (err != cudaSuccess) {
    std::cerr << "cudaMemset failed: " << cudaGetErrorString(err) << "\n";
    cudaFree(ptr);
    return 3;
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "creator sync failed: " << cudaGetErrorString(err) << "\n";
    cudaFree(ptr);
    return 4;
  }

  HandleMessage msg {};
  msg.creator_device = creator_device;
  msg.bytes = bytes;
  err = cudaIpcGetMemHandle(&msg.handle, ptr);
  if (err != cudaSuccess) {
    std::cerr << "cudaIpcGetMemHandle failed: " << cudaGetErrorString(err) << "\n";
    cudaFree(ptr);
    return 5;
  }

  std::string sock = "/tmp/pccl_cuda_device_ipc_" + std::to_string(getpid()) + ".sock";
  int server = makeServer(sock, static_cast<int>(opener_devices.size()));
  if (server < 0) {
    cudaFree(ptr);
    return 6;
  }

  std::vector<pid_t> children;
  std::string exe = selfExe();
  for (int d : opener_devices) {
    pid_t pid = fork();
    if (pid == 0) {
      std::string dev_arg = std::to_string(d);
      execl(exe.c_str(), exe.c_str(), "--joiner", sock.c_str(), dev_arg.c_str(),
            static_cast<char*>(nullptr));
      perror("execl");
      _exit(127);
    }
    if (pid > 0)
      children.push_back(pid);
  }

  for (size_t i = 0; i < children.size(); ++i) {
    int client = accept(server, nullptr, nullptr);
    if (client < 0) {
      perror("accept");
      continue;
    }
    writeAll(client, &msg, sizeof(msg));
    close(client);
  }

  int failures = 0;
  for (pid_t pid : children) {
    int st = 0;
    waitpid(pid, &st, 0);
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0)
      ++failures;
  }

  close(server);
  unlink(sock.c_str());
  cudaFree(ptr);
  std::cout << "SUMMARY creator=" << creator_device << " failures=" << failures
            << " openers=" << children.size() << "\n";
  return failures == 0 ? 0 : 30 + failures;
}

int sameProcessMain(int creator_device, int open_device, size_t bytes) {
  cudaError_t err = cudaSetDevice(creator_device);
  if (err != cudaSuccess) {
    std::cerr << "cudaSetDevice(" << creator_device << ") failed: " << cudaGetErrorString(err) << "\n";
    return 1;
  }

  void* ptr = nullptr;
  err = cudaMalloc(&ptr, bytes);
  if (err != cudaSuccess) {
    std::cerr << "cudaMalloc(" << bytes << ") failed: " << cudaGetErrorString(err) << "\n";
    return 2;
  }

  cudaIpcMemHandle_t handle {};
  err = cudaIpcGetMemHandle(&handle, ptr);
  if (err != cudaSuccess) {
    std::cerr << "cudaIpcGetMemHandle failed: " << cudaGetErrorString(err) << "\n";
    cudaFree(ptr);
    return 3;
  }

  err = cudaSetDevice(open_device);
  if (err != cudaSuccess) {
    std::cerr << "cudaSetDevice(" << open_device << ") failed: " << cudaGetErrorString(err) << "\n";
    cudaFree(ptr);
    return 4;
  }

  printPairAccess(open_device, creator_device);
  void* remote = nullptr;
  err = cudaIpcOpenMemHandle(&remote, handle, cudaIpcMemLazyEnablePeerAccess);
  if (err != cudaSuccess) {
    std::cout << "RESULT same_process creator=" << creator_device << " opener=" << open_device
              << " open=FAIL " << cudaGetErrorString(err) << "\n";
    cudaGetLastError();
    cudaFree(ptr);
    return 20;
  }

  std::cout << "RESULT same_process creator=" << creator_device << " opener=" << open_device
            << " open=OK remote=" << remote << "\n";
  cudaIpcCloseMemHandle(remote);
  cudaSetDevice(creator_device);
  cudaFree(ptr);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 2 && std::string(argv[1]) == "--joiner") {
    if (argc != 4) {
      std::cerr << "usage: " << argv[0] << " --joiner <sock> <open_device>\n";
      return 2;
    }
    return joinerMain(argv[2], std::atoi(argv[3]));
  }

  int creator = 0;
  int same_process_open_device = -1;
  std::vector<int> devices = parseDevices("0,1,2,3");
  size_t bytes = parseSize("64M");
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--creator" && i + 1 < argc)
      creator = std::atoi(argv[++i]);
    else if (a == "--same-process-open-device" && i + 1 < argc)
      same_process_open_device = std::atoi(argv[++i]);
    else if (a == "--devices" && i + 1 < argc)
      devices = parseDevices(argv[++i]);
    else if (a == "--size" && i + 1 < argc)
      bytes = parseSize(argv[++i]);
    else {
      std::cerr << "usage: " << argv[0]
                << " [--creator DEV] [--devices 0,1,2,3] [--size 64M]\n"
                << "       " << argv[0]
                << " --creator DEV --same-process-open-device DEV [--size 64M]\n";
      return 2;
    }
  }
  if (same_process_open_device >= 0)
    return sameProcessMain(creator, same_process_open_device, bytes);
  return creatorMain(creator, devices, bytes);
}
