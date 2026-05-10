// In-process all-thread backtrace dumper, for environments where ptrace/gdb
// attach is unavailable (CAP_SYS_PTRACE dropped, ptrace_scope=1). Enabled by
// PCCL_SIGDUMP=1. Send SIGUSR1 to a process to dump every thread's user-space
// backtrace to PCCL_SIGDUMP_FILE (default: stderr) -- e.g. to find which thread
// holds the CUDA driver lock during a hang.
//
// Mechanism (all ptrace-free):
//   * A library constructor installs a capture handler on a real-time signal and
//     a trigger handler on SIGUSR1, then spawns a watcher thread.
//   * SIGUSR1 just bumps a flag + writes to a self-pipe (async-signal-safe).
//   * The watcher thread (normal context, so malloc/opendir are safe) enumerates
//     /proc/self/task, sends the RT signal to each thread one at a time, and
//     waits on a semaphore for that thread's handler to record its backtrace.
//   * The RT-signal handler runs in the target thread and calls backtrace()
//     (capturing return addresses only) then posts the semaphore.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dirent.h>
#include <execinfo.h>
#include <fcntl.h>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sys/syscall.h>

namespace {

constexpr int kMaxFrames = 96;

int g_rt_sig = 0;             // chosen at init (SIGRTMIN+5)
int g_pipe[2] = {-1, -1};     // self-pipe: SIGUSR1 handler -> watcher thread
int g_out_fd = 2;             // dump target fd
sem_t g_captured;             // posted by the capture handler
void* g_frames[kMaxFrames];   // single shared slot (dump is serialized)
volatile int g_nframes = 0;

void writeStr(int fd, const char* s) { (void)!write(fd, s, std::strlen(s)); }

void writeTid(int fd, long tid) {
  char buf[32];
  int n = std::snprintf(buf, sizeof(buf), "%ld", tid);
  (void)!write(fd, buf, n);
}

// RT-signal handler: runs in the *target* thread. backtrace() is practically
// safe here; we only stash return addresses and post the semaphore.
void captureHandler(int) {
  g_nframes = backtrace(g_frames, kMaxFrames);
  sem_post(&g_captured);
}

// SIGUSR1 handler: async-signal-safe only -- just kick the watcher.
void triggerHandler(int) {
  char c = 1;
  (void)!write(g_pipe[1], &c, 1);
}

long readSelfTids(long* tids, long max) {
  long count = 0;
  int fd = open("/proc/self/task", O_RDONLY | O_DIRECTORY);
  if (fd < 0)
    return 0;
  DIR* d = fdopendir(fd);
  if (!d) {
    close(fd);
    return 0;
  }
  struct dirent* e;
  while ((e = readdir(d)) != nullptr && count < max) {
    if (e->d_name[0] < '0' || e->d_name[0] > '9')
      continue;
    tids[count++] = std::strtol(e->d_name, nullptr, 10);
  }
  closedir(d);
  return count;
}

void dumpAll() {
  int fd = g_out_fd;
  time_t now = time(nullptr);
  writeStr(fd, "\n===== PCCL SIGDUMP: all-thread backtraces (pid ");
  writeTid(fd, (long)getpid());
  writeStr(fd, ", t=");
  writeTid(fd, (long)now);
  writeStr(fd, ") =====\n");

  long tids[1024];
  long n = readSelfTids(tids, 1024);
  long self = (long)syscall(SYS_gettid);

  for (long i = 0; i < n; ++i) {
    long tid = tids[i];
    writeStr(fd, "--- thread ");
    writeTid(fd, tid);
    // include thread comm name
    char path[64], comm[64] = {0};
    std::snprintf(path, sizeof(path), "/proc/self/task/%ld/comm", tid);
    int cfd = open(path, O_RDONLY);
    if (cfd >= 0) {
      ssize_t r = read(cfd, comm, sizeof(comm) - 1);
      if (r > 0) {
        if (comm[r - 1] == '\n') comm[r - 1] = 0;
        writeStr(fd, " (");
        writeStr(fd, comm);
        writeStr(fd, ")");
      }
      close(cfd);
    }
    writeStr(fd, " ---\n");

    if (tid == self) {
      int nf = backtrace(g_frames, kMaxFrames);
      backtrace_symbols_fd(g_frames, nf, fd);
      continue;
    }

    g_nframes = 0;
    if (syscall(SYS_tgkill, getpid(), tid, g_rt_sig) != 0) {
      writeStr(fd, "  <tgkill failed>\n");
      continue;
    }
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += 2;
    if (sem_timedwait(&g_captured, &ts) != 0) {
      writeStr(fd, "  <no response (running in non-interruptible state?)>\n");
      continue;
    }
    backtrace_symbols_fd(g_frames, g_nframes, fd);
  }
  writeStr(fd, "===== PCCL SIGDUMP end =====\n");
}

void* watcherMain(void*) {
  for (;;) {
    char c;
    ssize_t r = read(g_pipe[0], &c, 1);
    if (r <= 0) {
      if (r < 0 && (errno == EINTR))
        continue;
      break;
    }
    dumpAll();
  }
  return nullptr;
}

__attribute__((constructor)) void sigdumpInit() {
  const char* en = std::getenv("PCCL_SIGDUMP");
  if (!en || en[0] != '1')
    return;

  const char* path = std::getenv("PCCL_SIGDUMP_FILE");
  if (path && path[0]) {
    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd >= 0)
      g_out_fd = fd;
  }

  g_rt_sig = SIGRTMIN + 5;
  sem_init(&g_captured, 0, 0);
  if (pipe2(g_pipe, O_CLOEXEC) != 0)
    return;

  struct sigaction sa;
  std::memset(&sa, 0, sizeof(sa));
  sa.sa_handler = captureHandler;
  sa.sa_flags = SA_RESTART;
  sigfillset(&sa.sa_mask);
  sigaction(g_rt_sig, &sa, nullptr);

  std::memset(&sa, 0, sizeof(sa));
  sa.sa_handler = triggerHandler;
  sa.sa_flags = SA_RESTART;
  sigfillset(&sa.sa_mask);
  sigaction(SIGUSR1, &sa, nullptr);

  pthread_t th;
  pthread_create(&th, nullptr, watcherMain, nullptr);
  pthread_detach(th);

  writeStr(2, "[sigdump] enabled: send SIGUSR1 to dump all-thread backtraces\n");
}

}  // namespace
