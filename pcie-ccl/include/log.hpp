#ifndef PCCL_LOG_HPP
#define PCCL_LOG_HPP

#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>

namespace pccl {
namespace log {

// Log level enumeration
enum class Level { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 };

// Get mutex for thread-safe logging
inline std::mutex& getLogMutex() {
  static std::mutex mtx;
  return mtx;
}

// Check if color output is enabled
inline bool isColorEnabled() {
  static int cached = -1;
  if (cached == -1) {
    // Disable color if NO_COLOR environment variable is set
    cached = (getenv("NO_COLOR") == nullptr) ? 1 : 0;
  }
  return cached == 1;
}

// Get ANSI color code for log level
inline const char* getColorCode(Level level) {
  if (!isColorEnabled()) {
    return "";
  }

  switch (level) {
    case Level::DEBUG:
      return "\033[36m";  // Cyan
    case Level::INFO:
      return "\033[32m";  // Green
    case Level::WARN:
      return "\033[33m";  // Yellow
    case Level::ERROR:
      return "\033[31m";  // Red
    default:
      return "";
  }
}

// Get color reset code
inline const char* getColorReset() {
  return isColorEnabled() ? "\033[0m" : "";
}

// Get level name string
inline const char* getLevelName(Level level) {
  switch (level) {
    case Level::DEBUG:
      return "DEBUG";
    case Level::INFO:
      return "INFO";
    case Level::WARN:
      return "WARN";
    case Level::ERROR:
      return "ERROR";
    default:
      return "UNKNOWN";
  }
}

// Core logging implementation
inline void logImpl(Level level, const char* file, int line, const char* func, const char* fmt, ...) {
  // Lock for thread safety
  std::lock_guard<std::mutex> lock(getLogMutex());

  // Get current time with millisecond precision
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  struct tm tm_info;
  localtime_r(&tv.tv_sec, &tm_info);

  char time_buf[64];
  snprintf(time_buf, sizeof(time_buf), "%04d-%02d-%02d %02d:%02d:%02d.%03d", tm_info.tm_year + 1900, tm_info.tm_mon + 1,
           tm_info.tm_mday, tm_info.tm_hour, tm_info.tm_min, tm_info.tm_sec, static_cast<int>(tv.tv_usec / 1000));

  // Get process and thread IDs
  pid_t pid = getpid();
  pthread_t tid = pthread_self();

  // Extract basename from file path
  const char* basename = strrchr(file, '/');
  basename = basename ? basename + 1 : file;

  // Choose output stream: ERROR to stderr, others to stdout
  FILE* stream = (level == Level::ERROR) ? stderr : stdout;

  // Print log header with color
  fprintf(stream, "%s[%s] [%s]%s [pid:%d|tid:%lu] [%s:%d:%s] ", getColorCode(level), time_buf, getLevelName(level),
          getColorReset(), pid, (unsigned long)tid, basename, line, func);

  // Print user message
  va_list args;
  va_start(args, fmt);
  vfprintf(stream, fmt, args);
  va_end(args);

  // Print newline
  fprintf(stream, "\n");

  // Flush to ensure immediate output
  fflush(stream);
}

}  // namespace log
}  // namespace pccl

// Compile-time log level control
// Define LOG_LEVEL before including this header to control which logs are
// compiled in:
//   -DLOG_LEVEL=0  (DEBUG and above)
//   -DLOG_LEVEL=1  (INFO and above) [default]
//   -DLOG_LEVEL=2  (WARN and above)
//   -DLOG_LEVEL=3  (ERROR only)
#ifndef LOG_LEVEL
#define LOG_LEVEL 1  // Default to INFO level
#endif

// Log macros with compile-time level filtering
#if LOG_LEVEL <= 0
#define LOG_DEBUG(fmt, ...) \
  ::pccl::log::logImpl(::pccl::log::Level::DEBUG, __FILE__, __LINE__, __FUNCTION__, fmt, ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt, ...) ((void)0)
#endif

#if LOG_LEVEL <= 1
#define LOG_INFO(fmt, ...) \
  ::pccl::log::logImpl(::pccl::log::Level::INFO, __FILE__, __LINE__, __FUNCTION__, fmt, ##__VA_ARGS__)
#else
#define LOG_INFO(fmt, ...) ((void)0)
#endif

#if LOG_LEVEL <= 2
#define LOG_WARN(fmt, ...) \
  ::pccl::log::logImpl(::pccl::log::Level::WARN, __FILE__, __LINE__, __FUNCTION__, fmt, ##__VA_ARGS__)
#else
#define LOG_WARN(fmt, ...) ((void)0)
#endif

#if LOG_LEVEL <= 3
#define LOG_ERROR(fmt, ...) \
  ::pccl::log::logImpl(::pccl::log::Level::ERROR, __FILE__, __LINE__, __FUNCTION__, fmt, ##__VA_ARGS__)
#else
#define LOG_ERROR(fmt, ...) ((void)0)
#endif

#endif  // PCCL_LOG_HPP
