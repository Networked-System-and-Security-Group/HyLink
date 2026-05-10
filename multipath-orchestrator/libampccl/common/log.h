#ifndef AMPCCL_COMMON_LOG_H_
#define AMPCCL_COMMON_LOG_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace ampccl {

inline std::mutex& GetLogMutex() {
    static std::mutex m;
    return m;
}

// Log levels: higher value = more verbose.
// Set via AMPCCL_LOG_LEVEL env (0-4 or off/error/warn/info/debug) or SetLogLevel().
enum class LogLevel : int {
    OFF = 0,
    ERROR = 1,
    WARN = 2,
    INFO = 3,
    DEBUG = 4
};

// Global log level. Can be set by env AMPCCL_LOG_LEVEL or by SetLogLevel() at runtime.
inline int& GetLogLevelRef() {
    static int level = -1;
    if (level < 0) {
        const char* env = std::getenv("AMPCCL_LOG_LEVEL");
        if (env == nullptr) {
            level = static_cast<int>(LogLevel::OFF);
        } else if (std::strcmp(env, "off") == 0 || std::strcmp(env, "0") == 0) {
            level = static_cast<int>(LogLevel::OFF);
        } else if (std::strcmp(env, "error") == 0 || std::strcmp(env, "1") == 0) {
            level = static_cast<int>(LogLevel::ERROR);
        } else if (std::strcmp(env, "warn") == 0 || std::strcmp(env, "2") == 0) {
            level = static_cast<int>(LogLevel::WARN);
        } else if (std::strcmp(env, "info") == 0 || std::strcmp(env, "3") == 0) {
            level = static_cast<int>(LogLevel::INFO);
        } else if (std::strcmp(env, "debug") == 0 || std::strcmp(env, "4") == 0) {
            level = static_cast<int>(LogLevel::DEBUG);
        } else {
            level = static_cast<int>(LogLevel::OFF);
        }
    }
    return level;
}

inline LogLevel GetLogLevel() {
    return static_cast<LogLevel>(GetLogLevelRef());
}

inline void SetLogLevel(LogLevel l) {
    GetLogLevelRef() = static_cast<int>(l);
}

inline void SetLogLevel(int l) {
    if (l < 0) l = 0;
    if (l > 4) l = 4;
    GetLogLevelRef() = l;
}

inline const char* LogLevelName(LogLevel l) {
    switch (l) {
        case LogLevel::OFF:   return "OFF";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::WARN:  return "WARN";
        case LogLevel::INFO:  return "INFO";
        case LogLevel::DEBUG: return "DEBUG";
        default: return "?";
    }
}

inline bool LogColorEnabled() {
    static int cached = -1;
    if (cached < 0) {
        if (std::getenv("NO_COLOR") != nullptr) {
            cached = 0;
        } else {
            const char* v = std::getenv("AMPCCL_LOG_COLOR");
            cached = (v == nullptr || (std::strcmp(v, "0") != 0 && std::strcmp(v, "off") != 0)) ? 1 : 0;
        }
    }
    return cached != 0;
}

inline const char* LogLevelColor(LogLevel l) {
    if (!LogColorEnabled()) return "";
    switch (l) {
        case LogLevel::OFF:   return "\033[90m";
        case LogLevel::ERROR: return "\033[1;31m";
        case LogLevel::WARN:  return "\033[1;33m";
        case LogLevel::INFO:  return "\033[32m";
        case LogLevel::DEBUG: return "\033[36m";
        default: return "";
    }
}

inline const char* LogLevelColorReset() {
    return LogColorEnabled() ? "\033[0m" : "";
}

#define AMPCCL_LOG(level, ...) do { \
    if (static_cast<int>(ampccl::LogLevel::level) <= ampccl::GetLogLevelRef()) { \
        std::lock_guard<std::mutex> _ampccl_log_guard(ampccl::GetLogMutex()); \
        const ampccl::LogLevel _ampccl_ll = ampccl::LogLevel::level; \
        std::fprintf(stderr, "%s[AMPCCL][%s]%s ", ampccl::LogLevelColor(_ampccl_ll), ampccl::LogLevelName(_ampccl_ll), ampccl::LogLevelColorReset()); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n"); \
        std::fflush(stderr); \
    } \
} while (0)

}  // namespace ampccl

#endif  // AMPCCL_COMMON_LOG_H_
