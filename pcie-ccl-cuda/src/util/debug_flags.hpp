#ifndef PCCL_DEBUG_FLAGS_HPP
#define PCCL_DEBUG_FLAGS_HPP

#include <cstdint>
#include <cstdlib>

namespace pccl {

inline bool debugEnvFlagEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value && value[0] != '\0' && std::atoi(value) != 0;
}

inline bool debugCommaListContainsUnsigned(const char* value, uint64_t needle) {
  if (!value || value[0] == '\0')
    return true;
  const char* p = value;
  while (*p) {
    while (*p == ',' || *p == ' ' || *p == '\t')
      ++p;
    if (!*p)
      break;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(p, &end, 10);
    if (end != p && parsed == needle)
      return true;
    p = end && end != p ? end : p + 1;
    while (*p && *p != ',')
      ++p;
  }
  return false;
}

inline bool debugAsyncSubmitEnabled() {
  static int enabled = -1;
  if (enabled < 0) {
    enabled = debugEnvFlagEnabled("PCCL_DEBUG_ASYNC_SUBMIT") ? 1 : 0;
  }
  return enabled != 0;
}

inline bool debugProgressKeysEnabled() {
  static int enabled = -1;
  if (enabled < 0)
    enabled = debugEnvFlagEnabled("PCCL_DEBUG_PROGRESS_KEYS") ? 1 : 0;
  return enabled != 0;
}

inline bool debugProgressHostOnly() {
  static int enabled = -1;
  if (enabled < 0)
    enabled = debugEnvFlagEnabled("PCCL_DEBUG_PROGRESS_HOST_ONLY") ? 1 : 0;
  return enabled != 0;
}

inline bool debugProgressChunkSelected(int chunk) {
  if (chunk < 0)
    return false;
  return debugCommaListContainsUnsigned(std::getenv("PCCL_DEBUG_PROGRESS_CHUNKS"),
                                        static_cast<uint64_t>(chunk));
}

inline bool debugProgressGenerationSelected(uint64_t generation) {
  return debugCommaListContainsUnsigned(std::getenv("PCCL_DEBUG_PROGRESS_GENERATIONS"), generation);
}

}  // namespace pccl

#endif  // PCCL_DEBUG_FLAGS_HPP
