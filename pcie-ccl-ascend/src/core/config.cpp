#include "config.hpp"

#include <cctype>
#include <cstdlib>

#include "../util/log.hpp"

namespace pccl {

Config* Config::s_instance = nullptr;

namespace {

constexpr size_t kDefaultHostBufferSize = 8ULL * 1024ULL * 1024ULL * 1024ULL;

size_t parseSizeEnv(const char* value, size_t fallback) {
  if (!value || value[0] == '\0')
    return fallback;
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value)
    return fallback;
  while (*end && std::isspace(static_cast<unsigned char>(*end)))
    ++end;
  unsigned long long multiplier = 1;
  if (*end) {
    switch (std::tolower(static_cast<unsigned char>(*end))) {
      case 'k':
        multiplier = 1024ULL;
        break;
      case 'm':
        multiplier = 1024ULL * 1024ULL;
        break;
      case 'g':
        multiplier = 1024ULL * 1024ULL * 1024ULL;
        break;
      default:
        multiplier = 1;
        break;
    }
  }
  return static_cast<size_t>(parsed * multiplier);
}

}  // namespace

Config Config::load(const pcclConfig_t* config) {
  Config cfg;

  if (config) {
    cfg.nranks = config->nranks;
    cfg.device_count = config->device_count;
    if (config->bootstrap_addr)
      cfg.bootstrap_addr = config->bootstrap_addr;
    cfg.host_buffer_size = config->host_buffer_size;
    if (config->ir_path)
      cfg.ir_path = config->ir_path;
  }

  // Environment variable overrides
  const char* env;
  if ((env = getenv("PCCL_NRANKS")) != nullptr)
    cfg.nranks = atoi(env);
  if ((env = getenv("PCCL_BOOTSTRAP_ADDR")) != nullptr)
    cfg.bootstrap_addr = env;
  if ((env = getenv("PCCL_HOST_BUFFER_SIZE")) != nullptr)
    cfg.host_buffer_size = parseSizeEnv(env, cfg.host_buffer_size);
  if ((env = getenv("PCCL_IR_PATH")) != nullptr)
    cfg.ir_path = env;
  if ((env = getenv("PCCL_IR_DIR")) != nullptr)
    cfg.ir_path = env;

  // Auto-detect if not set
  if (cfg.device_count <= 0)
    cfg.device_count = cfg.nranks;
  if (cfg.host_buffer_size == 0)
    cfg.host_buffer_size = kDefaultHostBufferSize;

  // Discover topology
  const char* explicit_numa_map = nullptr;
  if (config) {
    explicit_numa_map = config->npu_numa_map ? config->npu_numa_map : config->gpu_numa_map;
  }
  if (explicit_numa_map) {
    cfg.topology = parseDeviceNumaMap(explicit_numa_map, cfg.device_count);
  } else {
    cfg.topology = discoverLocalTopology(cfg.device_count);
  }

  LOG_INFO("Config: nranks=%d, devices=%d, numa_nodes=%d, buffer_size=%zu", cfg.nranks, cfg.device_count,
           cfg.topology.num_numa_nodes, cfg.host_buffer_size);

  return cfg;
}

Config& Config::instance() {
  if (!s_instance) {
    static Config default_cfg;
    s_instance = &default_cfg;
  }
  return *s_instance;
}

}  // namespace pccl
