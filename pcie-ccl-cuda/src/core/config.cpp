#include "config.hpp"

#include <cstdlib>

#include "../util/log.hpp"

namespace pccl {

Config* Config::s_instance = nullptr;

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
    cfg.host_buffer_size = atol(env);
  if ((env = getenv("PCCL_IR_PATH")) != nullptr)
    cfg.ir_path = env;
  if ((env = getenv("PCCL_IR_DIR")) != nullptr)
    cfg.ir_path = env;

  // Auto-detect if not set
  if (cfg.device_count <= 0)
    cfg.device_count = cfg.nranks;
  if (cfg.host_buffer_size == 0)
    cfg.host_buffer_size = 8 * 1024UL * 1024 * 1024;  // 8GB default

  // Discover topology
  if (config && config->gpu_numa_map) {
    cfg.topology = parseGpuNumaMap(config->gpu_numa_map, cfg.device_count);
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
