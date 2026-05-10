#ifndef PCCL_CONFIG_HPP
#define PCCL_CONFIG_HPP

#include <string>

#include "../topology/topology.hpp"
#include "pccl/config.h"

namespace pccl {

class Config {
 public:
  int nranks;
  int device_count;
  std::string bootstrap_addr;
  size_t host_buffer_size;
  std::string ir_path;
  LocalTopology topology;

  // Multi-process support
  bool multi_process = false;
  std::string unique_id;
  int local_rank = 0;

  // Load from pcclConfig_t + environment variables
  static Config load(const pcclConfig_t* config);

  // Get the singleton instance
  static Config& instance();

 private:
  static Config* s_instance;
};

}  // namespace pccl

#endif /* PCCL_CONFIG_HPP */
