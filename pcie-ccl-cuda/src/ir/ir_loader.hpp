#ifndef PCCL_IR_LOADER_HPP
#define PCCL_IR_LOADER_HPP

#include <string>

#include "ir.hpp"

namespace pccl {

struct LocalTopology;

class IRLoader {
 public:
  // Load IR program from an XML file for a specific rank
  static IRProgram loadFromFile(const std::string& path, int rank, const LocalTopology* topo = nullptr);

  // Load IR program from an XML string for a specific rank
  static IRProgram loadFromString(const std::string& xml, int rank, const LocalTopology* topo = nullptr);

  // Find the IR XML file for a collective operation.
  // The filename encodes the logical NUMA pattern derived from topo.gpu_to_numa,
  // e.g. [0,0,1,1] -> "allreduce_0011.xml".
  static std::string findIRFile(const std::string& collective, const LocalTopology& topo,
                                const std::string& ir_path = "");

  static std::string findRelayIRFile(const std::string& collective, const LocalTopology& topo,
                                     const std::string& ir_path = "");
};

}  // namespace pccl

#endif /* PCCL_IR_LOADER_HPP */
