#ifndef PCCL_BOOTSTRAP_HPP
#define PCCL_BOOTSTRAP_HPP

#include <string>

namespace pccl {

class Transport;

// Bootstrap multi-node initialization
// Returns nullptr for single-node setups
Transport* bootstrapTransport(const std::string& addr, int rank, int world_size);

}  // namespace pccl

#endif /* PCCL_BOOTSTRAP_HPP */
