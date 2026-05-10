#include "bootstrap.hpp"

#include "../util/log.hpp"
#include "tcp_transport.hpp"

namespace pccl {

Transport* bootstrapTransport(const std::string& addr, int rank, int world_size) {
  if (addr.empty()) {
    LOG_INFO("Bootstrap: Single-node mode (no transport)");
    return nullptr;
  }

  auto* transport = new TcpTransport();
  transport->connect(addr, rank, world_size);
  return transport;
}

}  // namespace pccl
