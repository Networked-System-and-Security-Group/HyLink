#include "tcp_transport.hpp"

#include "../util/log.hpp"

namespace pccl {

TcpTransport::TcpTransport() : connected_(false), rank_(-1), world_size_(0) {}

TcpTransport::~TcpTransport() {
  if (connected_) {
    LOG_INFO("TcpTransport: Disconnecting");
    connected_ = false;
  }
}

int TcpTransport::connect(const std::string& bootstrap_addr, int rank, int world_size) {
  rank_ = rank;
  world_size_ = world_size;
  // TODO: Implement TCP bootstrap connection
  LOG_WARN("TcpTransport: Multi-node transport not yet implemented (addr=%s)", bootstrap_addr.c_str());
  connected_ = false;
  return 0;
}

int TcpTransport::barrier() {
  if (!connected_)
    return -1;
  // TODO: Implement TCP barrier
  return 0;
}

int TcpTransport::readSlice(int remote_node, int numa_id, int chunk_idx, void* local_dst, size_t offset, size_t size) {
  (void)remote_node;
  (void)numa_id;
  (void)chunk_idx;
  (void)local_dst;
  (void)offset;
  (void)size;
  // TODO: Implement TCP read
  return -1;
}

int TcpTransport::writeSlice(int remote_node, int numa_id, int chunk_idx, const void* local_src, size_t offset,
                             size_t size) {
  (void)remote_node;
  (void)numa_id;
  (void)chunk_idx;
  (void)local_src;
  (void)offset;
  (void)size;
  // TODO: Implement TCP write
  return -1;
}

bool TcpTransport::isConnected() const {
  return connected_;
}

}  // namespace pccl
