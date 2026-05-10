#ifndef PCCL_TCP_TRANSPORT_HPP
#define PCCL_TCP_TRANSPORT_HPP

#include <string>

#include "transport.hpp"

namespace pccl {

class TcpTransport : public Transport {
 public:
  TcpTransport();
  ~TcpTransport() override;

  // Connect to peers
  int connect(const std::string& bootstrap_addr, int rank, int world_size);

  // Transport interface
  int barrier() override;
  int readSlice(int remote_node, int numa_id, int chunk_idx, void* local_dst, size_t offset, size_t size) override;
  int writeSlice(int remote_node, int numa_id, int chunk_idx, const void* local_src, size_t offset,
                 size_t size) override;
  bool isConnected() const override;

 private:
  bool connected_;
  int rank_;
  int world_size_;
};

}  // namespace pccl

#endif /* PCCL_TCP_TRANSPORT_HPP */
