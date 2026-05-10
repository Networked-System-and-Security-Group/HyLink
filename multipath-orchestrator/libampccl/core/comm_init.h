#ifndef AMPCCL_CORE_COMM_INIT_H_
#define AMPCCL_CORE_COMM_INIT_H_

#include "domain.h"
#include <cstddef>
#include <vector>
#include "common/log.h"

namespace ampccl {

// Build our Comm identity (CommDomainKey) from NCCL init parameters.
// Same (nranks, commId, rank set) across ranks yields the same key, so
// dividing-param table is keyed by our Comm and can be shared/reused.
inline CommDomainKey BuildKeyFromNcclInit(int nranks,
                                         const void* comm_id_bytes,
                                         size_t comm_id_len,
                                         int rank) {
    CommDomainKey key;
    key.world_size = nranks;
    key.ranks.assign(1, rank);
    key.topology_hash = 0;
    return key;
}

// Build our Comm identity from HCCL init parameters.
inline CommDomainKey BuildKeyFromHcclInit(int nranks,
                                          const void* comm_id_bytes,
                                          size_t comm_id_len,
                                          int rank) {
    return BuildKeyFromNcclInit(nranks, comm_id_bytes, comm_id_len, rank);
}

// Build key for HcclCommInitRootInfo(nRanks, rootInfo, rank, comm).
inline CommDomainKey BuildKeyFromHcclInitRootInfo(int nRanks,
                                                  const void* rootInfo,
                                                  size_t rootInfoLen,
                                                  int rank) {
    (void)rootInfo;
    (void)rootInfoLen;
    CommDomainKey key;
    key.world_size = nRanks;
    key.ranks.assign(1, rank);
    key.topology_hash = 0;
    // if (rootInfo && rootInfoLen > 0) {
    //     const unsigned char* p = static_cast<const unsigned char*>(rootInfo);
    //     for (size_t i = 0; i < rootInfoLen; ++i) {
    //         key.topology_hash = key.topology_hash * 131 + static_cast<uint64_t>(p[i]);
    //     }
    // }
    return key;
}

inline CommDomainKey BuildKeyFromHcclInitAll(int ndev, int rank) {
    CommDomainKey key;
    key.world_size = ndev;
    key.ranks.assign(1, rank);
    key.topology_hash = 0;
    return key;
}

inline CommDomainKey BuildKeyFromNcclInitAll(int ndev, int rank) {
    return BuildKeyFromHcclInitAll(ndev, rank);
}

// PCIe (PCCL) communicator init for this domain. Called from CommInit hook
// after raw comm is created and domain is registered. Implemented in comm_init.cc.
void InitPCIeForDomain(CommDomain* domain, int rank, int nranks);

// PCIe (PCCL) communicator destroy for this domain. Destroys stream then comm via comm.hpp API.
// Safe to call when PCIe was never inited or when ENABLE_PCIE is off (no-op).
void DestroyPCIeForDomain(CommDomain* domain);

}  // namespace ampccl

#endif  // AMPCCL_CORE_COMM_INIT_H_
