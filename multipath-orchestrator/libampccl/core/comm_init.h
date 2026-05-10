#ifndef AMPCCL_CORE_COMM_INIT_H_
#define AMPCCL_CORE_COMM_INIT_H_

#include "core/logical_comm_key.h"
#include "core/domain_key.h"
#include <cstddef>
#include <cstdint>
#include <vector>
#include "common/log.h"

namespace ampccl {

class CommDomain;

inline uint64_t HashBytesForComm(const void* data, size_t len, uint64_t seed = 1469598103934665603ULL) {
    uint64_t h = seed;
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; p != nullptr && i < len; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ULL;
    }
    h ^= static_cast<uint64_t>(len) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

inline LogicalCommKey BuildLogicalKeyFromBootstrap(int nranks, const void* bootstrap, size_t len,
                                                   uint64_t fallback_tag) {
    LogicalCommKey key;
    key.world_size = nranks;
    if (bootstrap && len > 0) {
        key.comm_hash = HashBytesForComm(bootstrap, len);
    } else {
        key.comm_hash = fallback_tag ^ (static_cast<uint64_t>(nranks) * 0x9e3779b97f4a7c15ULL);
    }
    return key;
}

inline LogicalCommKey BuildLogicalKeyFromNcclInit(int nranks, const void* comm_id_bytes, size_t comm_id_len) {
    return BuildLogicalKeyFromBootstrap(nranks, comm_id_bytes, comm_id_len, 0x4e43434c5f494449ULL);
}

inline LogicalCommKey BuildLogicalKeyFromHcclInitRootInfo(int nranks, const void* root_info, size_t root_info_len) {
    return BuildLogicalKeyFromBootstrap(nranks, root_info, root_info_len, 0x4843434c5f494449ULL);
}

inline LogicalCommKey BuildLogicalKeyFromInitAll(int ndev) {
    return BuildLogicalKeyFromBootstrap(ndev, nullptr, 0, 0x494e49545f414c4cULL);
}

inline LogicalCommKey BuildLogicalKeyFromInitAll(int ndev, uint64_t local_nonce) {
    uint64_t data[2] = {static_cast<uint64_t>(ndev), local_nonce};
    return BuildLogicalKeyFromBootstrap(ndev, data, sizeof(data), 0x494e49545f414c4cULL);
}

inline LogicalCommKey BuildLogicalKeyFromCommSplit(const LogicalCommKey& parent,
                                                   int color,
                                                   uint64_t split_ordinal,
                                                   int child_nranks) {
    uint64_t data[5] = {
        parent.comm_hash,
        static_cast<uint64_t>(parent.world_size),
        static_cast<uint64_t>(child_nranks),
        static_cast<uint64_t>(static_cast<int64_t>(color)),
        split_ordinal,
    };
    return BuildLogicalKeyFromBootstrap(child_nranks, data, sizeof(data), 0x53504c49545f4e43ULL);
}

inline LogicalCommKey BuildLogicalKeyFromHcclSubComm(const LogicalCommKey& parent,
                                                     uint32_t rank_num,
                                                     const uint32_t* rank_ids,
                                                     uint64_t sub_comm_id) {
    LogicalCommKey key;
    key.world_size = static_cast<int>(rank_num);
    uint64_t data[4] = {
        parent.comm_hash,
        static_cast<uint64_t>(parent.world_size),
        static_cast<uint64_t>(rank_num),
        sub_comm_id,
    };
    uint64_t h = HashBytesForComm(data, sizeof(data), 0x4843434c5f535542ULL);
    if (rank_ids != nullptr && rank_num > 0) {
        h = HashBytesForComm(rank_ids, sizeof(uint32_t) * static_cast<size_t>(rank_num), h);
    }
    key.comm_hash = h;
    return key;
}

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
    key.topology_hash = BuildLogicalKeyFromNcclInit(nranks, comm_id_bytes, comm_id_len).comm_hash;
    return key;
}

// Build key for HcclCommInitRootInfo(nRanks, rootInfo, rank, comm).
inline CommDomainKey BuildKeyFromHcclInitRootInfo(int nRanks,
                                                  const void* rootInfo,
                                                  size_t rootInfoLen,
                                                  int rank) {
    CommDomainKey key;
    key.world_size = nRanks;
    key.ranks.assign(1, rank);
    key.topology_hash = BuildLogicalKeyFromHcclInitRootInfo(nRanks, rootInfo, rootInfoLen).comm_hash;
    return key;
}

inline CommDomainKey BuildKeyFromHcclInitAll(int ndev, int rank) {
    CommDomainKey key;
    key.world_size = ndev;
    key.ranks.assign(1, rank);
    key.topology_hash = BuildLogicalKeyFromInitAll(ndev).comm_hash;
    return key;
}

inline CommDomainKey BuildKeyFromNcclInitAll(int ndev, int rank) {
    return BuildKeyFromHcclInitAll(ndev, rank);
}

inline CommDomainKey BuildKeyFromNcclCommSplit(const LogicalCommKey& parent,
                                               int color,
                                               uint64_t split_ordinal,
                                               int child_nranks,
                                               int child_rank) {
    CommDomainKey key;
    key.world_size = child_nranks;
    key.ranks.assign(1, child_rank);
    key.topology_hash =
        BuildLogicalKeyFromCommSplit(parent, color, split_ordinal, child_nranks).comm_hash;
    return key;
}

inline CommDomainKey BuildKeyFromHcclSubComm(const LogicalCommKey& parent,
                                             uint32_t rank_num,
                                             const uint32_t* rank_ids,
                                             uint64_t sub_comm_id,
                                             uint32_t sub_comm_rank_id) {
    CommDomainKey key;
    key.world_size = static_cast<int>(rank_num);
    key.ranks.assign(1, static_cast<int>(sub_comm_rank_id));
    key.topology_hash =
        BuildLogicalKeyFromHcclSubComm(parent, rank_num, rank_ids, sub_comm_id).comm_hash;
    return key;
}

// PCIe (PCCL) communicator init for this domain. Called from CommInit hook
// after raw comm is created and domain is registered. Implemented in comm_init.cc.
// pccl_bootstrap: NCCL ncclUniqueId bytes, or HCCL HcclRootInfo (hashed if longer than pccl id);
// nullptr / pccl_bootstrap_len==0 => zero pcclUniqueId (single-process pccl mode).
void InitPCIeForDomain(CommDomain* domain, int rank, int nranks, const void* pccl_bootstrap,
                       size_t pccl_bootstrap_len);

// Complete a deferred PCIe/PCCL initialization recorded on a domain. This is
// used for communicators created by ncclCommSplit so the hook does not inject a
// new multi-process PCCL bootstrap before NCCL returns the child communicator to
// the framework.
void EnsurePCIeForDomain(CommDomain* domain);

// PCIe (PCCL) communicator destroy for this domain. Destroys stream then comm via pccl API.
// Safe to call when PCIe was never inited or when ENABLE_PCIE is off (no-op).
void DestroyPCIeForDomain(CommDomain* domain);

}  // namespace ampccl

#endif  // AMPCCL_CORE_COMM_INIT_H_
