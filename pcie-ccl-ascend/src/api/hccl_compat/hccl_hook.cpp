#include "hccl_compat_internal.hpp"

#include <cstring>
#include <new>

using namespace pccl::hccl_compat;

namespace {

constexpr unsigned char kSymWinDummy = 1;

bool rootInfoToUniqueId(const HcclRootInfo* rootInfo, pcclUniqueId* uniqueId) {
  if (!rootInfo || !uniqueId)
    return false;
  std::memset(uniqueId, 0, sizeof(*uniqueId));
  std::memcpy(uniqueId, rootInfo->internal,
              sizeof(*uniqueId) < sizeof(rootInfo->internal) ? sizeof(*uniqueId) : sizeof(rootInfo->internal));
  return true;
}

HcclResult submitTyped(HcclComm comm, const char* collective, const void* sendBuf, void* recvBuf,
                       uint64_t segCount, HcclDataType dtype, HcclReduceOp op, int sendSegs,
                       int recvSegs, aclrtStream stream) {
  HcclCommShim* shim = asShim(comm);
  if (!shim || !shim->pccl || !collective || !sendBuf || !recvBuf || segCount == 0)
    return HCCL_E_PARA;
  if (op == HCCL_REDUCE_PROD)
    return HCCL_E_NOT_SUPPORT;

  pccl::CompileOptions options = makeOptions(dtype, op);
  if (options.element_size == 0)
    return HCCL_E_NOT_SUPPORT;
  if (!mapReduceOp(op, &options.reduce_op))
    return HCCL_E_NOT_SUPPORT;

  return toHcclResult(pccl::pcclSubmitTypedCollective(shim->pccl, collective, sendBuf, recvBuf,
                                                      static_cast<size_t>(segCount), 0, sendSegs, recvSegs,
                                                      stream, &options));
}

}  // namespace

extern "C" {

HcclResult HcclGetRootInfo(HcclRootInfo* rootInfo) {
  if (!rootInfo)
    return HCCL_E_PTR;
  pcclUniqueId id{};
  pcclResult_t result = pcclGetUniqueId(&id);
  if (result != pcclSuccess)
    return toHcclResult(result);
  std::memset(rootInfo, 0, sizeof(*rootInfo));
  std::memcpy(rootInfo->internal, id.internal, sizeof(id.internal));
  return HCCL_SUCCESS;
}

HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank,
                                HcclComm* comm) {
  return HcclCommInitRootInfoConfig(nRanks, rootInfo, rank, nullptr, comm);
}

HcclResult HcclCommInitRootInfoConfig(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank,
                                      const HcclCommConfig* config, HcclComm* comm) {
  (void)config;
  if (!comm || !rootInfo || nRanks == 0 || rank >= nRanks)
    return HCCL_E_PARA;

  pcclUniqueId id{};
  if (!rootInfoToUniqueId(rootInfo, &id))
    return HCCL_E_PARA;

  auto* shim = new (std::nothrow) HcclCommShim();
  if (!shim)
    return HCCL_E_MEMORY;

  pcclResult_t result = pcclCommInitRank(&shim->pccl, static_cast<int>(nRanks), id, static_cast<int>(rank));
  if (result != pcclSuccess) {
    delete shim;
    return toHcclResult(result);
  }
  *comm = reinterpret_cast<HcclComm>(shim);
  return HCCL_SUCCESS;
}

HcclResult HcclCommDestroy(HcclComm comm) {
  HcclCommShim* shim = asShim(comm);
  if (!shim)
    return HCCL_E_PARA;
  pcclResult_t result = shim->pccl ? pcclCommDestroy(shim->pccl) : pcclSuccess;
  delete shim;
  return toHcclResult(result);
}

static HcclResult getRankSize(HcclComm comm, uint32_t* rankSize) {
  HcclCommShim* shim = asShim(comm);
  if (!shim || !shim->pccl || !rankSize)
    return HCCL_E_PARA;
  int size = 0;
  pcclResult_t result = pcclCommSize(shim->pccl, &size);
  if (result != pcclSuccess)
    return toHcclResult(result);
  *rankSize = static_cast<uint32_t>(size);
  return HCCL_SUCCESS;
}

uint32_t HcclGetCommConfigCapability(void) {
  return 0;
}

HcclResult HcclAllReduce(void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType,
                         HcclReduceOp op, HcclComm comm, aclrtStream stream) {
  if (!isD2DReduceSupported(dataType, op))
    return HCCL_E_NOT_SUPPORT;
  return submitTyped(comm, "allreduce", sendBuf, recvBuf, count, dataType, op, 1, 1, stream);
}

HcclResult HcclAllGather(void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType,
                         HcclComm comm, aclrtStream stream) {
  HcclCommShim* shim = asShim(comm);
  if (!shim || !shim->pccl)
    return HCCL_E_PARA;
  uint32_t nranks = 0;
  HcclResult rank_result = getRankSize(comm, &nranks);
  if (rank_result != HCCL_SUCCESS)
    return rank_result;
  return submitTyped(comm, "allgather", sendBuf, recvBuf, sendCount, dataType, HCCL_REDUCE_SUM, 1,
                     static_cast<int>(nranks), stream);
}

HcclResult HcclReduceScatter(void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType,
                             HcclReduceOp op, HcclComm comm, aclrtStream stream) {
  if (!isD2DReduceSupported(dataType, op))
    return HCCL_E_NOT_SUPPORT;
  HcclCommShim* shim = asShim(comm);
  if (!shim || !shim->pccl)
    return HCCL_E_PARA;
  uint32_t nranks = 0;
  HcclResult rank_result = getRankSize(comm, &nranks);
  if (rank_result != HCCL_SUCCESS)
    return rank_result;
  return submitTyped(comm, "reducescatter", sendBuf, recvBuf, recvCount, dataType, op,
                     static_cast<int>(nranks), 1, stream);
}

HcclResult HcclAlltoAll(const void* sendBuf, uint64_t sendCount, HcclDataType sendType,
                        const void* recvBuf, uint64_t recvCount, HcclDataType recvType,
                        HcclComm comm, aclrtStream stream) {
  if (sendType != recvType || sendCount != recvCount)
    return HCCL_E_NOT_SUPPORT;
  HcclCommShim* shim = asShim(comm);
  if (!shim || !shim->pccl)
    return HCCL_E_PARA;
  uint32_t nranks = 0;
  HcclResult rank_result = getRankSize(comm, &nranks);
  if (rank_result != HCCL_SUCCESS)
    return rank_result;
  return submitTyped(comm, "alltoall", sendBuf, const_cast<void*>(recvBuf), sendCount, sendType,
                     HCCL_REDUCE_SUM, static_cast<int>(nranks), static_cast<int>(nranks), stream);
}

HcclResult HcclBroadcast(void* buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm,
                         aclrtStream stream) {
  if (root != 0)
    return HCCL_E_NOT_SUPPORT;
  HcclCommShim* shim = asShim(comm);
  if (!shim || !shim->pccl || !buf)
    return HCCL_E_PARA;
  pccl::CompileOptions options = makeOptions(dataType, HCCL_REDUCE_SUM);
  if (options.element_size == 0)
    return HCCL_E_NOT_SUPPORT;
  return toHcclResult(pccl::pcclSubmitTypedCollective(shim->pccl, "broadcast_root0", buf, buf,
                                                      static_cast<size_t>(count), 0, 1, 1, stream,
                                                      &options));
}

HcclResult HcclCommSetMemoryRange(HcclComm comm, void* baseVirPtr, size_t size, size_t alignment,
                                  uint64_t flags) {
  (void)comm;
  (void)baseVirPtr;
  (void)size;
  (void)alignment;
  (void)flags;
  return HCCL_SUCCESS;
}

HcclResult HcclCommUnsetMemoryRange(HcclComm comm, void* baseVirPtr) {
  (void)comm;
  (void)baseVirPtr;
  return HCCL_SUCCESS;
}

HcclResult HcclCommActivateCommMemory(HcclComm comm, void* virPtr, size_t size, size_t offset,
                                      aclrtDrvMemHandle handle, uint64_t flags) {
  (void)comm;
  (void)virPtr;
  (void)size;
  (void)offset;
  (void)handle;
  (void)flags;
  return HCCL_SUCCESS;
}

HcclResult HcclCommDeactivateCommMemory(HcclComm comm, void* virPtr) {
  (void)comm;
  (void)virPtr;
  return HCCL_SUCCESS;
}

HcclResult HcclCommSymWinRegister(HcclComm comm, void* addr, uint64_t size, CommSymWindow* winHandle,
                                  uint32_t flag) {
  (void)comm;
  (void)addr;
  (void)size;
  (void)flag;
  if (!winHandle)
    return HCCL_E_PTR;
  *winHandle = const_cast<unsigned char*>(&kSymWinDummy);
  return HCCL_SUCCESS;
}

HcclResult HcclCommSymWinDeregister(CommSymWindow winHandle) {
  (void)winHandle;
  return HCCL_SUCCESS;
}

}  // extern "C"
