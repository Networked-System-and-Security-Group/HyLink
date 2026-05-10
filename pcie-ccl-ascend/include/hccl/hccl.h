#ifndef HCCL_H_
#define HCCL_H_

#include <acl/acl.h>
#include <hccl/hccl_types.h>

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcclGetRootInfo(HcclRootInfo* rootInfo);
HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm);
HcclResult HcclCommInitRootInfoConfig(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank,
                                      const HcclCommConfig* config, HcclComm* comm);
HcclResult HcclCommDestroy(HcclComm comm);
uint32_t HcclGetCommConfigCapability(void);

HcclResult HcclAllReduce(void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType,
                         HcclReduceOp op, HcclComm comm, aclrtStream stream);
HcclResult HcclAllGather(void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType,
                         HcclComm comm, aclrtStream stream);
HcclResult HcclReduceScatter(void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType,
                             HcclReduceOp op, HcclComm comm, aclrtStream stream);
HcclResult HcclAlltoAll(const void* sendBuf, uint64_t sendCount, HcclDataType sendType,
                        const void* recvBuf, uint64_t recvCount, HcclDataType recvType,
                        HcclComm comm, aclrtStream stream);
HcclResult HcclBroadcast(void* buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm,
                         aclrtStream stream);

HcclResult HcclCommSetMemoryRange(HcclComm comm, void* baseVirPtr, size_t size, size_t alignment, uint64_t flags);
HcclResult HcclCommUnsetMemoryRange(HcclComm comm, void* baseVirPtr);
HcclResult HcclCommActivateCommMemory(HcclComm comm, void* virPtr, size_t size, size_t offset,
                                      aclrtDrvMemHandle handle, uint64_t flags);
HcclResult HcclCommDeactivateCommMemory(HcclComm comm, void* virPtr);
HcclResult HcclCommSymWinRegister(HcclComm comm, void* addr, uint64_t size, CommSymWindow* winHandle, uint32_t flag);
HcclResult HcclCommSymWinDeregister(CommSymWindow winHandle);

#ifdef __cplusplus
}
#endif

#endif /* HCCL_H_ */
