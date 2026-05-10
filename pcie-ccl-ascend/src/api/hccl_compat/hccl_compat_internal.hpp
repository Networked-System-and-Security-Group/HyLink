#ifndef PCCL_HCCL_COMPAT_INTERNAL_HPP
#define PCCL_HCCL_COMPAT_INTERNAL_HPP

#include <cstddef>

#include "hccl/hccl.h"
#include "pccl/pccl.h"
#include "../../ir/compiler.hpp"

namespace pccl::hccl_compat {

struct HcclCommShim {
  pcclComm_t pccl = nullptr;
};

HcclResult toHcclResult(pcclResult_t result);
pcclResult_t toPcclResult(HcclResult result);

bool mapDataType(HcclDataType dtype, ScalarDataType* scalar_type, size_t* element_size);
bool mapReduceOp(HcclReduceOp op, ReduceOpKind* reduce_op);
bool isD2DReduceSupported(HcclDataType dtype, HcclReduceOp op);
CompileOptions makeOptions(HcclDataType dtype, HcclReduceOp op);
HcclCommShim* asShim(HcclComm comm);

}  // namespace pccl::hccl_compat

#endif /* PCCL_HCCL_COMPAT_INTERNAL_HPP */
