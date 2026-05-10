#include "hccl_compat_internal.hpp"

#include "../../runtime/ascend_runtime.hpp"

namespace pccl::hccl_compat {

HcclResult toHcclResult(pcclResult_t result) {
  switch (result) {
    case pcclSuccess:
      return HCCL_SUCCESS;
    case pcclInvalidArgument:
      return HCCL_E_PARA;
    case pcclSystemError:
      return HCCL_E_SYSCALL;
    case pcclInternalError:
      return HCCL_E_INTERNAL;
    case pcclDeviceError:
      return HCCL_E_RUNTIME;
    case pcclNetworkError:
      return HCCL_E_UNAVAIL;
    case pcclTimeout:
      return HCCL_E_TIMEOUT;
  }
  return HCCL_E_INTERNAL;
}

pcclResult_t toPcclResult(HcclResult result) {
  switch (result) {
    case HCCL_SUCCESS:
      return pcclSuccess;
    case HCCL_E_PARA:
    case HCCL_E_PTR:
      return pcclInvalidArgument;
    case HCCL_E_TIMEOUT:
      return pcclTimeout;
    case HCCL_E_RUNTIME:
      return pcclDeviceError;
    case HCCL_E_SYSCALL:
      return pcclSystemError;
    case HCCL_E_UNAVAIL:
      return pcclNetworkError;
    case HCCL_E_MEMORY:
    case HCCL_E_INTERNAL:
    case HCCL_E_NOT_SUPPORT:
    case HCCL_E_NOT_FOUND:
    case HCCL_E_RESERVED:
      break;
  }
  return pcclInternalError;
}

bool mapDataType(HcclDataType dtype, ScalarDataType* scalar_type, size_t* element_size) {
  if (!scalar_type || !element_size)
    return false;
  switch (dtype) {
    case HCCL_DATA_TYPE_INT8:
      *scalar_type = ScalarDataType::Int8;
      *element_size = 1;
      return true;
    case HCCL_DATA_TYPE_UINT8:
      *scalar_type = ScalarDataType::Uint8;
      *element_size = 1;
      return true;
    case HCCL_DATA_TYPE_INT32:
      *scalar_type = ScalarDataType::Int32;
      *element_size = 4;
      return true;
    case HCCL_DATA_TYPE_UINT32:
      *scalar_type = ScalarDataType::Uint32;
      *element_size = 4;
      return true;
    case HCCL_DATA_TYPE_INT64:
      *scalar_type = ScalarDataType::Int64;
      *element_size = 8;
      return true;
    case HCCL_DATA_TYPE_UINT64:
      *scalar_type = ScalarDataType::Uint64;
      *element_size = 8;
      return true;
    case HCCL_DATA_TYPE_FP16:
      *scalar_type = ScalarDataType::Float16;
      *element_size = 2;
      return true;
    case HCCL_DATA_TYPE_FP32:
      *scalar_type = ScalarDataType::Float32;
      *element_size = 4;
      return true;
    case HCCL_DATA_TYPE_FP64:
      *scalar_type = ScalarDataType::Float64;
      *element_size = 8;
      return true;
    case HCCL_DATA_TYPE_BFP16:
      *scalar_type = ScalarDataType::Bfloat16;
      *element_size = 2;
      return true;
    case HCCL_DATA_TYPE_INT16:
    case HCCL_DATA_TYPE_UINT16:
    case HCCL_DATA_TYPE_INT128:
    case HCCL_DATA_TYPE_RESERVED:
      break;
  }
  return false;
}

bool mapReduceOp(HcclReduceOp op, ReduceOpKind* reduce_op) {
  if (!reduce_op)
    return false;
  switch (op) {
    case HCCL_REDUCE_SUM:
      *reduce_op = ReduceOpKind::Sum;
      return true;
    case HCCL_REDUCE_MAX:
      *reduce_op = ReduceOpKind::Max;
      return true;
    case HCCL_REDUCE_MIN:
      *reduce_op = ReduceOpKind::Min;
      return true;
    case HCCL_REDUCE_PROD:
    case HCCL_REDUCE_RESERVED:
      break;
  }
  return false;
}

bool isD2DReduceSupported(HcclDataType dtype, HcclReduceOp op) {
  ScalarDataType scalar_type;
  size_t element_size = 0;
  ReduceOpKind reduce_op;
  if (!mapDataType(dtype, &scalar_type, &element_size) || !mapReduceOp(op, &reduce_op))
    return false;
  return devD2DReduceSupported(static_cast<int>(scalar_type), static_cast<int>(reduce_op));
}

CompileOptions makeOptions(HcclDataType dtype, HcclReduceOp op) {
  CompileOptions options;
  if (!mapDataType(dtype, &options.scalar_type, &options.element_size))
    options.element_size = 0;
  if (!mapReduceOp(op, &options.reduce_op))
    options.reduce_op = ReduceOpKind::Sum;
  return options;
}

HcclCommShim* asShim(HcclComm comm) {
  return reinterpret_cast<HcclCommShim*>(comm);
}

}  // namespace pccl::hccl_compat
