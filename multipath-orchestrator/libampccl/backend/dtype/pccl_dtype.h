#ifndef AMPCCL_BACKEND_DTYPE_PCCL_DTYPE_H_
#define AMPCCL_BACKEND_DTYPE_PCCL_DTYPE_H_

#include <cstddef>

#include <pccl/pccl.h>

#include "backend/dtype/dtype_hccl.h"
#include "backend/dtype/native_dtype.h"

namespace ampccl {

// HCCL and PCCL intentionally have separate enums. Keep this switch explicit
// so a CANN enum change cannot silently reinterpret a PCCL datatype.
inline bool MapHcclDataTypeToPccl(int datatype, pcclDataType_t* mapped) {
    if (mapped == nullptr) {
        return false;
    }
    switch (static_cast<HcclDataTypeAlias>(datatype)) {
        case HcclDataTypeAlias::Int8:
            *mapped = pcclInt8;
            return true;
        case HcclDataTypeAlias::Uint8:
            *mapped = pcclUint8;
            return true;
        case HcclDataTypeAlias::Int32:
            *mapped = pcclInt32;
            return true;
        case HcclDataTypeAlias::Uint32:
            *mapped = pcclUint32;
            return true;
        case HcclDataTypeAlias::Int64:
            *mapped = pcclInt64;
            return true;
        case HcclDataTypeAlias::Uint64:
            *mapped = pcclUint64;
            return true;
        case HcclDataTypeAlias::Float16:
            *mapped = pcclFloat16;
            return true;
        case HcclDataTypeAlias::Float:
            *mapped = pcclFloat32;
            return true;
        case HcclDataTypeAlias::Float64:
            *mapped = pcclFloat64;
            return true;
        case HcclDataTypeAlias::Bfloat16:
            *mapped = pcclBfloat16;
            return true;
        case HcclDataTypeAlias::Int16:
        case HcclDataTypeAlias::Uint16:
        case HcclDataTypeAlias::Int128:
        case HcclDataTypeAlias::Reserved:
            break;
    }
    *mapped = pcclInvalidDataType;
    return false;
}

inline std::size_t PcclDataTypeElementSize(pcclDataType_t datatype) {
    switch (datatype) {
        case pcclInt8:
        case pcclUint8:
            return 1;
        case pcclFloat16:
        case pcclBfloat16:
            return 2;
        case pcclInt32:
        case pcclUint32:
        case pcclFloat32:
            return 4;
        case pcclInt64:
        case pcclUint64:
        case pcclFloat64:
            return 8;
        case pcclInvalidDataType:
            break;
    }
    return 0;
}

inline bool HcclPcclDataTypeSizeMatches(int hccl_datatype, pcclDataType_t pccl_datatype) {
    return NativeDataTypeElemSize(hccl_datatype) == PcclDataTypeElementSize(pccl_datatype);
}

inline bool MapHcclReduceOpToPccl(int op, pcclReduceOp_t* mapped) {
    if (mapped == nullptr) {
        return false;
    }
    switch (op) {
        case 0:
            *mapped = pcclReduceSum;
            return true;
        case 2:
            *mapped = pcclReduceMax;
            return true;
        case 3:
            *mapped = pcclReduceMin;
            return true;
        case 1:
        default:
            break;
    }
    *mapped = pcclInvalidReduceOp;
    return false;
}

}  // namespace ampccl

#endif  // AMPCCL_BACKEND_DTYPE_PCCL_DTYPE_H_
