#ifndef AMPCCL_BACKEND_DTYPE_DTYPE_HCCL_H_
#define AMPCCL_BACKEND_DTYPE_DTYPE_HCCL_H_

#include <cstddef>

namespace ampccl {

// Subset matching current CANN/HCCL public enum values.
enum class HcclDataTypeAlias : int {
    Int8 = 0,
    Int16 = 1,
    Int32 = 2,
    Float16 = 3,
    Float = 4,
    Int64 = 5,
    Uint64 = 6,
    Uint8 = 7,
    Uint16 = 8,
    Uint32 = 9,
    Float64 = 10,
    Bfloat16 = 11,
    Int128 = 12,
    Reserved = 255
};

// Map raw int from HcclDataType / extended HCCL enums to element size (bytes).
inline size_t ElemSizeHcclDataType(HcclDataTypeAlias dt) {
    switch (dt) {
        case HcclDataTypeAlias::Int8:
        case HcclDataTypeAlias::Uint8:
            return 1;
        case HcclDataTypeAlias::Int16:
        case HcclDataTypeAlias::Float16:
        case HcclDataTypeAlias::Uint16:
        case HcclDataTypeAlias::Bfloat16:
            return 2;
        case HcclDataTypeAlias::Float:
        case HcclDataTypeAlias::Int32:
        case HcclDataTypeAlias::Uint32:
            return 4;
        case HcclDataTypeAlias::Int64:
        case HcclDataTypeAlias::Uint64:
        case HcclDataTypeAlias::Float64:
            return 8;
        case HcclDataTypeAlias::Int128:
            return 16;
        default:
            return 0;
    }
}

inline size_t ElemSizeHcclDataTypeFromInt(int datatype) {
    switch (datatype) {
        case 0:   // INT8 / FLOAT per deployment
        case 7:
        case 13:
            return 1;
        case 1:
        case 3:
        case 8:
        case 11:
            return 2;
        case 2:
        case 4:
        case 9:
        case 12:
            return 4;
        case 5:
        case 6:
        case 10:
            return 8;
        case 14:
            return 16;
        default:
            return 0;
    }
}

}  // namespace ampccl

#endif  // AMPCCL_BACKEND_DTYPE_DTYPE_HCCL_H_
