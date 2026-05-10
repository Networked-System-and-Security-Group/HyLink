#ifndef AMPCCL_BACKEND_DTYPE_DTYPE_NCCL_H_
#define AMPCCL_BACKEND_DTYPE_DTYPE_NCCL_H_

#include <cstddef>

namespace ampccl {

// Mirrors common ncclDataType_t ordering used by NCCL hook stubs when nccl.h is absent.
enum class NcclDataTypeAlias : int {
    Int8 = 0,
    Uint8,
    Int32,
    Uint32,
    Int64,
    Uint64,
    Float16,
    Float32,
    Float64
};

inline size_t ElemSizeNcclDataType(NcclDataTypeAlias dt) {
    switch (dt) {
        case NcclDataTypeAlias::Int8:
        case NcclDataTypeAlias::Uint8:
            return 1;
        case NcclDataTypeAlias::Float16:
            return 2;
        case NcclDataTypeAlias::Int32:
        case NcclDataTypeAlias::Uint32:
        case NcclDataTypeAlias::Float32:
            return 4;
        case NcclDataTypeAlias::Int64:
        case NcclDataTypeAlias::Uint64:
        case NcclDataTypeAlias::Float64:
            return 8;
        default:
            return 0;
    }
}

}  // namespace ampccl

#endif  // AMPCCL_BACKEND_DTYPE_DTYPE_NCCL_H_
