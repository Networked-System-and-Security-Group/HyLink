#ifndef AMPCCL_BACKEND_DTYPE_NATIVE_DTYPE_H_
#define AMPCCL_BACKEND_DTYPE_NATIVE_DTYPE_H_

#include <cstddef>

#if defined(AMPCCL_BUILD_NCCL)
#include "dtype_nccl.h"
#elif defined(AMPCCL_BUILD_HCCL)
#include "dtype_hccl.h"
#else
#error "AMPCCL_BUILD_NCCL or AMPCCL_BUILD_HCCL must be defined"
#endif

namespace ampccl {

inline size_t NativeDataTypeElemSize(int datatype) {
#if defined(AMPCCL_BUILD_NCCL)
    return ElemSizeNcclDataType(static_cast<NcclDataTypeAlias>(datatype));
#elif defined(AMPCCL_BUILD_HCCL)
    size_t s = ElemSizeHcclDataType(static_cast<HcclDataTypeAlias>(datatype));
    if (s != 0) {
        return s;
    }
    return ElemSizeHcclDataTypeFromInt(datatype);
#endif
}

}  // namespace ampccl

#endif  // AMPCCL_BACKEND_DTYPE_NATIVE_DTYPE_H_
