#include "backend/dtype/pccl_dtype.h"

#include <cassert>
#include <cstddef>

int main() {
    struct DtypeCase {
        int hccl_dtype;
        pcclDataType_t pccl_dtype;
        std::size_t element_size;
    };

    const DtypeCase cases[] = {
        {0, pcclInt8, 1},
        {7, pcclUint8, 1},
        {3, pcclFloat16, 2},
        {11, pcclBfloat16, 2},
        {2, pcclInt32, 4},
        {9, pcclUint32, 4},
        {4, pcclFloat32, 4},
        {5, pcclInt64, 8},
        {6, pcclUint64, 8},
        {10, pcclFloat64, 8},
    };

    for (const auto& test_case : cases) {
        pcclDataType_t mapped = pcclInvalidDataType;
        assert(ampccl::MapHcclDataTypeToPccl(test_case.hccl_dtype, &mapped));
        assert(mapped == test_case.pccl_dtype);
        assert(ampccl::PcclDataTypeElementSize(mapped) == test_case.element_size);
        assert(ampccl::HcclPcclDataTypeSizeMatches(test_case.hccl_dtype, mapped));
    }

    pcclDataType_t unsupported = pcclInvalidDataType;
    assert(!ampccl::MapHcclDataTypeToPccl(1, &unsupported));
    assert(!ampccl::MapHcclDataTypeToPccl(8, &unsupported));
    assert(!ampccl::MapHcclDataTypeToPccl(12, &unsupported));
    assert(!ampccl::MapHcclDataTypeToPccl(255, &unsupported));

    pcclReduceOp_t reduce_op = pcclInvalidReduceOp;
    assert(ampccl::MapHcclReduceOpToPccl(0, &reduce_op));
    assert(reduce_op == pcclReduceSum);
    assert(ampccl::MapHcclReduceOpToPccl(2, &reduce_op));
    assert(reduce_op == pcclReduceMax);
    assert(ampccl::MapHcclReduceOpToPccl(3, &reduce_op));
    assert(reduce_op == pcclReduceMin);
    assert(!ampccl::MapHcclReduceOpToPccl(1, &reduce_op));
    assert(!ampccl::MapHcclReduceOpToPccl(255, &reduce_op));
    return 0;
}
