#include "backend/pcie_kernels/allreduce_pcie.h"
#include "core/domain.h"

#ifdef AMPCCL_ENABLE_PCIE
#include "comm.hpp"
#include "ir.hpp"
#endif

namespace ampccl {
namespace pcie_kernels {

#ifdef AMPCCL_ENABLE_PCIE
namespace {

using namespace pccl;

IRProgram BuildAllReduceIR2(int rank) {
    IRProgram program;
    program.input_chunk_count = 1;
    program.output_chunk_count = 1;

    if (rank == 0) {
        Instruction inst0;
        inst0.op = OpCode::D2H;
        inst0.src_numa = 0;
        inst0.src_chunk_idx = 0;
        inst0.dst_chunk_idx = 0;
        inst0.deps = {};
        inst0.effects = {{0, 1}};

        Instruction inst1;
        inst1.op = OpCode::H2D;
        inst1.src_numa = 0;
        inst1.src_chunk_idx = 0;
        inst1.dst_chunk_idx = 0;
        inst1.deps = {{0, 0, 2}};
        inst1.effects = {};

        program.instructions = {inst0, inst1};
    } else {
        Instruction inst0;
        inst0.op = OpCode::D2H;
        inst0.src_numa = 0;
        inst0.src_chunk_idx = 0;
        inst0.dst_chunk_idx = 1;
        inst0.deps = {};
        inst0.effects = {{1, 1}};

        Instruction inst1;
        inst1.op = OpCode::H2H_REDUCE;
        inst1.src_numa = 0;
        inst1.src_chunk_idx = 1;
        inst1.dst_chunk_idx = 0;
        inst1.deps = {{0, 0, 1}};
        inst1.effects = {{0, 1}};

        Instruction inst2;
        inst2.op = OpCode::H2D;
        inst2.src_numa = 0;
        inst2.src_chunk_idx = 0;
        inst2.dst_chunk_idx = 0;
        inst2.deps = {{0, 0, 2}};
        inst2.effects = {};

        program.instructions = {inst0, inst1, inst2};
    }
    return program;
}

static const int kNumChunks4 = 4;

IRProgram BuildAllReduceIR4(int rank) {
    IRProgram program;
    program.input_chunk_count = kNumChunks4;
    program.output_chunk_count = kNumChunks4;

    for (int i = 0; i < kNumChunks4; i++) {
        Instruction d2h;
        d2h.op = OpCode::D2H;
        d2h.src_chunk_idx = i;
        d2h.dst_chunk_idx = rank * kNumChunks4 + i;
        d2h.effects = {{rank * kNumChunks4 + i, 1}};
        program.instructions.push_back(d2h);
    }

    for (int s = 0; s < 4; s++) {
        Instruction reduce;
        reduce.op = OpCode::H2H_REDUCE;
        reduce.src_numa = 0;
        reduce.src_chunk_idx = s * kNumChunks4 + rank;
        reduce.dst_chunk_idx = 16 + rank;
        reduce.deps = {{0, s * kNumChunks4 + rank, 1}};
        reduce.effects = {{16 + rank, 1}};
        program.instructions.push_back(reduce);
    }

    for (int i = 0; i < kNumChunks4; i++) {
        Instruction h2d;
        h2d.op = OpCode::H2D;
        h2d.src_chunk_idx = 16 + i;
        h2d.dst_chunk_idx = i;
        h2d.deps = {{0, 16 + i, 4}};
        program.instructions.push_back(h2d);
    }
    return program;
}

IRProgram BuildAllReduceIR8(int rank) {
    IRProgram program;
    program.input_chunk_count = 1;
    program.output_chunk_count = 1;

    Instruction d2h;
    d2h.op = OpCode::D2H;
    d2h.src_numa = 0;
    d2h.src_chunk_idx = 0;
    d2h.dst_chunk_idx = rank;
    d2h.deps = {};
    d2h.effects = {{rank, 1}};
    program.instructions.push_back(d2h);

    if (rank == 0) {
        for (int s = 1; s < 8; ++s) {
            Instruction red;
            red.op = OpCode::H2H_REDUCE;
            red.src_numa = s;
            red.src_chunk_idx = s;
            red.dst_chunk_idx = 0;
            red.effects = {{0, 1}};
            if (s == 1) {
                red.deps = {{1, 1, 1}};
            } else {
                red.deps = {{s, s, 1}, {0, 0, static_cast<uint64_t>(s)}};
            }
            program.instructions.push_back(red);
        }
        Instruction h2d;
        h2d.op = OpCode::H2D;
        h2d.src_numa = 0;
        h2d.src_chunk_idx = 0;
        h2d.dst_chunk_idx = 0;
        h2d.deps = {{0, 0, 8}};
        h2d.effects = {};
        program.instructions.push_back(h2d);
    } else {
        Instruction h2h;
        h2h.op = OpCode::H2H;
        h2h.src_numa = 0;
        h2h.src_chunk_idx = 0;
        h2h.dst_chunk_idx = 0;
        h2h.deps = {{0, 0, 8}};
        h2h.effects = {{0, 1}};
        program.instructions.push_back(h2h);

        Instruction h2d;
        h2d.op = OpCode::H2D;
        h2d.src_numa = 0;
        h2d.src_chunk_idx = 0;
        h2d.dst_chunk_idx = 0;
        h2d.deps = {{0, 0, 9}};
        h2d.effects = {};
        program.instructions.push_back(h2d);
    }
    return program;
}

}  // namespace

BackendResult RunAllReducePcie(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    int datatype,
    int op,
    void* stream) {
    (void)datatype;
    (void)op;
    int nranks = domain ? domain->pcie_nranks() : 0;
    if (!domain || !domain->pcie_comm() || nranks < 2 || nranks > 8) {
        return BackendResult::Success;
    }
    pcclComm_t comm = static_cast<pcclComm_t>(domain->pcie_comm());
    void* pcie_stream = domain->pcie_stream();
    if (!pcie_stream) {
        return BackendResult::UnhandledError;
    }
    int rank = domain->pcie_rank();
    pccl::IRProgram program;
    switch (nranks) {
        case 2:
            program = BuildAllReduceIR2(rank);
            break;
        case 4:
            program = BuildAllReduceIR4(rank);
            break;
        case 8:
            program = BuildAllReduceIR8(rank);
            break;
        default:
            return BackendResult::Success;
    }
    pcclResult_t ret = pcclSubmit(comm, program,
                                  const_cast<void*>(sendbuff), recvbuff,
                                  count, static_cast<pcclStream_t>(pcie_stream));
    return (ret == pcclSuccess) ? BackendResult::Success : BackendResult::UnhandledError;
}

#else

BackendResult RunAllReducePcie(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    int datatype,
    int op,
    void* stream) {
    (void)domain;
    (void)sendbuff;
    (void)recvbuff;
    (void)count;
    (void)datatype;
    (void)op;
    (void)stream;
    return BackendResult::Success;
}

#endif  // AMPCCL_ENABLE_PCIE

}  // namespace pcie_kernels
}  // namespace ampccl

