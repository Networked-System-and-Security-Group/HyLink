#include "backend/pcie_kernels/allgather_pcie.h"
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

IRProgram BuildAllGatherIR2(int rank) {
    IRProgram program;
  program.input_chunk_count = 1;
  program.output_chunk_count = 2;

  if (rank == 0) {
    Instruction inst0;
    inst0.op = OpCode::D2H;
    inst0.src_chunk_idx = 0;
    inst0.dst_chunk_idx = 0;
    inst0.effects = {{0}};

    Instruction inst1;
    inst1.op = OpCode::D2D;
    inst1.src_chunk_idx = 0;
    inst1.dst_chunk_idx = 0;

    Instruction inst2;
    inst2.op = OpCode::H2D;
    inst2.src_chunk_idx = 1;
    inst2.dst_chunk_idx = 1;
    inst2.deps = {{0, 1, 1}};

    program.instructions = {inst0, inst1, inst2};
  } else {
    Instruction inst0;
    inst0.op = OpCode::D2H;
    inst0.src_chunk_idx = 0;
    inst0.dst_chunk_idx = 1;
    inst0.effects = {{1}};

    Instruction inst1;
    inst1.op = OpCode::D2D;
    inst1.src_chunk_idx = 0;
    inst1.dst_chunk_idx = 1;

    Instruction inst2;
    inst2.op = OpCode::H2D;
    inst2.src_chunk_idx = 0;
    inst2.dst_chunk_idx = 0;
    inst2.deps = {{0, 0, 1}};

    program.instructions = {inst0, inst1, inst2};
  }
  return program;
}

IRProgram BuildAllGatherIR4(int rank) {
    IRProgram program;
    program.input_chunk_count = 1;
    program.output_chunk_count = 4;

    // D2H: upload own data to host chunk[rank]
    Instruction d2h;
    d2h.op = OpCode::D2H;
    d2h.src_chunk_idx = 0;
    d2h.dst_chunk_idx = rank;
    d2h.effects = {{rank}};
    program.instructions.push_back(d2h);

    // D2D: local copy of own data to recvbuff[rank]
    Instruction d2d;
    d2d.op = OpCode::D2D;
    d2d.src_chunk_idx = 0;
    d2d.dst_chunk_idx = rank;
    program.instructions.push_back(d2d);

    // H2D: pull other ranks' data from host
    for (int j = 0; j < 4; j++) {
        if (j == rank) continue;
        Instruction h2d;
        h2d.op = OpCode::H2D;
        h2d.src_chunk_idx = j;
        h2d.dst_chunk_idx = j;
        h2d.deps = {{0, j, 1}};
        program.instructions.push_back(h2d);
    }
    return program;
}

IRProgram BuildAllGatherIR8(int rank) {
    IRProgram program;
    program.input_chunk_count = 1;
    program.output_chunk_count = 8;

    Instruction d2h;
    d2h.op = OpCode::D2H;
    d2h.src_chunk_idx = 0;
    d2h.dst_chunk_idx = rank;
    d2h.deps = {};
    d2h.effects = {{rank}};
    program.instructions.push_back(d2h);

    Instruction d2d;
    d2d.op = OpCode::D2D;
    d2d.src_chunk_idx = 0;
    d2d.dst_chunk_idx = rank;
    d2d.deps = {};
    d2d.effects = {};
    program.instructions.push_back(d2d);

    for (int j = 0; j < 8; ++j) {
        if (j == rank) continue;
        Instruction h2d;
        h2d.op = OpCode::H2D;
        h2d.src_chunk_idx = j;
        h2d.dst_chunk_idx = j;
        h2d.deps = {{0, j, 1}};
        program.instructions.push_back(h2d);
    }
    return program;
}

}  // namespace

BackendResult RunAllGatherPcie(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    int datatype,
    void* stream) {
    (void)datatype;
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
            program = BuildAllGatherIR2(rank);
            break;
        case 4:
            program = BuildAllGatherIR4(rank);
            break;
        case 8:
            program = BuildAllGatherIR8(rank);
            break;
        default:
            return BackendResult::Success;
    }
    pcclResult_t ret = pcclSubmit(comm, program,
                                  const_cast<void*>(sendbuff), recvbuff,
                                  sendcount, static_cast<pcclStream_t>(pcie_stream));
    return (ret == pcclSuccess) ? BackendResult::Success : BackendResult::UnhandledError;
}

#else

BackendResult RunAllGatherPcie(
    CommDomain* domain,
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    int datatype,
    void* stream) {
    (void)domain;
    (void)sendbuff;
    (void)recvbuff;
    (void)sendcount;
    (void)datatype;
    (void)stream;
    return BackendResult::Success;
}

#endif  // AMPCCL_ENABLE_PCIE

}  // namespace pcie_kernels
}  // namespace ampccl
