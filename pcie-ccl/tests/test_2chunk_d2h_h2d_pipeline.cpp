#include "include/comm.hpp"
#include "include/hal/device.hpp"
#include "include/ir.hpp"
#include "test_utils.hpp"

using namespace pccl;

static size_t parseSize(const char* str) {
  char* end;
  size_t val = strtoull(str, &end, 10);
  if (*end == 'K' || *end == 'k') val *= 1024;
  else if (*end == 'M' || *end == 'm') val *= 1024 * 1024;
  else if (*end == 'G' || *end == 'g') val *= 1024 * 1024 * 1024;
  return val;
}

int main(int argc, char** argv) {
  size_t size = argc > 1 ? parseSize(argv[1]) : DEFAULT_SIZE;

  printf("\n=== Test: 2-Chunk D2H + H2D Pipeline ===\n");
  printf("Size per chunk: %.0f MB\n", size / (1024.0 * 1024.0));
  printf("Total data: %.0f MB (2 chunks)\n", 2 * size / (1024.0 * 1024.0));
  printf("Warmup: %d iterations, Measure: %d iterations\n\n", WARMUP_ITERS, MEASURE_ITERS);

  pcclComm_t comm;
  PCCL_CHECK(pcclInit(0, 1, &comm));

  void* sendbuff;
  void* recvbuff;
  // count represents total elements for all input chunks
  // With 2 input chunks of 'size' bytes each, total = 2 * size
  size_t count = (2 * size) / sizeof(float);

  // sendbuff: 2 chunks (source for D2H)
  // chunk 0 filled with 0xAA, chunk 1 filled with 0xBB
  if (devMalloc(&sendbuff, size * 2) != 0) {
    fprintf(stderr, "Failed to allocate sendbuff\n");
    return 1;
  }
  // recvbuff: 4 chunks (H2D writes to chunks 2 and 3)
  if (devMalloc(&recvbuff, size * 4) != 0) {
    fprintf(stderr, "Failed to allocate recvbuff\n");
    return 1;
  }

  // Allocate host buffers for initialization and verification
  void* h_init;
  void* h_verify;
  DEV_CHECK(devMallocHost(&h_init, size));
  DEV_CHECK(devMallocHost(&h_verify, size * 4));

  // Initialize sendbuff chunk 0 with 0xAA pattern
  memset(h_init, 0xAA, size);
  DEV_CHECK(devMemcpySync(sendbuff, h_init, size, MemcpyDirection::H2D));

  // Initialize sendbuff chunk 1 with 0xBB pattern
  memset(h_init, 0xBB, size);
  DEV_CHECK(devMemcpySync((char*)sendbuff + size, h_init, size, MemcpyDirection::H2D));

  // Zero out recvbuff (all 4 chunks)
  memset(h_init, 0x00, size);
  for (int i = 0; i < 4; i++) {
    DEV_CHECK(devMemcpySync((char*)recvbuff + i * size, h_init, size, MemcpyDirection::H2D));
  }

  devStream stream;
  if (devCreateStream(&stream) != 0) {
    fprintf(stderr, "Failed to create stream\n");
    return 1;
  }

  // Build IR program:
  // D2H (chunk 0 -> host chunk 0) + D2H (chunk 1 -> host chunk 1)
  // H2D (host chunk 0 -> chunk 2) + H2D (host chunk 1 -> chunk 3)
  IRProgram program;
  program.input_chunk_count = 2;
  program.output_chunk_count = 4;

  // Instruction 0: D2H - copy from sendbuff[0] to host chunk 0
  Instruction inst0;
  inst0.op = OpCode::D2H;
  inst0.src_chunk_idx = 0;
  inst0.dst_chunk_idx = 0;
  inst0.effects = {{0, 1}};  // chunk 0, version += 1

  // Instruction 1: D2H - copy from sendbuff[1] to host chunk 1
  Instruction inst1;
  inst1.op = OpCode::D2H;
  inst1.src_chunk_idx = 1;
  inst1.dst_chunk_idx = 1;
  inst1.effects = {{1, 1}};  // chunk 1, version += 1

  // Instruction 2: H2D - copy from host chunk 0 to recvbuff[2]
  Instruction inst2;
  inst2.op = OpCode::H2D;
  inst2.src_chunk_idx = 0;   // host chunk 0
  inst2.dst_chunk_idx = 2;   // recvbuff[2]
  inst2.deps = {{0, 0, 1}};  // wait for numa 0, chunk 0, version >= 1

  // Instruction 3: H2D - copy from host chunk 1 to recvbuff[3]
  Instruction inst3;
  inst3.op = OpCode::H2D;
  inst3.src_chunk_idx = 1;   // host chunk 1
  inst3.dst_chunk_idx = 3;   // recvbuff[3]
  inst3.deps = {{0, 1, 1}};  // wait for numa 0, chunk 1, version >= 1

  program.instructions = {inst0, inst1, inst2, inst3};

  HostTimer timer;
  std::vector<float> times;

  // Total data moved through pipeline: 2 chunks
  size_t total_data = size * 2;

  for (int i = 0; i < WARMUP_ITERS + MEASURE_ITERS; i++) {
    timer.start();
    PCCL_CHECK(pcclSubmit(comm, program, sendbuff, recvbuff, count, stream));
    if (devSynchronizeStream(stream) != 0) {
      fprintf(stderr, "Failed to synchronize stream\n");
      return 1;
    }
    timer.end();

    float ms = timer.getElapsedMs();
    double gbps = (total_data / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);

    if (i >= WARMUP_ITERS) {
      times.push_back(ms);
      printf("Iteration %d: %.2f ms, %.2f GB/s\n", i, ms, gbps);
    } else {
      printf("Iteration %d (warmup): %.2f ms, %.2f GB/s\n", i, ms, gbps);
    }
  }

  auto stats = calculateStats(times, total_data);
  printf("\nStatistics:\n");
  printf("  Time:      Avg: %.2f ms\n", stats.avg_time_ms);
  printf("  Bandwidth: Avg: %.2f GB/s\n", stats.avg_bw_gbps);

#if LOG_LEVEL <= 0
  pcclPrintProfilingStats(comm);
#endif

  // Synchronize internal streams before verification and freeing memory
  pcclSynchronizeInternalStreams(comm);

  // Verify data
  DEV_CHECK(devMemcpySync(h_verify, recvbuff, size * 4, MemcpyDirection::D2H));

  // recvbuff[0] should be untouched (0x00)
  // recvbuff[1] should be untouched (0x00)
  // recvbuff[2] should contain data from sendbuff[0] (0xAA)
  // recvbuff[3] should contain data from sendbuff[1] (0xBB)
  bool chunk0_ok = verifyData((char*)h_verify, size, (char)0x00, "recvbuff[0]");
  bool chunk1_ok = verifyData((char*)h_verify + size, size, (char)0x00, "recvbuff[1]");
  bool chunk2_ok = verifyData((char*)h_verify + size * 2, size, (char)0xAA, "recvbuff[2]");
  bool chunk3_ok = verifyData((char*)h_verify + size * 3, size, (char)0xBB, "recvbuff[3]");

  if (chunk0_ok && chunk1_ok && chunk2_ok && chunk3_ok) {
    printf("Verification: PASSED\n");
  } else {
    printf("Verification: FAILED\n");
  }

  devDestroyStream(stream);
  devFree(sendbuff);
  devFree(recvbuff);
  PCCL_CHECK(pcclDestroy(comm));
  DEV_CHECK(devFreeHost(h_init));
  DEV_CHECK(devFreeHost(h_verify));

  return 0;
}
