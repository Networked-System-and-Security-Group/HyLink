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

  printf("\n=== Test: D2H + H2D Pipeline ===\n");
  printf("Size: %.0f MB\n", size / (1024.0 * 1024.0));
  printf("Warmup: %d iterations, Measure: %d iterations\n\n", WARMUP_ITERS, MEASURE_ITERS);

  pcclComm_t comm;
  PCCL_CHECK(pcclInit(0, 1, &comm));

  void* sendbuff;
  void* recvbuff;
  size_t count = size / sizeof(float);

  // sendbuff: 1 chunk (source for D2H)
  if (devMalloc(&sendbuff, size) != 0) {
    fprintf(stderr, "Failed to allocate sendbuff\n");
    return 1;
  }
  // recvbuff: 2 chunks (H2D writes to chunk index 1)
  if (devMalloc(&recvbuff, size * 2) != 0) {
    fprintf(stderr, "Failed to allocate recvbuff\n");
    return 1;
  }

  // Allocate host buffers for initialization and verification
  void* h_init;
  void* h_verify;
  DEV_CHECK(devMallocHost(&h_init, size));
  DEV_CHECK(devMallocHost(&h_verify, size * 2));

  // Initialize sendbuff with 0xAA pattern
  memset(h_init, 0xAA, size);
  DEV_CHECK(devMemcpySync(sendbuff, h_init, size, MemcpyDirection::H2D));

  // Zero out recvbuff (both chunks)
  memset(h_init, 0x00, size);
  DEV_CHECK(devMemcpySync(recvbuff, h_init, size, MemcpyDirection::H2D));
  DEV_CHECK(devMemcpySync((char*)recvbuff + size, h_init, size, MemcpyDirection::H2D));

  devStream stream;
  if (devCreateStream(&stream) != 0) {
    fprintf(stderr, "Failed to create stream\n");
    return 1;
  }

  // Build IR program: D2H (chunk 0 -> host chunk 0) + H2D (host chunk 0 -> chunk 1)
  IRProgram program;
  program.input_chunk_count = 1;
  program.output_chunk_count = 2;

  // Instruction 0: D2H - copy from sendbuff[0] to host chunk 0
  Instruction inst0;
  inst0.op = OpCode::D2H;
  inst0.src_chunk_idx = 0;
  inst0.dst_chunk_idx = 0;
  inst0.effects = {{0, 1}};  // chunk 0, version += 1

  // Instruction 1: H2D - copy from host chunk 0 to recvbuff[1]
  Instruction inst1;
  inst1.op = OpCode::H2D;
  inst1.src_chunk_idx = 0;   // host chunk 0
  inst1.dst_chunk_idx = 1;   // recvbuff[1]
  inst1.deps = {{0, 0, 1}};  // wait for numa 0, chunk 0, version >= 1

  program.instructions = {inst0, inst1};

  HostTimer timer;
  std::vector<float> times;

  for (int i = 0; i < WARMUP_ITERS + MEASURE_ITERS; i++) {
    timer.start();
    PCCL_CHECK(pcclSubmit(comm, program, sendbuff, recvbuff, count, stream));
    if (devSynchronizeStream(stream) != 0) {
      fprintf(stderr, "Failed to synchronize stream\n");
      return 1;
    }
    timer.end();

    float ms = timer.getElapsedMs();
    double gbps = (size / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);

    if (i >= WARMUP_ITERS) {
      times.push_back(ms);
      printf("Iteration %d: %.2f ms, %.2f GB/s\n", i, ms, gbps);
    } else {
      printf("Iteration %d (warmup): %.2f ms, %.2f GB/s\n", i, ms, gbps);
    }
  }

  auto stats = calculateStats(times, size);
  printf("\nStatistics:\n");
  printf("  Time:      Avg: %.2f ms\n", stats.avg_time_ms);
  printf("  Bandwidth: Avg: %.2f GB/s\n", stats.avg_bw_gbps);

#if LOG_LEVEL <= 0
  pcclPrintProfilingStats(comm);
#endif

  // Synchronize internal streams before verification and freeing memory
  pcclSynchronizeInternalStreams(comm);

  // Verify data
  DEV_CHECK(devMemcpySync(h_verify, recvbuff, size * 2, MemcpyDirection::D2H));

  // recvbuff[0] should be untouched (0x00)
  // recvbuff[1] should contain the copied data (0xAA)
  bool chunk0_ok = verifyData((char*)h_verify, size, (char)0x00, "recvbuff[0]");
  bool chunk1_ok = verifyData((char*)h_verify + size, size, (char)0xAA, "recvbuff[1]");

  if (chunk0_ok && chunk1_ok) {
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
