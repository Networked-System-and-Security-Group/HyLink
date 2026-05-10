#include "include/comm.hpp"
#include "include/hal/device.hpp"
#include "include/ir.hpp"
#include "test_utils.hpp"

using namespace pccl;

int main(int argc, char** argv) {
  size_t size = argc > 1 ? atoll(argv[1]) : DEFAULT_SIZE;

  printf("\n=== Test: IR-based D2H ===\n");
  printf("Size: %.0f MB\n", size / (1024.0 * 1024.0));
  printf("Warmup: %d iterations, Measure: %d iterations\n\n", WARMUP_ITERS, MEASURE_ITERS);

  pcclComm_t comm;
  PCCL_CHECK(pcclInit(0, 1, &comm));

  void* sendbuff;
  void* recvbuff;
  size_t count = size / sizeof(float);

  if (devMalloc(&sendbuff, size) != 0) {
    fprintf(stderr, "Failed to allocate sendbuff\n");
    return 1;
  }
  if (devMalloc(&recvbuff, size) != 0) {
    fprintf(stderr, "Failed to allocate recvbuff\n");
    return 1;
  }

  devStream stream;
  if (devCreateStream(&stream) != 0) {
    fprintf(stderr, "Failed to create stream\n");
    return 1;
  }

  IRProgram program;
  program.input_chunk_count = 1;
  program.output_chunk_count = 1;

  Instruction inst;
  inst.op = OpCode::D2H;
  inst.src_chunk_idx = 0;
  inst.dst_chunk_idx = 0;
  inst.effects = {{0, 1}};

  program.instructions = {inst};

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

  // Synchronize internal streams before freeing memory
  pcclSynchronizeInternalStreams(comm);

  devDestroyStream(stream);
  devFree(sendbuff);
  devFree(recvbuff);
  PCCL_CHECK(pcclDestroy(comm));

  return 0;
}
