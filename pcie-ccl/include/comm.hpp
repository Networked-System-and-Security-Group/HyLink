#ifndef COMM_HPP
#define COMM_HPP

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Result codes for PCCL operations
typedef enum {
  pcclSuccess = 0,
  pcclInvalidArgument = 1,
  pcclSystemError = 2,
  pcclInternalError = 3,
  pcclDeviceError = 4
} pcclResult_t;

// Opaque communicator handle
struct pcclComm;
typedef struct pcclComm* pcclComm_t;

// Opaque stream handle (matches device stream type)
typedef void* pcclStream_t;

// Memory copy direction
typedef enum { pcclMemcpyH2D = 0, pcclMemcpyD2H = 1, pcclMemcpyD2D = 2 } pcclMemcpyDirection_t;

// Initialize PCCL communicator
// Parameters:
//   rank: Process rank (0 to nranks-1)
//   nranks: Total number of processes
//   comm: Output parameter for communicator handle
// Returns: pcclSuccess on success, error code otherwise
pcclResult_t pcclInit(int rank, int nranks, pcclComm_t* comm);

// Destroy PCCL communicator and free resources
// Parameters:
//   comm: Communicator handle to destroy
// Returns: pcclSuccess on success, error code otherwise
pcclResult_t pcclDestroy(pcclComm_t comm);

// Get communicator rank
pcclResult_t pcclCommRank(pcclComm_t comm, int* rank);

// Get communicator world size
pcclResult_t pcclCommSize(pcclComm_t comm, int* size);

// Synchronize internal D2H/H2D streams
// Must be called before freeing device memory to ensure internal streams complete
void pcclSynchronizeInternalStreams(pcclComm_t comm);

#ifdef __cplusplus
}
#endif

// C++ API for IR program submission
#ifdef __cplusplus
#include "log.hpp"

namespace pccl {
struct IRProgram;  // Forward declaration
}

// Submit IR program (C++ only)
pcclResult_t pcclSubmit(pcclComm_t comm, pccl::IRProgram program, void* dev_sendbuff, void* dev_recvbuff, size_t count,
                        pcclStream_t stream);

#if LOG_LEVEL <= 0
// Print D2H/H2D profiling statistics (call after stream sync, debug mode only)
void pcclPrintProfilingStats(pcclComm_t comm);

// Clear profiling records for next submission (debug mode only)
void pcclClearProfilingRecords(pcclComm_t comm);
#endif

#endif

#endif /* COMM_HPP */
