#ifndef PCCL_CHECKS_HPP
#define PCCL_CHECKS_HPP

#include "comm.hpp"
#include "log.hpp"

// Check PCCL call, return error on failure
#define PCCLCHECK(call)                                  \
  do {                                                   \
    pcclResult_t RES = call;                             \
    if (RES != pcclSuccess) {                            \
      LOG_ERROR("%s:%d -> %d", __FILE__, __LINE__, RES); \
      return RES;                                        \
    }                                                    \
  } while (0)

// Check PCCL call, goto label on failure
#define PCCLCHECKGOTO(call, RES, label)                  \
  do {                                                   \
    RES = call;                                          \
    if (RES != pcclSuccess) {                            \
      LOG_ERROR("%s:%d -> %d", __FILE__, __LINE__, RES); \
      goto label;                                        \
    }                                                    \
  } while (0)

// Check device call
#define DEVCHECK(cmd)                                           \
  do {                                                          \
    auto err = cmd;                                             \
    if (err != devSuccess) {                                    \
      LOG_ERROR("Device failure at %s:%d", __FILE__, __LINE__); \
      return pcclDeviceError;                                   \
    }                                                           \
  } while (0)

// Check system call
#define SYSCHECK(statement, name)                                \
  do {                                                           \
    if ((statement) < 0) {                                       \
      LOG_ERROR("Call to " name " failed: %s", strerror(errno)); \
      return pcclSystemError;                                    \
    }                                                            \
  } while (0)

#endif  // PCCL_CHECKS_HPP
