#ifndef PCCL_ERROR_HPP
#define PCCL_ERROR_HPP

#include "pccl/types.h"

namespace pccl {

inline const char* pcclGetErrorString(pcclResult_t result) {
  switch (result) {
    case pcclSuccess:
      return "pcclSuccess";
    case pcclInvalidArgument:
      return "pcclInvalidArgument";
    case pcclSystemError:
      return "pcclSystemError";
    case pcclInternalError:
      return "pcclInternalError";
    case pcclDeviceError:
      return "pcclDeviceError";
    case pcclNetworkError:
      return "pcclNetworkError";
    case pcclTimeout:
      return "pcclTimeout";
    default:
      return "pcclUnknownError";
  }
}

}  // namespace pccl

#endif  // PCCL_ERROR_HPP
