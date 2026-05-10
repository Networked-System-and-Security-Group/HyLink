#ifndef PCCL_ERROR_HPP
#define PCCL_ERROR_HPP

#include "comm.hpp"  // Use C API's pcclResult_t

namespace pccl {

// Helper function to convert error code to string
inline const char* pcclGetErrorString(pcclResult_t result) {
  switch (result) {
    case pcclSuccess:
      return "pcclSuccess";
    case pcclInvalidArgument:
      return "pcclInvalidArgument";
    case pcclSystemError:
      return "pcclSystemError";
    case pcclDeviceError:
      return "pcclDeviceError";
    case pcclInternalError:
      return "pcclInternalError";
    default:
      return "pcclUnknownError";
  }
}

}  // namespace pccl

#endif  // PCCL_ERROR_HPP
