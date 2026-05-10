#ifndef DEVICE_HPP
#define DEVICE_HPP

// Device runtime: Compile-time selection of device implementation
// This header provides backward compatibility by including the new tag dispatch implementation

#include "device_rt.hpp"

// Backend detection for logging/debugging
#if defined(PCCL_DEVICE_CUDA)
#define PCCL_DEVICE_BACKEND "CUDA"
#elif defined(PCCL_DEVICE_ASCEND)
#define PCCL_DEVICE_BACKEND "CANN"
#endif

#endif /* DEVICE_HPP */
