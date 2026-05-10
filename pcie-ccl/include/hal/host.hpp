#ifndef HOST_HPP
#define HOST_HPP

// Host runtime: Compile-time selection of optimal CPU implementation
// This header provides backward compatibility by including the new tag dispatch implementation

#include "host_rt.hpp"

// Backend detection for logging/debugging
#if defined(__AVX512F__)
#define PCCL_HOST_BACKEND "AVX512"
#elif defined(__AVX2__)
#define PCCL_HOST_BACKEND "AVX256"
#elif defined(__ARM_NEON)
#define PCCL_HOST_BACKEND "NEON"
#else
#define PCCL_HOST_BACKEND "Standard"
#endif

#endif /* HOST_HPP */
