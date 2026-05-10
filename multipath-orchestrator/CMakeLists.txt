cmake_minimum_required(VERSION 3.12)
project(Adaptive-CCL VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_SHARED_LIBS "Build shared library" ON)
# Mutually exclusive: build one architecture at a time (libampccl_nccl.so or libampccl_hccl.so)
option(NCCL_ONLY "Build NCCL hook only (output: libampccl_nccl.so)" OFF)
option(HCCL_ONLY "Build HCCL hook only (output: libampccl_hccl.so)" OFF)

# Must select exactly one architecture
if(NOT NCCL_ONLY AND NOT HCCL_ONLY)
    message(FATAL_ERROR "Set exactly one of -DNCCL_ONLY=ON or -DHCCL_ONLY=ON")
endif()
if(NCCL_ONLY AND HCCL_ONLY)
    message(FATAL_ERROR "Set only one of NCCL_ONLY or HCCL_ONLY, not both")
endif()

# ---------------------------------------------------------------------------
# NCCL / CUDA paths (when NCCL_ONLY)
# ---------------------------------------------------------------------------
if(DEFINED ENV{CUDA_HOME})
    set(CUDA_HOME "$ENV{CUDA_HOME}" CACHE PATH "CUDA installation directory")
endif()
if(NOT CUDA_HOME)
    set(CUDA_HOME "/usr/local/cuda" CACHE PATH "CUDA installation directory")
endif()
if(DEFINED ENV{NCCL_HOME})
    set(NCCL_HOME "$ENV{NCCL_HOME}" CACHE PATH "NCCL installation directory (optional)")
endif()

# ---------------------------------------------------------------------------
# HCCL / Ascend paths (when HCCL_ONLY)
# ---------------------------------------------------------------------------
if(DEFINED ENV{ASCEND_HOME})
    set(ASCEND_HOME "$ENV{ASCEND_HOME}" CACHE PATH "Ascend CANN installation directory")
endif()

# Include directories (project headers)
set(AMPCCL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libampccl")
include_directories(${AMPCCL_INCLUDE_DIR})

# Core sources (no hooks)
set(AMPCCL_CORE_SOURCES
    libampccl/backend/fast_backend.cc
    libampccl/backend/original_collective_api.cc
    libampccl/backend/pcie_backend.cc
    libampccl/backend/memory/collective_layout.cc
    libampccl/backend/memory/device_mem.cc
    libampccl/backend/pcie_kernels/allreduce_pcie.cc
    libampccl/backend/pcie_kernels/allgather_pcie.cc
    libampccl/core/comm_init.cc
    libampccl/core/stream_sync.cc
    libampccl/core/shm_store.cc
)

# Hook sources (mutually exclusive, one of NCCL_ONLY or HCCL_ONLY is ON)
if(NCCL_ONLY)
    set(AMPCCL_HOOK_SOURCES libampccl/hook/nccl_hook.cc)
    set(AMPCCL_TARGET_NAME ampccl_nccl)
else()
    set(AMPCCL_HOOK_SOURCES libampccl/hook/hccl_hook.cc)
    set(AMPCCL_TARGET_NAME ampccl_hccl)
endif()

set(AMPCCL_SOURCES ${AMPCCL_CORE_SOURCES} ${AMPCCL_HOOK_SOURCES})

# Header files (for IDE support)
set(AMPCCL_HEADERS
    libampccl/common/op_key.h
    libampccl/common/config.h
    libampccl/common/log.h
    libampccl/telemetry/timer.h
    libampccl/telemetry/stats.h
    libampccl/cache/param_cache.h
    libampccl/backend/backend_base.h
    libampccl/backend/fast_backend.h
    libampccl/backend/pcie_backend.h
    libampccl/backend/memory/collective_layout.h
    libampccl/backend/memory/device_mem.h
    libampccl/backend/pcie_kernels/allreduce_pcie.h
    libampccl/backend/pcie_kernels/allgather_pcie.h
    libampccl/controller/algo_base.h
    libampccl/controller/algo_tcp.h
    libampccl/controller/algo_dcqcn.h
    libampccl/controller/algo_factory.h
    libampccl/controller/controller.h
    libampccl/core/domain_key.h
    libampccl/core/domain.h
    libampccl/core/domain_manager.h
    libampccl/core/shm_store.h
    libampccl/core/comm_init.h
    libampccl/core/planner.h
    libampccl/core/stream_sync.h
    libampccl/core/virtual_collective.h
)

# Create library (name: ampccl, ampccl_nccl, or ampccl_hccl)
add_library(${AMPCCL_TARGET_NAME} SHARED ${AMPCCL_SOURCES} ${AMPCCL_HEADERS})

# No link against libnccl/libhccl at build time; hooks use dlopen/dlsym at runtime.
target_link_libraries(${AMPCCL_TARGET_NAME} PRIVATE
    ${CMAKE_DL_LIBS}  # For dlopen/dlsym
)

# Timer backend + single-arch include/link paths
if(NCCL_ONLY)
    set(NCCL_PATH_ADDED OFF)
    if(EXISTS "${CUDA_HOME}/include")
        target_include_directories(${AMPCCL_TARGET_NAME} PRIVATE ${CUDA_HOME}/include)
        target_link_directories(${AMPCCL_TARGET_NAME} PRIVATE ${CUDA_HOME}/lib64)
        target_link_libraries(${AMPCCL_TARGET_NAME} PRIVATE cudart)
        target_compile_definitions(${AMPCCL_TARGET_NAME} PRIVATE AMPCCL_USE_CUDA_TIMER)
        set(NCCL_PATH_ADDED ON)
        message(STATUS "  NCCL/CUDA: include=${CUDA_HOME}/include, lib=${CUDA_HOME}/lib64")
    else()
        find_package(CUDA QUIET)
        if(CUDA_FOUND)
            target_compile_definitions(${AMPCCL_TARGET_NAME} PRIVATE AMPCCL_USE_CUDA_TIMER)
            target_include_directories(${AMPCCL_TARGET_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
            target_link_libraries(${AMPCCL_TARGET_NAME} PRIVATE ${CUDA_LIBRARIES})
            set(NCCL_PATH_ADDED ON)
            message(STATUS "  NCCL/CUDA: from FindCUDA")
        endif()
    endif()
    if(NCCL_HOME AND EXISTS "${NCCL_HOME}/include")
        target_include_directories(${AMPCCL_TARGET_NAME} PRIVATE ${NCCL_HOME}/include)
        target_link_directories(${AMPCCL_TARGET_NAME} PRIVATE ${NCCL_HOME}/lib)
        message(STATUS "  NCCL: extra include=${NCCL_HOME}/include")
    endif()
    if(NOT NCCL_PATH_ADDED)
        message(FATAL_ERROR "NCCL_ONLY requires CUDA (set CUDA_HOME or ensure FindCUDA finds it)")
    endif()
else()
    # HCCL_ONLY
    if(ASCEND_HOME AND EXISTS "${ASCEND_HOME}/include")
        target_include_directories(${AMPCCL_TARGET_NAME} PRIVATE ${ASCEND_HOME}/include)
        target_link_directories(${AMPCCL_TARGET_NAME} PRIVATE ${ASCEND_HOME}/lib64 ${ASCEND_HOME}/runtime/lib64)
        target_link_libraries(${AMPCCL_TARGET_NAME} PRIVATE ascendcl runtime)
        target_compile_definitions(${AMPCCL_TARGET_NAME} PRIVATE AMPCCL_USE_ACL_TIMER)
        message(STATUS "  HCCL/Ascend: include=${ASCEND_HOME}/include, lib=${ASCEND_HOME}/lib64")
    else()
        message(FATAL_ERROR "HCCL_ONLY requires ASCEND_HOME with include (e.g. -DASCEND_HOME=/path/to/ascend-toolkit/latest)")
    endif()
endif()

# PCIe backend is required; must link pcieccl (PCCL)
if(DEFINED ENV{PCIECCL_ROOT})
    set(PCIECCL_ROOT "$ENV{PCIECCL_ROOT}")
endif()
if(NOT PCIECCL_ROOT)
    message(FATAL_ERROR "PCIe backend is required. Set PCIECCL_ROOT to pcieccl project root (e.g. export PCIECCL_ROOT=/path/to/pcieccl)")
endif()
set(PCIECCL_INCLUDE_DIR "${PCIECCL_ROOT}/include")
set(PCIECCL_LIB_DIR "${PCIECCL_ROOT}/build/lib")
if(NOT EXISTS "${PCIECCL_INCLUDE_DIR}/comm.hpp")
    message(FATAL_ERROR "PCIe backend requires comm.hpp. Not found at ${PCIECCL_INCLUDE_DIR}/comm.hpp. Build pcieccl first: cd pcieccl && make lib DEVICE=ascend")
endif()
if(NOT EXISTS "${PCIECCL_LIB_DIR}/libpcieccl_kernels.so")
    message(FATAL_ERROR "PCIe backend requires libpcieccl_kernels.so. Not found at ${PCIECCL_LIB_DIR}/libpcieccl_kernels.so. In pcieccl run: make kernels DEVICE=ascend (or make all)")
endif()
target_include_directories(${AMPCCL_TARGET_NAME} PRIVATE ${PCIECCL_INCLUDE_DIR})
target_compile_definitions(${AMPCCL_TARGET_NAME} PRIVATE LOG_LEVEL=1 AMPCCL_ENABLE_PCIE)
target_link_directories(${AMPCCL_TARGET_NAME} PRIVATE ${PCIECCL_LIB_DIR})
target_link_libraries(${AMPCCL_TARGET_NAME} PRIVATE pccl pcieccl_kernels numa rt pthread)
message(STATUS "  PCIeCCL: ${PCIECCL_ROOT} (required, linked libpccl.so + libpcieccl_kernels.so)")

# Installation
install(TARGETS ${AMPCCL_TARGET_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY libampccl/
    DESTINATION include/libampccl
    FILES_MATCHING PATTERN "*.h"
)

# Export for use in other projects
set(AMPCCL_INCLUDE_DIRS ${AMPCCL_INCLUDE_DIR})
set(AMPCCL_LIBRARIES ${AMPCCL_TARGET_NAME})

# Print configuration
message(STATUS "Adaptive-CCL Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Output: lib${AMPCCL_TARGET_NAME}.so")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  NCCL_ONLY: ${NCCL_ONLY}")
message(STATUS "  HCCL_ONLY: ${HCCL_ONLY}")
message(STATUS "  PCIe backend: required (linked)")
