cmake_minimum_required(VERSION 3.18)
project(pccl VERSION 1.0 LANGUAGES CXX)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Platform selection (explicit or auto-detect)
option(PCCL_USE_CUDA "Build with CUDA support" OFF)
option(PCCL_USE_CANN "Build with Ascend CANN support" OFF)

find_package(Threads REQUIRED)

if(NOT PCCL_USE_CUDA AND NOT PCCL_USE_CANN)
    # Auto-detect CUDA
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(PCCL_USE_CUDA ON)
        message(STATUS "Auto-detected CUDA ${CUDAToolkit_VERSION}")
    endif()

    # Auto-detect Ascend CANN
    if(NOT PCCL_USE_CUDA)
        set(_ASCEND_HOME "$ENV{ASCEND_HOME}")
        if(NOT _ASCEND_HOME)
            set(_ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest")
        endif()
        if(EXISTS "${_ASCEND_HOME}/include/acl/acl.h")
            set(PCCL_USE_CANN ON)
            message(STATUS "Auto-detected Ascend CANN at ${_ASCEND_HOME}")
        endif()
        unset(_ASCEND_HOME)
    endif()

    if(NOT PCCL_USE_CUDA AND NOT PCCL_USE_CANN)
        message(FATAL_ERROR "No device backend found. Install CUDA or Ascend CANN, or set -DPCCL_USE_CUDA=ON / -DPCCL_USE_CANN=ON.")
    endif()
endif()

# Feature options
option(PCCL_BUILD_NCCL_COMPAT "Build NCCL compatibility layer" ON)
option(PCCL_TRACE "Enable Perfetto-compatible trace output" OFF)

# Log level (0=DEBUG, 1=INFO, 2=WARN, 3=ERROR)
set(PCCL_LOG_LEVEL "2" CACHE STRING "Log level (0=DEBUG, 1=INFO, 2=WARN, 3=ERROR)")

# Ascend CANN configuration (only used when PCCL_USE_CANN is ON)
set(ASCEND_PRODUCT_TYPE "Ascend910B3" CACHE STRING
    "Ascend product type (Ascend910A, Ascend310P1, Ascend310B1, Ascend910B1, Ascend910B3)")
set(ASCEND_CORE_TYPE "VectorCore" CACHE STRING
    "Ascend core type (AiCore, VectorCore)")

# SIMD auto-detection — must test that the CPU can *run* the instructions,
# not just that the compiler can emit them (cross-compile / old-CPU pitfall).
include(CheckCXXSourceRuns)

set(CMAKE_REQUIRED_FLAGS "-mavx512f")
check_cxx_source_runs("
#include <immintrin.h>
int main() {
    __m512i a = _mm512_set1_epi32(1);
    __m512i b = _mm512_add_epi32(a, a);
    return _mm512_cvtsi512_si32(b) == 2 ? 0 : 1;
}
" HAVE_AVX512)
unset(CMAKE_REQUIRED_FLAGS)

set(CMAKE_REQUIRED_FLAGS "-mavx2")
check_cxx_source_runs("
#include <immintrin.h>
int main() {
    __m256i a = _mm256_set1_epi32(1);
    __m256i b = _mm256_add_epi32(a, a);
    return 0;
}
" HAVE_AVX2)
unset(CMAKE_REQUIRED_FLAGS)

# ---- tinyxml2 (embedded) ----
add_library(tinyxml2 STATIC src/third_party/tinyxml2/tinyxml2.cpp)
target_include_directories(tinyxml2 PUBLIC src/third_party/tinyxml2)
set_target_properties(tinyxml2 PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- Core library sources ----
set(PCCL_SOURCES
    src/api/pccl_api.cpp
    src/core/comm.cpp
    src/core/config.cpp
    src/core/process_state.cpp
    src/ir/ir_loader.cpp
    src/ir/compiler.cpp
    src/topology/topology.cpp
    src/topology/numa.cpp
    src/fastpath/small_message_executor.cpp
    src/executor/graph_executor.cpp
    src/executor/h2h_engine.cpp
    src/memory/host_pool.cpp
    src/memory/shared_state.cpp
    src/memory/completion_flag.cpp
    src/memory/shm_allocator.cpp
    src/transport/tcp_transport.cpp
    src/transport/bootstrap.cpp
    src/util/inflight_drain.cpp
    src/util/sigdump.cpp
)

add_library(pccl SHARED ${PCCL_SOURCES})

target_include_directories(pccl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_compile_definitions(pccl PRIVATE LOG_LEVEL=${PCCL_LOG_LEVEL})

# Link tinyxml2
target_link_libraries(pccl PRIVATE tinyxml2)

# System libraries
target_link_libraries(pccl PRIVATE pthread rt ${CMAKE_DL_LIBS})

# NUMA support (optional)
find_library(NUMA_LIB numa)
if(NUMA_LIB)
    target_link_libraries(pccl PRIVATE ${NUMA_LIB})
    message(STATUS "Found libnuma: ${NUMA_LIB}")
else()
    message(STATUS "libnuma not found, NUMA support disabled")
endif()

# Platform-specific configuration
if(PCCL_USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(PCCL_CUDA_ARCHITECTURES "auto" CACHE STRING
        "CUDA architectures for PCCL CUDA kernels: auto, native, all, all-major, OFF, or an explicit list like 70;80")
    set_property(CACHE PCCL_CUDA_ARCHITECTURES PROPERTY STRINGS auto native all all-major OFF)

    if(PCCL_CUDA_ARCHITECTURES STREQUAL "auto" OR PCCL_CUDA_ARCHITECTURES STREQUAL "")
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set(_PCCL_RESOLVED_CUDA_ARCHITECTURES native)
        else()
            find_program(_PCCL_NVIDIA_SMI nvidia-smi)
            if(_PCCL_NVIDIA_SMI)
                execute_process(
                    COMMAND ${_PCCL_NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
                    OUTPUT_VARIABLE _PCCL_CUDA_CAPS
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET
                )
                string(REPLACE "\n" ";" _PCCL_CUDA_CAPS_LIST "${_PCCL_CUDA_CAPS}")
                foreach(_PCCL_CUDA_CAP IN LISTS _PCCL_CUDA_CAPS_LIST)
                    string(STRIP "${_PCCL_CUDA_CAP}" _PCCL_CUDA_CAP)
                    if(_PCCL_CUDA_CAP MATCHES "^([0-9]+)\\.([0-9]+)$")
                        list(APPEND _PCCL_DETECTED_CUDA_ARCHS "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
                    endif()
                endforeach()
                if(_PCCL_DETECTED_CUDA_ARCHS)
                    list(REMOVE_DUPLICATES _PCCL_DETECTED_CUDA_ARCHS)
                endif()
            endif()

            if(_PCCL_DETECTED_CUDA_ARCHS)
                set(_PCCL_RESOLVED_CUDA_ARCHITECTURES "${_PCCL_DETECTED_CUDA_ARCHS}")
            else()
                set(_PCCL_RESOLVED_CUDA_ARCHITECTURES OFF)
                message(WARNING
                    "Could not auto-detect CUDA architectures; leaving CUDA arch flags to the compiler default. "
                    "Set -DPCCL_CUDA_ARCHITECTURES=... to override.")
            endif()
        endif()
    else()
        set(_PCCL_RESOLVED_CUDA_ARCHITECTURES "${PCCL_CUDA_ARCHITECTURES}")
    endif()
    message(STATUS "PCCL CUDA architectures: ${_PCCL_RESOLVED_CUDA_ARCHITECTURES}")

    target_compile_definitions(pccl PUBLIC PCCL_DEVICE_CUDA)
    target_link_libraries(pccl PRIVATE CUDA::cudart CUDA::cuda_driver)

    # Build CUDA kernel
    add_library(pccl_kernels SHARED kernels/wait_kernel.cu)
    set_target_properties(pccl_kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RUNTIME_LIBRARY Shared
        CUDA_ARCHITECTURES "${_PCCL_RESOLVED_CUDA_ARCHITECTURES}"
    )
    target_link_libraries(pccl_kernels PRIVATE CUDA::cudart)
    target_link_libraries(pccl PRIVATE pccl_kernels)
elseif(PCCL_USE_CANN)
    target_compile_definitions(pccl PUBLIC PCCL_DEVICE_ASCEND)

    set(ASCEND_HOME "$ENV{ASCEND_HOME}" CACHE PATH "Ascend toolkit path")
    if(NOT ASCEND_HOME)
        set(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest")
    endif()
    target_include_directories(pccl PUBLIC ${ASCEND_HOME}/include)
    target_link_directories(pccl PUBLIC ${ASCEND_HOME}/lib64)
    target_link_libraries(pccl PUBLIC ascendcl)

    # ---- Build Ascend kernel (libpccl_kernels.so) via ccec ----

    # Locate ccec compiler — two known paths depending on toolkit layout
    set(_CCEC_PATH1 "${ASCEND_HOME}/toolkit/tools/ccec_compiler/bin/ccec")
    set(_CCEC_PATH2 "${ASCEND_HOME}/compiler/ccec_compiler/bin/ccec")
    if(EXISTS "${_CCEC_PATH1}")
        set(CCEC "${_CCEC_PATH1}")
    elseif(EXISTS "${_CCEC_PATH2}")
        set(CCEC "${_CCEC_PATH2}")
    else()
        set(CCEC "${_CCEC_PATH1}")
        message(WARNING "ccec compiler not found at either ${_CCEC_PATH1} or ${_CCEC_PATH2}, build may fail")
    endif()
    unset(_CCEC_PATH1)
    unset(_CCEC_PATH2)
    message(STATUS "Using ccec compiler: ${CCEC}")

    # Architecture flag based on product type and core type
    if(ASCEND_PRODUCT_TYPE STREQUAL "Ascend910A")
        set(_ARCH_FLAG "--cce-aicore-arch=dav-c100")
    elseif(ASCEND_PRODUCT_TYPE STREQUAL "Ascend310P1")
        if(ASCEND_CORE_TYPE STREQUAL "AiCore")
            set(_ARCH_FLAG "--cce-aicore-arch=dav-m200")
        else()
            set(_ARCH_FLAG "--cce-aicore-arch=dav-m200-vec")
        endif()
    elseif(ASCEND_PRODUCT_TYPE STREQUAL "Ascend310B1")
        if(ASCEND_CORE_TYPE STREQUAL "AiCore")
            set(_ARCH_FLAG "--cce-aicore-arch=dav-m300")
        else()
            set(_ARCH_FLAG "--cce-aicore-arch=dav-m300-vec")
        endif()
    elseif(ASCEND_PRODUCT_TYPE STREQUAL "Ascend910B1" OR
           ASCEND_PRODUCT_TYPE STREQUAL "Ascend910B3")
        if(ASCEND_CORE_TYPE STREQUAL "AiCore")
            set(_ARCH_FLAG "--cce-aicore-arch=dav-c220-cube")
        else()
            set(_ARCH_FLAG "--cce-aicore-arch=dav-c220-vec")
        endif()
    else()
        message(FATAL_ERROR "Unknown ASCEND_PRODUCT_TYPE: ${ASCEND_PRODUCT_TYPE}")
    endif()
    message(STATUS "Ascend arch flag: ${_ARCH_FLAG}")

    # Include paths for tikcpp (both compiler/ and tools/ layouts)
    set(_CCEC_INCLUDES
        -I${ASCEND_HOME}/include
        -I${ASCEND_HOME}/compiler/tikcpp/tikcfw
        -I${ASCEND_HOME}/compiler/tikcpp/tikcfw/impl
        -I${ASCEND_HOME}/compiler/tikcpp/tikcfw/interface
        -I${ASCEND_HOME}/tools/tikcpp/tikcfw
        -I${ASCEND_HOME}/tools/tikcpp/tikcfw/impl
        -I${ASCEND_HOME}/tools/tikcpp/tikcfw/interface
        -I${ASCEND_HOME}/acllib/include
    )

    # Compile wait_kernel_ascendc.cpp -> .o
    set(_KERNEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/kernels/wait_kernel_ascendc.cpp")
    set(_KERNEL_OBJ "${CMAKE_CURRENT_BINARY_DIR}/wait_kernel_ascendc.o")
    set(_KERNEL_LIB "${CMAKE_CURRENT_BINARY_DIR}/libpccl_kernels.so")

    add_custom_command(
        OUTPUT "${_KERNEL_OBJ}"
        COMMAND ${CCEC} -xcce -O2 -std=c++17 ${_ARCH_FLAG} --cce-auto-sync
                ${_CCEC_INCLUDES} -fPIC
                -c "${_KERNEL_SRC}" -o "${_KERNEL_OBJ}"
        DEPENDS "${_KERNEL_SRC}"
        COMMENT "Compiling Ascend kernel with ccec"
        VERBATIM
    )

    # Link .o -> libpccl_kernels.so (fatobj-link embeds kernel binary)
    add_custom_command(
        OUTPUT "${_KERNEL_LIB}"
        COMMAND ${CCEC} -shared --cce-fatobj-link
                -L${ASCEND_HOME}/lib64
                -L${ASCEND_HOME}/runtime/lib64
                -L${ASCEND_HOME}/tools/simulator/${ASCEND_PRODUCT_TYPE}/lib
                -L${ASCEND_HOME}/tools/tikicpulib/lib/${ASCEND_PRODUCT_TYPE}
                "${_KERNEL_OBJ}"
                -lstdc++ -lruntime -lascendcl
                -o "${_KERNEL_LIB}"
        DEPENDS "${_KERNEL_OBJ}"
        COMMENT "Linking Ascend kernel library (libpccl_kernels.so)"
        VERBATIM
    )

    add_custom_target(pccl_kernels ALL DEPENDS "${_KERNEL_LIB}")

    # Make pccl depend on the kernel library and link it
    add_dependencies(pccl pccl_kernels)
    target_link_directories(pccl PRIVATE ${ASCEND_HOME}/runtime/lib64)
    target_link_libraries(pccl PRIVATE "${_KERNEL_LIB}" runtime)

    unset(_ARCH_FLAG)
    unset(_CCEC_INCLUDES)
endif()

# SIMD compiler flags
if(HAVE_AVX512)
    target_compile_options(pccl PRIVATE -mavx512f -mavx512bw)
    message(STATUS "AVX-512 support enabled")
elseif(HAVE_AVX2)
    target_compile_options(pccl PRIVATE -mavx2)
    message(STATUS "AVX2 support enabled")
endif()

# Perfetto-compatible trace output (works with both CUDA and CANN)
if(PCCL_TRACE)
    target_compile_definitions(pccl PRIVATE PCCL_TRACE_ENABLED=1)
    message(STATUS "Perfetto trace output enabled")
endif()

# NCCL compatibility layer
if(PCCL_BUILD_NCCL_COMPAT)
    target_sources(pccl PRIVATE
        src/api/nccl_compat/nccl_hook.cpp
        src/api/nccl_compat/compat_common.cpp
        src/api/nccl_compat/host_fallback.cpp
    )
    # Interpose host-blocking CUDA runtime calls to drain in-flight PCCL collectives
    # before the user thread parks inside the driver (avoids the driver-lock deadlock).
    if(PCCL_USE_CUDA)
        target_sources(pccl PRIVATE
            src/api/nccl_compat/cuda_block_shim.cpp
        )
    endif()
endif()

# ---- Tests (always built when a GPU backend is available) ----
if(PCCL_USE_CUDA OR PCCL_USE_CANN)
    # Collective communication tests (pass <nranks> as first argument at runtime)
    set(COLL_TESTS
        test_allreduce
        test_allgather
        test_reducescatter
        test_alltoall
        test_allgather_strided
        test_reducescatter_strided
        test_alltoall_strided
    )
    foreach(test ${COLL_TESTS})
        add_executable(${test} tests/coll/${test}.cpp)
        target_link_libraries(${test} PRIVATE pccl)
        target_include_directories(${test} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/common)
        if(PCCL_USE_CUDA)
            target_link_libraries(${test} PRIVATE CUDA::cudart)
        endif()
    endforeach()

    # Multi-process collective tests (require MPI)
    find_package(MPI QUIET)
    if(MPI_CXX_FOUND)
        set(MP_COLL_TESTS test_allreduce_mp test_allgather_mp test_reducescatter_mp test_alltoall_mp test_broadcast_mp repro_ag_inplace_mp)
        foreach(test ${MP_COLL_TESTS})
            add_executable(${test} tests/coll/${test}.cpp)
            target_link_libraries(${test} PRIVATE pccl MPI::MPI_CXX)
            target_include_directories(${test} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/src
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/common)
            if(PCCL_USE_CUDA)
                target_link_libraries(${test} PRIVATE CUDA::cudart)
            endif()
        endforeach()
        message(STATUS "MPI found: multi-process tests enabled")
    else()
        message(STATUS "MPI not found: multi-process tests disabled")
    endif()

    if(PCCL_USE_CUDA AND PCCL_BUILD_NCCL_COMPAT)
        add_executable(test_nccl_compat_allreduce_count1
            tests/nccl_compat/test_nccl_compat_allreduce_count1.cpp)
        target_link_libraries(test_nccl_compat_allreduce_count1 PRIVATE pccl CUDA::cudart Threads::Threads)
        target_include_directories(test_nccl_compat_allreduce_count1 PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include)
    endif()

    # Device DMA tests
    set(DEV_TESTS test_gpu_dma_single test_gpu_dma_all test_gpu_dma_ag_0011 test_gpu_dma_ag_00112233 test_h2h)
    foreach(test ${DEV_TESTS})
        add_executable(${test} tests/dev/${test}.cpp)
        target_link_libraries(${test} PRIVATE pccl Threads::Threads)
        target_include_directories(${test} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/common)
        if(PCCL_USE_CUDA)
            target_link_libraries(${test} PRIVATE CUDA::cudart)
        endif()
    endforeach()

    if(PCCL_USE_CUDA)
        find_library(NCCL_LIB nccl
            PATHS /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/local/cuda/targets/x86_64-linux/lib)
        if(NCCL_LIB)
            add_executable(test_nccl_compute_overlap tests/dev/test_nccl_compute_overlap.cu)
            set_target_properties(test_nccl_compute_overlap PROPERTIES
                CUDA_SEPARABLE_COMPILATION OFF
                CUDA_RUNTIME_LIBRARY Shared
                CUDA_ARCHITECTURES "${_PCCL_RESOLVED_CUDA_ARCHITECTURES}"
            )
            target_link_libraries(test_nccl_compute_overlap PRIVATE CUDA::cudart ${NCCL_LIB} Threads::Threads)
            if(MPI_CXX_FOUND)
                add_executable(test_nccl_compute_overlap_mp tests/dev/test_nccl_compute_overlap_mp.cu)
                set_target_properties(test_nccl_compute_overlap_mp PROPERTIES
                    CUDA_SEPARABLE_COMPILATION OFF
                    CUDA_RUNTIME_LIBRARY Shared
                    CUDA_ARCHITECTURES "${_PCCL_RESOLVED_CUDA_ARCHITECTURES}"
                )
                target_link_libraries(test_nccl_compute_overlap_mp PRIVATE CUDA::cudart ${NCCL_LIB} MPI::MPI_CXX Threads::Threads)
            else()
                message(WARNING "MPI not found: test_nccl_compute_overlap_mp disabled")
            endif()
        else()
            message(WARNING "libnccl not found: test_nccl_compute_overlap disabled")
        endif()
    endif()
endif()

# ---- Symlink ir/ into build dir so tests work from build/ ----
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/ir
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${CMAKE_SOURCE_DIR}/ir ${CMAKE_BINARY_DIR}/ir
    COMMENT "Symlinking ir/ into build directory"
)
add_custom_target(symlink_ir ALL DEPENDS ${CMAKE_BINARY_DIR}/ir)

# ---- Installation ----
install(TARGETS pccl EXPORT pcclTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
if(PCCL_USE_CUDA AND TARGET pccl_kernels)
    install(TARGETS pccl_kernels EXPORT pcclTargets
        LIBRARY DESTINATION lib
    )
elseif(PCCL_USE_CANN)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libpccl_kernels.so" DESTINATION lib)
endif()
install(DIRECTORY include/pccl DESTINATION include)
install(DIRECTORY ir DESTINATION share/pccl)
install(EXPORT pcclTargets
    FILE pcclConfig.cmake
    NAMESPACE pccl::
    DESTINATION lib/cmake/pccl
)
