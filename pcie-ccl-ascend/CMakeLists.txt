cmake_minimum_required(VERSION 3.18)
project(pccl VERSION 1.0 LANGUAGES CXX)

option(PCCL_ENABLE_AICORE_REDUCE "Build the AICore local reduce implementation" ON)

if(PCCL_ENABLE_AICORE_REDUCE)
    find_package(ASC REQUIRED)
    enable_language(ASC)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "Ascend CANN toolkit path")
set(ASCEND_PRODUCT_TYPE "Ascend910B3" CACHE STRING "Ascend product type")
set(ASCEND_CORE_TYPE "VectorCore" CACHE STRING "Ascend core type")
set(PCCL_LOG_LEVEL "2" CACHE STRING "Log level (0=DEBUG, 1=INFO, 2=WARN, 3=ERROR)")
set(PCCL_DMA_LANES_PER_DIRECTION "12" CACHE STRING "Max D2H/H2D DMA streams per direction")
set(PCCL_SUBCHUNK_SLICE_SIZE "1048576" CACHE STRING "Graph executor subchunk slice size in bytes")
option(PCCL_TRACE "Enable Perfetto-compatible trace output" OFF)
option(PCCL_BUILD_TESTS "Build native PCCL smoke/correctness tests" ON)
option(PCCL_BUILD_HCCL_TEST "Build bundled CANN hccl_test binaries" ON)

find_package(Threads REQUIRED)
include(CTest)

if(NOT EXISTS "${ASCEND_HOME}/include/acl/acl.h")
    message(FATAL_ERROR "Ascend CANN headers not found under ASCEND_HOME=${ASCEND_HOME}")
endif()

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
    return _mm256_extract_epi32(b, 0) == 2 ? 0 : 1;
}
" HAVE_AVX2)
unset(CMAKE_REQUIRED_FLAGS)

add_library(tinyxml2 STATIC src/third_party/tinyxml2/tinyxml2.cpp)
target_include_directories(tinyxml2 PUBLIC src/third_party/tinyxml2)
set_target_properties(tinyxml2 PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(PCCL_SOURCES
    src/api/pccl_api.cpp
    src/api/hccl_compat/compat_common.cpp
    src/api/hccl_compat/hccl_hook.cpp
    src/core/comm.cpp
    src/core/config.cpp
    src/core/process_state.cpp
    src/executor/graph_executor.cpp
    src/executor/h2h_engine.cpp
    src/ir/compiler.cpp
    src/ir/ir_loader.cpp
    src/memory/completion_flag.cpp
    src/memory/host_pool.cpp
    src/memory/shared_state.cpp
    src/memory/shm_allocator.cpp
    src/runtime/ascend_runtime.cpp
    src/topology/numa.cpp
    src/topology/topology.cpp
    src/util/inflight_drain.cpp
    src/util/sigdump.cpp
)

if(PCCL_ENABLE_AICORE_REDUCE)
    list(APPEND PCCL_SOURCES
        reduceKernel_aclrt/src/pccl_aicore_reduce.cc
        reduceKernel_aclrt/src/launch_reduce_aicore.asc
    )
endif()

add_library(pccl SHARED ${PCCL_SOURCES})
target_include_directories(pccl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${ASCEND_HOME}/include>
    $<INSTALL_INTERFACE:include>
)
if(PCCL_ENABLE_AICORE_REDUCE)
    target_include_directories(pccl PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/reduceKernel_aclrt/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/reduceKernel_aclrt/src
        ${ASCEND_HOME}/aarch64-linux/include
        ${ASCEND_HOME}/aarch64-linux/pkg_inc
        ${ASCEND_HOME}/aarch64-linux/ascendc/include
    )
    target_compile_definitions(pccl PRIVATE PCCL_AICORE_REDUCE_ENABLED=1)
endif()
target_compile_definitions(pccl PUBLIC PCCL_ASCEND_ONLY PCCL_DEVICE_ASCEND)
target_compile_definitions(pccl PRIVATE
    LOG_LEVEL=${PCCL_LOG_LEVEL}
    PCCL_DMA_LANES_PER_DIRECTION=${PCCL_DMA_LANES_PER_DIRECTION}
    PCCL_SUBCHUNK_SLICE_SIZE=${PCCL_SUBCHUNK_SLICE_SIZE}
)
target_link_directories(pccl PUBLIC ${ASCEND_HOME}/lib64 ${ASCEND_HOME}/runtime/lib64)
if(PCCL_ENABLE_AICORE_REDUCE)
    set(ASCEND_DRIVER_LIB_DIR "/usr/local/Ascend/driver/lib64/driver" CACHE PATH
        "Ascend driver library directory used at runtime")
    target_link_directories(pccl PRIVATE
        ${ASCEND_HOME}/aarch64-linux/lib64
        ${ASCEND_DRIVER_LIB_DIR}
    )
endif()
target_link_libraries(pccl PRIVATE tinyxml2 ascendcl runtime Threads::Threads rt ${CMAKE_DL_LIBS})
if(PCCL_ENABLE_AICORE_REDUCE)
    target_link_libraries(pccl PRIVATE
        ascendc_runtime
        profapi
        ascendalog
        mmpa
        error_manager
        ascend_dump
        c_sec
        unified_dlog
    )
    target_compile_options(pccl PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201 -std=c++17 -Wno-ignored-attributes>
    )
endif()

find_library(NUMA_LIB numa)
if(NUMA_LIB)
    target_link_libraries(pccl PRIVATE ${NUMA_LIB})
    message(STATUS "Found libnuma: ${NUMA_LIB}")
endif()

if(HAVE_AVX512)
    target_compile_options(pccl PRIVATE -mavx512f -mavx512bw)
elseif(HAVE_AVX2)
    target_compile_options(pccl PRIVATE -mavx2)
endif()

if(PCCL_TRACE)
    target_compile_definitions(pccl PRIVATE PCCL_TRACE_ENABLED=1)
endif()

add_custom_command(
    TARGET pccl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE_NAME:pccl>"
            "$<TARGET_FILE_DIR:pccl>/libhccl.so"
    COMMENT "Creating HCCL compatibility library symlink"
)

if(PCCL_BUILD_TESTS)
    add_executable(test_completion_notification tests/unit/test_completion_notification.cpp)
    target_include_directories(test_completion_notification PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${ASCEND_HOME}/include)
    target_link_libraries(test_completion_notification PRIVATE pccl ascendcl Threads::Threads ${CMAKE_DL_LIBS})
    target_link_options(test_completion_notification PRIVATE -rdynamic)
    # These use a real NPU; invoke explicitly, not concurrently in ctest.

    add_executable(test_h2h_completion_publication tests/unit/test_h2h_completion_publication.cpp)
    target_include_directories(test_h2h_completion_publication PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_h2h_completion_publication PRIVATE pccl Threads::Threads ${CMAKE_DL_LIBS})
    target_link_options(test_h2h_completion_publication PRIVATE -rdynamic)
    add_test(NAME h2h_completion_publication COMMAND test_h2h_completion_publication)
    set_tests_properties(h2h_completion_publication PROPERTIES TIMEOUT 15)

    add_executable(test_host_chunk_compaction tests/unit/test_host_chunk_compaction.cpp)
    target_link_libraries(test_host_chunk_compaction PRIVATE pccl)
    target_include_directories(test_host_chunk_compaction PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME host_chunk_compaction COMMAND test_host_chunk_compaction)

    add_executable(test_internal_buffer_budget tests/unit/test_internal_buffer_budget.cpp)
    target_include_directories(test_internal_buffer_budget PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME internal_buffer_budget COMMAND test_internal_buffer_budget)

    add_executable(test_comm_memory_isolation tests/unit/test_comm_memory_isolation.cpp)
    target_include_directories(test_comm_memory_isolation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME comm_memory_isolation COMMAND test_comm_memory_isolation)

    add_executable(test_comm_topology_projection tests/unit/test_comm_topology_projection.cpp)
    target_include_directories(test_comm_topology_projection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME comm_topology_projection COMMAND test_comm_topology_projection)

    add_executable(test_shared_state_slots tests/unit/test_shared_state_slots.cpp)
    target_link_libraries(test_shared_state_slots PRIVATE pccl)
    target_include_directories(test_shared_state_slots PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME shared_state_slots COMMAND test_shared_state_slots)

    add_executable(test_host_staging_lease tests/unit/test_host_staging_lease.cpp)
    target_link_libraries(test_host_staging_lease PRIVATE pccl)
    target_include_directories(test_host_staging_lease PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME host_staging_lease COMMAND test_host_staging_lease)

    add_executable(test_dma_lane_policy tests/unit/test_dma_lane_policy.cpp)
    target_include_directories(test_dma_lane_policy PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${ASCEND_HOME}/include
    )
    add_test(NAME dma_lane_policy COMMAND test_dma_lane_policy)

    set(STRIDED_COLL_TESTS
        test_allgather_strided
        test_reducescatter_strided
        test_alltoall_strided
        test_strided_baseline_perf
    )
    foreach(test ${STRIDED_COLL_TESTS})
        add_executable(${test} tests/coll/${test}.cpp)
        target_link_libraries(${test} PRIVATE pccl)
        target_include_directories(${test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests/common)
    endforeach()

    add_executable(bench_alltoall_lanes tests/coll/bench_alltoall_lanes.cpp)
    target_link_libraries(bench_alltoall_lanes PRIVATE pccl)
    target_include_directories(bench_alltoall_lanes PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/common
    )

    if(PCCL_ENABLE_AICORE_REDUCE)
        add_executable(test_aicore_reduce_device reduceKernel_aclrt/test/reduce_device_correctness.cc)
        target_link_libraries(test_aicore_reduce_device PRIVATE pccl ascendcl runtime)
        target_include_directories(test_aicore_reduce_device PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/reduceKernel_aclrt/inc
        )
        add_test(NAME aicore_reduce_device_correctness COMMAND test_aicore_reduce_device)

        add_executable(benchmark_aicore_reduce_device reduceKernel_aclrt/test/reduce_device_benchmark.cc)
        target_link_libraries(benchmark_aicore_reduce_device PRIVATE pccl ascendcl runtime)
        target_include_directories(benchmark_aicore_reduce_device PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/reduceKernel_aclrt/inc
        )
    endif()

    add_executable(bench_hccl_compat_sp tests/coll/bench_hccl_compat_sp.cpp)
    target_include_directories(bench_hccl_compat_sp PRIVATE ${ASCEND_HOME}/include)
    target_link_directories(bench_hccl_compat_sp PRIVATE ${ASCEND_HOME}/lib64 ${ASCEND_HOME}/runtime/lib64)
    target_link_libraries(bench_hccl_compat_sp PRIVATE hccl ascendcl Threads::Threads)
    set_target_properties(bench_hccl_compat_sp PROPERTIES
        BUILD_RPATH "${ASCEND_HOME}/lib64;${ASCEND_HOME}/runtime/lib64"
    )

    if(PCCL_BUILD_HCCL_TEST)
        find_package(MPI QUIET)
        if(MPI_CXX_FOUND)
            set(HCCL_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/hccl_test)
            file(GLOB HCCL_TEST_COMMON_SOURCES CONFIGURE_DEPENDS
                ${HCCL_TEST_DIR}/common/src/*.cc
            )
            set(HCCL_TEST_CASES
                all_gather:hccl_allgather_rootinfo_test.cc
                all_gatherv:hccl_allgatherv_rootinfo_test.cc
                all_reduce:hccl_allreduce_rootinfo_test.cc
                alltoallv:hccl_alltoallv_rootinfo_test.cc
                alltoall:hccl_alltoall_rootinfo_test.cc
                broadcast:hccl_brocast_rootinfo_test.cc
                reduce_scatter:hccl_reducescatter_rootinfo_test.cc
                reduce_scatterv:hccl_reducescatterv_rootinfo_test.cc
                reduce:hccl_reduce_rootinfo_test.cc
                scatter:hccl_scatter_rootinfo_test.cc
                alltoallvc:hccl_alltoallvc_rootinfo_test.cc
            )
            set(HCCL_TEST_TARGETS)
            foreach(case ${HCCL_TEST_CASES})
                string(REPLACE ":" ";" case_parts ${case})
                list(GET case_parts 0 case_name)
                list(GET case_parts 1 case_source)
                set(target hccl_test_${case_name})
                add_executable(${target}
                    ${HCCL_TEST_COMMON_SOURCES}
                    ${HCCL_TEST_DIR}/opbase_test/${case_source}
                )
                set_target_properties(${target} PROPERTIES
                    OUTPUT_NAME ${case_name}_test
                    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/hccl_test/bin
                    BUILD_RPATH "${ASCEND_HOME}/lib64;${ASCEND_HOME}/runtime/lib64"
                )
                target_compile_features(${target} PRIVATE cxx_std_11)
                target_compile_options(${target} PRIVATE
                    -fstack-protector-strong
                    -fPIE
                )
                target_link_options(${target} PRIVATE
                    -pie
                    -Wl,-z,relro
                    -Wl,-z,now
                    -Wl,-z,noexecstack
                    -Wl,--copy-dt-needed-entries
                )
                target_include_directories(${target} PRIVATE
                    ${HCCL_TEST_DIR}/common/src
                    ${HCCL_TEST_DIR}/opbase_test
                    ${ASCEND_HOME}/include
                )
                target_link_directories(${target} PRIVATE ${ASCEND_HOME}/lib64 ${ASCEND_HOME}/runtime/lib64)
                target_link_libraries(${target} PRIVATE hccl acl_rt msprofiler MPI::MPI_CXX)
                list(APPEND HCCL_TEST_TARGETS ${target})
            endforeach()
            add_custom_target(hccl_test DEPENDS ${HCCL_TEST_TARGETS})
            message(STATUS "Bundled hccl_test build enabled")
        else()
            message(WARNING "MPI CXX not found: bundled hccl_test build disabled")
        endif()
    endif()
endif()

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/ir
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR}/ir ${CMAKE_BINARY_DIR}/ir
    COMMENT "Symlinking ir/ into build directory"
)
add_custom_target(symlink_ir ALL DEPENDS ${CMAKE_BINARY_DIR}/ir)

install(TARGETS pccl EXPORT pcclTargets LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(CODE "execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink libpccl.so libhccl.so WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/lib)")
install(DIRECTORY include/pccl include/hccl DESTINATION include)
install(DIRECTORY ir DESTINATION share/pccl)
install(EXPORT pcclTargets FILE pcclConfig.cmake NAMESPACE pccl:: DESTINATION lib/cmake/pccl)
