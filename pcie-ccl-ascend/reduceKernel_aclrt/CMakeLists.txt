cmake_minimum_required(VERSION 3.16.0)

set(CMAKE_SYSTEM_PROCESSOR "aarch64")

if(NOT ASCEND_PATH)
    set(ASCEND_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

find_package(ASC REQUIRED)

project(pccl_aicore_reduce LANGUAGES CXX ASC)

if(NOT ASCEND_PATH)
    message(FATAL_ERROR "ASCEND_PATH is not set; source the CANN environment first")
endif()

set(AICORE_REDUCE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ASCEND_PATH}/include
    ${ASCEND_PATH}/aarch64-linux/include
    ${ASCEND_PATH}/aarch64-linux/pkg_inc
    ${ASCEND_PATH}/aarch64-linux/ascendc/include
)

add_library(pccl_aicore_reduce SHARED
    src/pccl_aicore_reduce.cc
    src/launch_reduce_aicore.asc
)

set_target_properties(pccl_aicore_reduce PROPERTIES
    LINKER_LANGUAGE CXX
    OUTPUT_NAME pccl_aicore_reduce
)

target_include_directories(pccl_aicore_reduce PRIVATE ${AICORE_REDUCE_INCLUDE_DIRS})
target_compile_definitions(pccl_aicore_reduce PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    HOST_COMPILE
)
target_compile_options(pccl_aicore_reduce PRIVATE
    -fno-common
    -fno-strict-aliasing
    -O2
    -std=c++14
    -fstack-protector-all
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201 -std=c++17 -Wno-ignored-attributes>
)
target_link_options(pccl_aicore_reduce PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)
target_link_directories(pccl_aicore_reduce PRIVATE
    ${ASCEND_PATH}/lib64
    ${ASCEND_PATH}/aarch64-linux/lib64
    ${ASCEND_PATH}/lib64/device/lib64
)
target_link_libraries(pccl_aicore_reduce PRIVATE
    -Wl,--no-as-needed
    ascendc_runtime
    profapi
    ascendalog
    ascendcl
    runtime
    mmpa
    error_manager
    ascend_dump
    c_sec
    unified_dlog
    dl
    -Wl,--no-as-needed
)

include(CTest)
if(BUILD_TESTING)
    add_executable(reduce_aicore_contract_test test/reduce_contract_test.cc)
    target_include_directories(reduce_aicore_contract_test PRIVATE ${AICORE_REDUCE_INCLUDE_DIRS})
    target_compile_options(reduce_aicore_contract_test PRIVATE -std=c++14)
    add_test(NAME reduce_aicore_contract_test COMMAND reduce_aicore_contract_test)
    add_test(NAME reduce_aicore_symbol_test
        COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test/check_reduce_symbols.sh $<TARGET_FILE:pccl_aicore_reduce>)
endif()

install(TARGETS pccl_aicore_reduce LIBRARY DESTINATION lib64)
