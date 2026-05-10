# Root Makefile for PCIeCCL
#
# Build Instructions:
#   make all       - Build everything (kernels, library, tests)
#   make lib       - Build libpccl.so shared library
#   make kernels   - Build kernel library (libpcieccl_kernels.so)
#   make test      - Build test programs
#   make clean     - Clean all build artifacts
#
# Environment Variables:
#   LOG_LEVEL      - Logging level: 0=DEBUG, 1=INFO (default), 2=WARN, 3=ERROR
#   DEVICE         - Device type: cuda or ascend (auto-detected if not specified)
#   CUDA_HOME      - Path to CUDA toolkit (default: /usr/local/cuda)
#   ASCEND_HOME    - Path to Ascend toolkit (default: /usr/local/Ascend/ascend-toolkit/latest)

# Compiler
CXX = g++

# Logging level
LOG_LEVEL ?= 0

# Debug mode (DEBUG=1 for debug build with -g -O0)
DEBUG ?= 0

# Build directories
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
LIB_DIR = $(BUILD_DIR)/lib
TEST_DIR = $(BUILD_DIR)/tests

# Device toolkit paths (defaults)
CUDA_HOME ?= /usr/local/cuda
ASCEND_HOME ?= /usr/local/Ascend/ascend-toolkit/latest

# Auto-detect device type based on toolkit availability
# Priority: CUDA > Ascend (can be overridden with DEVICE=cuda or DEVICE=ascend)
ifndef DEVICE
  ifneq ($(wildcard $(CUDA_HOME)),)
    DEVICE = cuda
  else ifneq ($(wildcard $(ASCEND_HOME)),)
    DEVICE = ascend
  else
    $(error No device toolkit found! Install CUDA at $(CUDA_HOME) or Ascend at $(ASCEND_HOME), or set DEVICE explicitly)
  endif
endif

# Device-specific configuration
ifeq ($(DEVICE),cuda)
  DEVICE_INCLUDE = $(CUDA_HOME)/include
  DEVICE_LIB = $(CUDA_HOME)/lib64
  DEVICE_FLAGS = -DPCCL_DEVICE_CUDA=1
  DEVICE_LIBS = -lcudart
  DEVICE_RUNTIME_LIB =
else ifeq ($(DEVICE),ascend)
  DEVICE_INCLUDE = $(ASCEND_HOME)/include
  DEVICE_LIB = $(ASCEND_HOME)/lib64
  DEVICE_FLAGS = -DPCCL_DEVICE_ASCEND=1
  DEVICE_LIBS = -lascendcl
  DEVICE_RUNTIME_LIB = -lruntime
else
  $(error Unsupported DEVICE=$(DEVICE). Use DEVICE=cuda or DEVICE=ascend)
endif

# Architecture detection and flags
ARCH := $(shell uname -m)
ifeq ($(ARCH),aarch64)
    ARCH_FLAGS = -march=native
else ifeq ($(ARCH),x86_64)
    # Auto-detect AVX-512/AVX2 support
    HAS_AVX512 := $(shell grep -q avx512f /proc/cpuinfo 2>/dev/null && echo 1 || echo 0)
    HAS_AVX2 := $(shell grep -q avx2 /proc/cpuinfo 2>/dev/null && echo 1 || echo 0)
    ifeq ($(HAS_AVX512),1)
        ARCH_FLAGS = -march=native -mavx512f
    else ifeq ($(HAS_AVX2),1)
        ARCH_FLAGS = -march=native -mavx2
    else
        ARCH_FLAGS = -march=native
    endif
else
    ARCH_FLAGS =
endif

# Compiler flags for library
ifeq ($(DEBUG),1)
  OPT_FLAGS = -g -O0
else
  OPT_FLAGS = -O3
endif

CXXFLAGS = -std=c++11 -Wall -Wextra -fPIC $(OPT_FLAGS) \
           -I$(CURDIR) -I$(DEVICE_INCLUDE) \
           -DLOG_LEVEL=$(LOG_LEVEL) $(DEVICE_FLAGS) $(ARCH_FLAGS)

# NVTX profiling support (opt-in, CUDA only)
NVTX ?= 0
ifeq ($(NVTX),1)
  ifeq ($(DEVICE),cuda)
    CXXFLAGS += -DPCCL_NVTX_ENABLED=1
  endif
endif

# Single-process multi-thread mode (uses devMallocHost instead of shared memory)
SINGLE_PROCESS ?= 1
ifeq ($(SINGLE_PROCESS),1)
  CXXFLAGS += -DPCCL_SINGLE_PROCESS=1
endif

# Linker flags for shared library
LDFLAGS_LIB = -shared -L$(DEVICE_LIB) -Wl,-rpath,$(DEVICE_LIB)
LIBS_LIB = $(DEVICE_LIBS) -lnuma -lrt -pthread

ifeq ($(NVTX),1)
  ifeq ($(DEVICE),cuda)
    LIBS_LIB += -lnvToolsExt
  endif
endif

# Linker flags for test programs
# Use $ORIGIN to find libraries relative to executable location
LDFLAGS_TEST = -L$(LIB_DIR) -L$(DEVICE_LIB) -Wl,-rpath,'$$ORIGIN/../lib' -Wl,-rpath,$(DEVICE_LIB)
LIBS_TEST_BASIC = -lpccl $(DEVICE_LIBS) -lnuma -pthread
LIBS_TEST_ALLGATHER = -lpccl $(DEVICE_LIBS) -lnuma -lrt -lpcieccl_kernels $(DEVICE_RUNTIME_LIB) -pthread

# Source files for libpccl.so
LIB_SRCS = src/comm.cpp \
           src/scheduler.cpp \
           src/state_manager.cpp \
           src/host_memory_pool.cpp \
           src/mem_op_manager.cpp \
           src/pipeline_reduce_manager.cpp \
           src/numa_topology.cpp \
           src/sync_flags.cpp \
           src/transfer_tasks.cpp

# Object files
LIB_OBJS = $(patsubst src/%.cpp,$(OBJ_DIR)/%.o,$(LIB_SRCS))

# Test executables
TEST_BINS = $(TEST_DIR)/test_hal_d2h \
            $(TEST_DIR)/test_hal_h2h \
            $(TEST_DIR)/test_2gpu_allgather \
            $(TEST_DIR)/test_ir_d2h \
            $(TEST_DIR)/test_d2h_h2d_pipeline \
            $(TEST_DIR)/test_2chunk_d2h_h2d_pipeline \
            $(TEST_DIR)/test_4gpu_allgather \
            $(TEST_DIR)/test_4gpu_allreduce

ifeq ($(SINGLE_PROCESS),1)
TEST_BINS += $(TEST_DIR)/test_4gpu_allgather_st \
             $(TEST_DIR)/test_4gpu_allreduce_st \
             $(TEST_DIR)/test_4gpu_allgather_stress_st \
             $(TEST_DIR)/test_2gpu_allreduce_st \
             $(TEST_DIR)/test_2gpu_allgather_st \
             $(TEST_DIR)/test_perf_st
endif

# Output libraries
PCCL_LIB = $(LIB_DIR)/libpccl.so
KERNEL_LIB = $(LIB_DIR)/libpcieccl_kernels.so

.PHONY: all lib kernels test clean help dirs

# Default target
all: kernels lib test

# Create build directories
dirs:
	@mkdir -p $(OBJ_DIR) $(LIB_DIR) $(TEST_DIR)

# Build main library
lib: $(PCCL_LIB)

$(PCCL_LIB): $(LIB_OBJS) | dirs
	$(CXX) $(LDFLAGS_LIB) $^ $(LIBS_LIB) -o $@
	@echo "Built: $@"

# Compile source files to object files
$(OBJ_DIR)/%.o: src/%.cpp | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Build kernel library (device-specific)
kernels: $(KERNEL_LIB)

ifeq ($(DEVICE),cuda)
$(KERNEL_LIB): | dirs
	@$(MAKE) -C kernels cuda_lib CUDA_HOME=$(CUDA_HOME)
	@cp kernels/lib/libpcieccl_kernels.so $(LIB_DIR)/
	@echo "Copied CUDA kernel library to $(LIB_DIR)/"
else
$(KERNEL_LIB): | dirs
	@$(MAKE) -C kernels lib ASCEND_HOME=$(ASCEND_HOME)
	@cp kernels/lib/libpcieccl_kernels.so $(LIB_DIR)/
	@echo "Copied Ascend kernel library to $(LIB_DIR)/"
endif

# Build test programs (depends on lib and kernels)
test: lib $(TEST_BINS)

# HAL-only tests (no pccl dependency)
$(TEST_DIR)/test_hal_d2h: tests/test_hal_d2h.cpp | dirs
	$(CXX) $(CXXFLAGS) $< -L$(DEVICE_LIB) -Wl,-rpath,$(DEVICE_LIB) $(DEVICE_LIBS) -lnuma -pthread -o $@
	@echo "Built: $@"

$(TEST_DIR)/test_hal_h2h: tests/test_hal_h2h.cpp | dirs
	$(CXX) $(CXXFLAGS) $< -lnuma -pthread -o $@
	@echo "Built: $@"

# Scheduling tests (full stack with pccl + kernels)
$(TEST_DIR)/test_2gpu_allgather: tests/test_2gpu_allgather.cpp $(PCCL_LIB) $(KERNEL_LIB) | dirs
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS_TEST) $(LIBS_TEST_ALLGATHER) -o $@
	@echo "Built: $@"

$(TEST_DIR)/test_ir_d2h: tests/test_ir_d2h.cpp $(PCCL_LIB) $(KERNEL_LIB) | dirs
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS_TEST) $(LIBS_TEST_ALLGATHER) -o $@
	@echo "Built: $@"

$(TEST_DIR)/test_d2h_h2d_pipeline: tests/test_d2h_h2d_pipeline.cpp $(PCCL_LIB) $(KERNEL_LIB) | dirs
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS_TEST) $(LIBS_TEST_ALLGATHER) -o $@
	@echo "Built: $@"

$(TEST_DIR)/test_2chunk_d2h_h2d_pipeline: tests/test_2chunk_d2h_h2d_pipeline.cpp $(PCCL_LIB) $(KERNEL_LIB) | dirs
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS_TEST) $(LIBS_TEST_ALLGATHER) -o $@
	@echo "Built: $@"

$(TEST_DIR)/test_4gpu_allgather: tests/test_4gpu_allgather.cpp $(PCCL_LIB) $(KERNEL_LIB) | dirs
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS_TEST) $(LIBS_TEST_ALLGATHER) -o $@
	@echo "Built: $@"

$(TEST_DIR)/test_4gpu_allreduce: tests/test_4gpu_allreduce.cpp $(PCCL_LIB) $(KERNEL_LIB) | dirs
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS_TEST) $(LIBS_TEST_ALLGATHER) -o $@
	@echo "Built: $@"

$(TEST_DIR)/test_4gpu_allgather_st: tests/test_4gpu_allgather_st.cpp $(PCCL_LIB) $(KERNEL_LIB) | dirs
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS_TEST) $(LIBS_TEST_ALLGATHER) -o $@
	@echo "Built: $@"

$(TEST_DIR)/test_4gpu_allreduce_st: tests/test_4gpu_allreduce_st.cpp $(PCCL_LIB) $(KERNEL_LIB) | dirs
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS_TEST) $(LIBS_TEST_ALLGATHER) -o $@
	@echo "Built: $@"

$(TEST_DIR)/test_4gpu_allgather_stress_st: tests/test_4gpu_allgather_stress_st.cpp $(PCCL_LIB) $(KERNEL_LIB) | dirs
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS_TEST) $(LIBS_TEST_ALLGATHER) -o $@
	@echo "Built: $@"

$(TEST_DIR)/test_2gpu_allreduce_st: tests/test_2gpu_allreduce_st.cpp $(PCCL_LIB) $(KERNEL_LIB) | dirs
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS_TEST) $(LIBS_TEST_ALLGATHER) -o $@
	@echo "Built: $@"

$(TEST_DIR)/test_2gpu_allgather_st: tests/test_2gpu_allgather_st.cpp $(PCCL_LIB) $(KERNEL_LIB) | dirs
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS_TEST) $(LIBS_TEST_ALLGATHER) -o $@
	@echo "Built: $@"

$(TEST_DIR)/test_perf_st: tests/test_perf_st.cpp $(PCCL_LIB) $(KERNEL_LIB) | dirs
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS_TEST) $(LIBS_TEST_ALLGATHER) -o $@
	@echo "Built: $@"

# Clean all build artifacts
clean:
	rm -rf $(BUILD_DIR)
	@$(MAKE) -C kernels clean
	@echo "Cleaned build artifacts"

# Help target
help:
	@echo "PCIeCCL Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make all       - Build everything (default)"
	@echo "  make lib       - Build libpccl.so shared library"
	@echo "  make kernels   - Build kernel library"
	@echo "  make test      - Build test programs"
	@echo "  make clean     - Remove all build artifacts"
	@echo ""
	@echo "Environment Variables:"
	@echo "  LOG_LEVEL=N    - Set log level: 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR"
	@echo "  DEBUG=1        - Enable debug build with -g -O0 (default: 0)"
	@echo "  DEVICE=X       - Device type: cuda or ascend (auto-detected if not set)"
	@echo "  CUDA_HOME      - Path to CUDA toolkit (default: /usr/local/cuda)"
	@echo "  ASCEND_HOME    - Path to Ascend toolkit"
	@echo ""
	@echo "Build Output:"
	@echo "  build/lib/libpccl.so              - Main library"
	@echo "  build/lib/libpcieccl_kernels.so   - Kernel library"
	@echo "  build/tests/test_*                - Test programs"
	@echo ""
	@echo "Current Configuration:"
	@echo "  ARCH          = $(ARCH)"
	@echo "  ARCH_FLAGS    = $(ARCH_FLAGS)"
	@echo "  DEVICE        = $(DEVICE)"
ifeq ($(DEVICE),cuda)
	@echo "  CUDA_HOME     = $(CUDA_HOME)"
else
	@echo "  ASCEND_HOME   = $(ASCEND_HOME)"
endif
	@echo "  LOG_LEVEL     = $(LOG_LEVEL)"
