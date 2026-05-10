# Makefile for PCIECCL kernels
#
# Build Instructions:
#   make                     - Build Ascend kernel library (default)
#   make lib                 - Build Ascend kernel library
#   make cuda_lib            - Build CUDA kernel library
#   make test                - Build Ascend test executable
#   make clean               - Clean build artifacts
#
# Environment Variables:
#   ASCEND_HOME              - Path to Ascend toolkit (default: /usr/local/Ascend/ascend-toolkit/latest)
#   ASCEND_PRODUCT_TYPE      - Product type (default: Ascend310P1; supports: Ascend910A, Ascend310P1, Ascend310B1, Ascend910B1, Ascend910B3)
#   ASCEND_CORE_TYPE         - Core type (default: VectorCore; options: AiCore, VectorCore)
#   CUDA_HOME                - Path to CUDA toolkit (default: /usr/local/cuda)
#   CUDA_ARCH                - CUDA architecture (default: native for auto-detection)

# Output directories - BUILD_DIR can be overridden from parent Makefile
BUILD_DIR ?= .
LIB_DIR = $(BUILD_DIR)/lib
KERNEL_LIB = $(LIB_DIR)/libpcieccl_kernels.so

#
# ============================================================================
# ASCEND CONFIGURATION
# ============================================================================
#

# Ascend toolkit paths
ASCEND_HOME ?= /usr/local/Ascend/ascend-toolkit/latest
ASCEND_PRODUCT_TYPE ?= Ascend310P1
ASCEND_CORE_TYPE ?= VectorCore

# CCE Compiler - use toolkit path for NPU compilation
# Check multiple possible paths for ccec compiler
CCEC_PATH1 = $(ASCEND_HOME)/toolkit/tools/ccec_compiler/bin/ccec
CCEC_PATH2 = $(ASCEND_HOME)/compiler/ccec_compiler/bin/ccec
CCEC = $(shell if [ -f $(CCEC_PATH1) ]; then echo $(CCEC_PATH1); elif [ -f $(CCEC_PATH2) ]; then echo $(CCEC_PATH2); else echo $(CCEC_PATH1); fi)

# Architecture flags based on product type
ifeq ($(ASCEND_PRODUCT_TYPE),Ascend910A)
    ARCH_FLAG = --cce-aicore-arch=dav-c100
else ifeq ($(ASCEND_PRODUCT_TYPE),Ascend310P1)
    ifeq ($(ASCEND_CORE_TYPE),AiCore)
        ARCH_FLAG = --cce-aicore-arch=dav-m200
    else
        ARCH_FLAG = --cce-aicore-arch=dav-m200-vec
    endif
else ifeq ($(ASCEND_PRODUCT_TYPE),Ascend310B1)
    ifeq ($(ASCEND_CORE_TYPE),AiCore)
        ARCH_FLAG = --cce-aicore-arch=dav-m300
    else
        ARCH_FLAG = --cce-aicore-arch=dav-m300-vec
    endif
else ifeq ($(ASCEND_PRODUCT_TYPE),Ascend910B1)
    ifeq ($(ASCEND_CORE_TYPE),AiCore)
        ARCH_FLAG = --cce-aicore-arch=dav-c220-cube
    else
        ARCH_FLAG = --cce-aicore-arch=dav-c220-vec
    endif
else ifeq ($(ASCEND_PRODUCT_TYPE),Ascend910B3)
    # Ascend910B3 uses same architecture as Ascend910B1
    ifeq ($(ASCEND_CORE_TYPE),AiCore)
        ARCH_FLAG = --cce-aicore-arch=dav-c220-cube
    else
        ARCH_FLAG = --cce-aicore-arch=dav-c220-vec
    endif
endif

# Include paths
INCLUDES = -I$(ASCEND_HOME)/include \
           -I$(ASCEND_HOME)/compiler/tikcpp/tikcfw \
           -I$(ASCEND_HOME)/compiler/tikcpp/tikcfw/impl \
           -I$(ASCEND_HOME)/compiler/tikcpp/tikcfw/interface \
           -I$(ASCEND_HOME)/tools/tikcpp/tikcfw \
           -I$(ASCEND_HOME)/tools/tikcpp/tikcfw/impl \
           -I$(ASCEND_HOME)/tools/tikcpp/tikcfw/interface \
           -I$(ASCEND_HOME)/acllib/include

# Compiler flags
# -xcce: Enable CCE mode for Ascend C syntax recognition
# -fPIC: Position Independent Code for shared library
CCEFLAGS = -xcce -O2 -std=c++17 $(ARCH_FLAG) --cce-auto-sync $(INCLUDES) -fPIC

# Link flags - use fatobj-link for embedding kernel code
LDFLAGS = --cce-fatobj-link \
          -L$(ASCEND_HOME)/lib64 \
          -L$(ASCEND_HOME)/runtime/lib64 \
          -L$(ASCEND_HOME)/tools/simulator/$(ASCEND_PRODUCT_TYPE)/lib \
          -L$(ASCEND_HOME)/tools/tikicpulib/lib/$(ASCEND_PRODUCT_TYPE)
LIBS = -lstdc++ -lruntime -lascendcl

# Source files
KERNEL_SRCS = wait_kernel_ascendc.cpp
KERNEL_OBJS = $(KERNEL_SRCS:.cpp=.o)

#
# ============================================================================
# CUDA CONFIGURATION
# ============================================================================
#

# CUDA toolkit paths
CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc

# Auto-detect GPU architecture (requires CUDA 11.0+)
# Use 'native' for auto-detection, or specify explicitly (e.g., sm_70, sm_80)
# For cross-compilation or older CUDA, explicitly set CUDA_ARCH=sm_XX
CUDA_ARCH ?= native

# NVCC compiler flags
# -O2: Optimization level
# -std=c++11: C++ standard
# --compiler-options '-fPIC': Position Independent Code for shared library
# -arch=$(CUDA_ARCH): Target GPU architecture
NVCCFLAGS = -O2 -std=c++11 --compiler-options '-fPIC' -arch=$(CUDA_ARCH)

# CUDA link flags
CUDA_LDFLAGS = -shared -L$(CUDA_HOME)/lib64
CUDA_LIBS = -lcudart

# CUDA source files
CUDA_KERNEL_SRCS = wait_kernel.cu
CUDA_KERNEL_OBJS = $(CUDA_KERNEL_SRCS:.cu=.cuda.o)

#
# ============================================================================
# BUILD TARGETS
# ============================================================================
#

.PHONY: all lib cuda_lib test clean help

# Default target (Ascend)
all: lib

# Ascend kernel library
lib: $(KERNEL_LIB)

$(KERNEL_LIB): $(KERNEL_OBJS)
	@mkdir -p $(LIB_DIR)
	$(CCEC) -shared $(LDFLAGS) $^ $(LIBS) -o $@

%.o: %.cpp
	$(CCEC) $(CCEFLAGS) -c $< -o $@

# CUDA kernel library
# Note: cuda_lib builds to same output path as lib target
# The parent Makefile calls either 'lib' or 'cuda_lib' exclusively based on DEVICE
.PHONY: cuda_lib_impl
cuda_lib: cuda_lib_impl

cuda_lib_impl: $(CUDA_KERNEL_OBJS)
	@mkdir -p $(LIB_DIR)
	$(NVCC) $(CUDA_LDFLAGS) $^ $(CUDA_LIBS) -o $(LIB_DIR)/libpcieccl_kernels.so

%.cuda.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f *.o
	@if [ "$(BUILD_DIR)" = "." ]; then \
		rm -rf lib; \
	else \
		rm -rf $(LIB_DIR); \
	fi

help:
	@echo "PCIECCL Kernel Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make              - Build Ascend kernel library (default)"
	@echo "  make lib          - Build Ascend kernel library"
	@echo "  make cuda_lib     - Build CUDA kernel library"
	@echo "  make clean        - Remove build artifacts"
	@echo ""
	@echo "Ascend Environment Variables:"
	@echo "  ASCEND_HOME           - Ascend toolkit path (default: /usr/local/Ascend/ascend-toolkit/latest)"
	@echo "  ASCEND_PRODUCT_TYPE   - Product type: Ascend910A, Ascend310P1, Ascend310B1, Ascend910B1, Ascend910B3"
	@echo "  ASCEND_CORE_TYPE      - Core type: AiCore, VectorCore"
	@echo ""
	@echo "CUDA Environment Variables:"
	@echo "  CUDA_HOME             - CUDA toolkit path (default: /usr/local/cuda)"
	@echo "  CUDA_ARCH             - GPU architecture: native (default), sm_70, sm_80, etc."
	@echo ""
	@echo "Current Ascend Configuration:"
	@echo "  ASCEND_HOME         = $(ASCEND_HOME)"
	@echo "  ASCEND_PRODUCT_TYPE = $(ASCEND_PRODUCT_TYPE)"
	@echo "  ASCEND_CORE_TYPE    = $(ASCEND_CORE_TYPE)"
	@echo "  ARCH_FLAG           = $(ARCH_FLAG)"
	@echo ""
	@echo "Current CUDA Configuration:"
	@echo "  CUDA_HOME           = $(CUDA_HOME)"
	@echo "  CUDA_ARCH           = $(CUDA_ARCH)"
