#
#loading path
#--------------------------------------------------------------------------------------------------------------------------------------------------
CXXFLAGS := -std=c++11\
        -Werror\
        -fstack-protector-strong\
        -fPIE -pie\
        -O2\
        -s\
        -Wl,-z,relro\
        -Wl,-z,now\
        -Wl,-z,noexecstack\
        -Wl,--copy-dt-needed-entries


ifneq (,$(filter HCCL_TEST_LOG_ENABLE, $(MAKECMDGOALS)))
	CXXFLAGS+= -DHCCL_TEST_LOG_ENABLE
endif

Common_DIR = ./common/src
Opbase_DIR = ./opbase_test

Common_SRC = $(wildcard ${Common_DIR}/*.cc)
Opbase_SRC = $(wildcard ${Opbase_DIR}/*.cc)

HCCL_INC_DIR = ${ASCEND_DIR}/include
HCCL_LIB_DIR = ${ASCEND_DIR}/lib64

ACL_INC_DIR = ${ASCEND_DIR}/include
ACL_LIB_DIR = ${ASCEND_DIR}/lib64

MPI_INC_DIR = ${MPI_HOME}/include
MPI_LIB_DIR = ${MPI_HOME}/lib

PROF_INC_DIR = ${ASCEND_DIR}/include
PROF_LIB_DIR = ${ASCEND_DIR}/lib64

LIST = all_gather_test all_gatherv_test all_reduce_test alltoallv_test alltoall_test broadcast_test reduce_scatter_test reduce_scatterv_test reduce_test scatter_test alltoallvc_test

#
#library flags
#--------------------------------------------------------------------------------------------------------------------------------------------------
LIBS = -L$(HCCL_LIB_DIR) -lhccl\
		-L$(ACL_LIB_DIR) -lacl_rt\
		-L$(MPI_LIB_DIR) -lmpi -lmpi_cxx\
        -L$(PROF_LIB_DIR) -lmsprofiler

INCLUDEDIRS = -I$(Common_DIR)\
				-I$(HCCL_INC_DIR)\
				-I$(ACL_INC_DIR)\
				-I$(MPI_INC_DIR)\
                -I$(PROF_INC_DIR)
#
#make
#--------------------------------------------------------------------------------------------------------------------------------------------------

HCCL_TEST_LOG_ENABLE: .PHONY all 

all:
	@mkdir -p bin
	g++ $(CXXFLAGS) $(Common_SRC) ${Opbase_DIR}/hccl_allgather_rootinfo_test.cc $(INCLUDEDIRS) -I${Opbase_DIR} -o all_gather_test $(LIBS)
	@printf "\033[0;32;32mall_gather_test compile completed\n\033[m"
	g++ $(CXXFLAGS) $(Common_SRC) ${Opbase_DIR}/hccl_allgatherv_rootinfo_test.cc $(INCLUDEDIRS) -I${Opbase_DIR} -o all_gatherv_test $(LIBS)
	@printf "\033[0;32;32mall_gatherv_test compile completed\n\033[m"
	g++ $(CXXFLAGS) $(Common_SRC) ${Opbase_DIR}/hccl_allreduce_rootinfo_test.cc $(INCLUDEDIRS) -I${Opbase_DIR} -o all_reduce_test $(LIBS)
	@printf "\033[0;32;32mall_reduce_test compile completed\n\033[m"
	g++ $(CXXFLAGS) $(Common_SRC) ${Opbase_DIR}/hccl_alltoallv_rootinfo_test.cc $(INCLUDEDIRS) -I${Opbase_DIR} -o alltoallv_test $(LIBS)
	@printf "\033[0;32;32mall_to_allv_test compile completed\n\033[m"
	g++ $(CXXFLAGS) $(Common_SRC) ${Opbase_DIR}/hccl_alltoall_rootinfo_test.cc $(INCLUDEDIRS) -I${Opbase_DIR} -o alltoall_test $(LIBS)
	@printf "\033[0;32;32mall_to_all_test compile completed\n\033[m"
	g++ $(CXXFLAGS) $(Common_SRC) ${Opbase_DIR}/hccl_brocast_rootinfo_test.cc $(INCLUDEDIRS) -I${Opbase_DIR} -o broadcast_test $(LIBS)
	@printf "\033[0;32;32mbroad_cast_test compile completed\n\033[m"
	g++ $(CXXFLAGS) $(Common_SRC) ${Opbase_DIR}/hccl_reducescatter_rootinfo_test.cc $(INCLUDEDIRS) -I${Opbase_DIR} -o reduce_scatter_test $(LIBS)
	@printf "\033[0;32;32mreduce_scatter_test compile completed\n\033[m"
	g++ $(CXXFLAGS) $(Common_SRC) ${Opbase_DIR}/hccl_reducescatterv_rootinfo_test.cc $(INCLUDEDIRS) -I${Opbase_DIR} -o reduce_scatterv_test $(LIBS)
	@printf "\033[0;32;32mreduce_scatterv_test compile completed\n\033[m"
	g++ $(CXXFLAGS) $(Common_SRC) ${Opbase_DIR}/hccl_reduce_rootinfo_test.cc $(INCLUDEDIRS) -I${Opbase_DIR} -o reduce_test $(LIBS)
	@printf "\033[0;32;32mreduce_test compile completed\n\033[m"
	g++ $(CXXFLAGS) $(Common_SRC) ${Opbase_DIR}/hccl_scatter_rootinfo_test.cc $(INCLUDEDIRS) -I${Opbase_DIR} -o scatter_test $(LIBS)
	@printf "\033[0;32;32mscatter_test compile completed\n\033[m"
	g++ $(CXXFLAGS) $(Common_SRC) ${Opbase_DIR}/hccl_alltoallvc_rootinfo_test.cc  $(INCLUDEDIRS) -I${Opbase_DIR} -o alltoallvc_test $(LIBS)
	@printf "\033[0;32;32malltoallvc_test compile completed\n\033[m"
	mv $(LIST) ./bin

.PHONY: clean
clean:
	rm -rf ./bin/*_test