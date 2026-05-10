# 	HCCL Test

HCCL Test提供HCCL通信性能与正确性测试

## 环境准备

* 安装CANN Toolkit包

  运行HCCL Test工具依赖CANN Toolkit开发套件包和CANN算子包，请根据操作系统架构，下载对应版本的CANN软件包，参考[昇腾文档中心-CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)进行安装。

* 设置CANN软件环境变量

  ```shell
  # 默认路径，root用户安装
  source /usr/local/Ascend/cann/set_env.sh
  # 默认路径，非root用户安装
  source $HOME/Ascend/cann/set_env.sh
  ```

* 安装并配置MPI环境变量

  MPI的安装请参见[HCCL性能测试工具用户指南](https://www.hiascend.com/document/redirect/CannCommunityToolHcclTest)的"MPI安装与配置"章节。

  环境变量配置示例如下：

  ```shell
  export PATH=/path/to/mpi/bin:$PATH
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/mpi/lib
  ```

* 多机集群训练时，需配置环境变量指定host网卡：（HCCL_SOCKET_IFNAME）

  ```shell
  # 配置HCCL的初始化root通信网卡名，HCCL可通过该网卡名获取Host IP，完成通信域创建。
  # 支持以下格式配置：(4种规格自行选择1种即可)
  # 精确匹配网卡
  export HCCL_SOCKET_IFNAME==eth0,enp0     # 使用指定的eth0或enp0网卡
  export HCCL_SOCKET_IFNAME=^=eth0,enp0    # 不使用eth0与enp0网卡
  # 模糊匹配网卡
  export HCCL_SOCKET_IFNAME=eth,enp        # 使用所有以eth或enp为前缀的网卡
  export HCCL_SOCKET_IFNAME=^eth,enp       # 不使用任何以eth或enp为前缀的网卡
  ```

  注：网卡名仅为举例说明，并不只对eth,enp网卡生效
  
* 多机集群训练时，需统计所有节点使用的host网卡信息

  编辑hostfile文件：

  ```shell
  vim hostfile
  ```

  将全部参与训练的节点信息写入hostfile文件，格式如下：

  ```shell
  # 全部参与训练的节点ip:每节点的进程数
  192.168.1.1:8
  192.168.1.2:8
  ...
  ```

## 构建

* 构建HCCL Test

  其中MPI_HOME为MPI安装路径，ASCEND_DIR为CANN Toolkit开发套件包的安装路径。

  ```shell
  make MPI_HOME=/path/to/mpi ASCEND_DIR=${ASCEND_HOME_PATH}
  ```

## 执行

* 单节点运行：

  ```shell
  mpirun -n 8 ./bin/all_reduce_test -b 8K -e 64M -f 2 -d fp32 -o sum -p 8
  ```

* 多节点运行：（两节点为例）

  ```shell
  mpirun -f hostfile -n 16 ./bin/all_reduce_test -b 8K -e 64M -f 2 -d fp32 -o sum -p 8
  ```

## 参数

所有测试都支持相同的参数集：

* NPU数量
  
  * `[-p,--npus <npus used for one node>] ` 每个计算节点上参与训练的npu个数，默认值：当前节点的npu总数
  
* 数据量
  * `-b,--minbytes <min size in bytes>` 数据量起始值，默认值：64M
  * `-e,--maxbytes <max size in bytes>` 数据量结束值，默认值：64M
  * 数据增量通过增量步长或乘法因子参数设置
    * `-i,--stepbytes <increment size>` 增量步长，默认值：(max-min)/10
    
      注：当输入增量步长（-i）为0时，会持续对数据量起始值（-b）进行测试。
    
    * `-f,--stepfactor <increment factor>` 乘法因子，默认值：不开启
  
* HCCL操作参数
  * `-o,--op <sum/prod/max/min>` 集合通信操作归约类型，默认值：sum
  
  * `-r,--root <root>` root节点，broadcast,reduce和scatter操作生效，默认值：0
  
  * `-d,--datatype <int8/int16/int32/fp16/fp32/int64/uint64/uint8/uint16/uint32/fp64>` 数据类型，默认值：fp32（即float32）

  * `-z,--zero_copy <0/1>` 开启0拷贝，allgather, reduce_scatter, broadcast, allreduce操作符合约束条件生效，默认值：0
  * `-m,--symmetric_memory <0/1>` 开启对称内存，allgather, reduce_scatter, allreduce操作符合约束条件生效，默认值：0
      注：不支持零拷贝和对称内存同时开启。
 
* 性能
  * `-n,--iters <iteration count>` 迭代次数，默认值：20
  * `-w,--warmup_iters <warmup iteration count>` 预热迭代次数（不参与性能统计，仅影响HCCL Test执行耗时），默认值：10
  * `-t,--onlydevicetime <0/1>` 将通信算子host侧软件耗时与kernel加载耗时排除在通信执行耗时之外，只统计device执行时间（影响HCCL TEST执行耗时），默认值：0
      注：
      1、当启动 -t 参数时，-w\-n参数配置不大于100轮次
      2、当启动 -t 参数时，不支持aicpu_ts模式
      3、当启动 -t 参数时，并且HCCL_BUFFSIZE配置小于等于100MB时，-t 参数不生效
* 结果校验
  
  * `-c,--check <0/1>` 校验集合通信操作结果正确性（大规模集群场景下，开启结果校验会使HCCL Test执行耗时增加），默认值：1（开启）

## 执行示例

* allreduce

  ```shell
  # 单节点8个NPU
  mpirun -n 8 ./bin/all_reduce_test -b 8K -e 64M -f 2 -p 8
  ```

  ```shell
  # 双节点16个NPU
  mpirun -f hostfile -n 16 ./bin/all_reduce_test -b 8K -e 64M -f 2 -p 8
  ```

* broadcast

  ```shell
  # 单节点8个NPU
  mpirun -n 8 ./bin/broadcast_test -b 8K -e 64M -f 2 -p 8 -r 1
  ```

  ```shell
  # 双节点16个NPU
  mpirun -f hostfile -n 16 ./bin/broadcast_test -b 8K -e 64M -f 2 -p 8 -r 1
  ```

* allgather

  ```shell
  # 单节点8个NPU
  mpirun -n 8 ./bin/all_gather_test -b 8K -e 64M -f 2 -p 8
  ```

  ```shell
  # 双节点16个NPU
  mpirun -f hostfile -n 16 ./bin/all_gather_test -b 8K -e 64M -f 2 -p 8
  ```

* alltoallv

  ```shell
  # 单节点8个NPU
  mpirun -n 8 ./bin/alltoallv_test -b 8K -e 64M -f 2 -p 8
  ```

  ```shell
  # 双节点16个NPU
  mpirun -f hostfile -n 16 ./bin/alltoallv_test -b 8K -e 64M -f 2 -p 8
  ```

* alltoall

  ```shell
  # 单节点8个NPU
  mpirun -n 8 ./bin/alltoall_test -b 8K -e 64M -f 2 -p 8
  ```

  ```shell
  # 双节点16个NPU
  mpirun -f hostfile -n 16 ./bin/alltoall_test -b 8K -e 64M -f 2 -p 8
  ```

* reducescatter

  ```shell
  # 单节点8个NPU
  mpirun -n 8 ./bin/reduce_scatter_test -b 8K -e 64M -f 2 -p 8
  ```

  ```shell
  # 双节点16个NPU
  mpirun -f hostfile -n 16 ./bin/reduce_scatter_test -b 8K -e 64M -f 2 -p 8
  ```

* reduce

  ```shell
  # 单节点8个NPU
  mpirun -n 8 ./bin/reduce_test -b 8K -e 64M -f 2 -p 8 -r 1
  ```
  
  ``` shell
  # 双节点16个NPU
  mpirun -f hostfile -n 16 ./bin/reduce_test -b 8K -e 64M -f 2 -p 8 -r 1
  ```

## 指定deviceId执行用例

  执行HCCL Test工具前，开启如下环境变量，即可指定需要启动的device。

  需要在当前计算节点的hccl_test目录下，创建一个可执行文件，例：run.sh。

  * 单server场景（启动4，5，6，7卡）

    * 创建可执行文件：

      run.sh文件内容如下：
      ```shell
      #HCCL_TEST_USE_DEVS后的数字为需要启动的deviceId
      export HCCL_TEST_USE_DEVS="4,5,6,7"
      $1
      ```
    * 用例执行：

      ```shell
      mpirun -n 4 ./run.sh "./all_reduce_test -b 8K -e 64M -f 2 -p 4"
      ```

* 多server场景（计算节点1启动0，1，2，3卡，计算节点2启动4，5，6，7卡）

  * 计算节点1：

    * 创建可执行文件：
    
      run.sh文件内容如下：
	    ```shell
      export HCCL_TEST_USE_DEVS="0,1,2,3"
      $1
	    ```
    
  * 计算节点2：

    * 创建可执行文件：
    
      run.sh文件内容如下：
	    ```shell
      export HCCL_TEST_USE_DEVS="4,5,6,7"
      $1
	    ```

  * 用例执行：

    ```shell
    mpirun -n 8 -f hostfile ./run.sh "./all_reduce_test -b 8K -e 64M -f 2 -p 4"
	  ```

## 开启性能数据采集

  执行HCCL Test工具前，设置如下环境变量，即可开启性能数据采集功能。

  ```shell
  # “1”代表开启profiling，“0”代表关闭profiling，默认值为“0”，开启时，执行HCCL Test时采集性能数据
  export HCCL_TEST_PROFILING=1
  # 指定profiling数据存放路径，默认为“/var/log/npu/profiling”
  export HCCL_TEST_PROFILING_PATH=/home/profiling
  ```
  HCCL Test工具执行完成后会在HCCL_TEST_PROFILING_PATH指定目录下生成profiling数据。