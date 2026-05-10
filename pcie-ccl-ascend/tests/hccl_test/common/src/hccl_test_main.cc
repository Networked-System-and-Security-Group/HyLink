/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <unistd.h>
#include "hccl_test_common.h"
#include "mpi.h"

using namespace hccl;

int main(int argc, char *argv[])
{
    // Make sure everyline is flushed so that we see the progress of the test
    setlinebuf(stdout);
    auto begin = system_clock::now();
    MPI_Init(&argc, &argv);
    auto diff = duration_cast<microseconds>(system_clock::now() - begin).count();
    HCCL_TEST_LOG("mpi init success, take time[%lld us].\n", diff);
    int ret = 0;

    //构造执行器
    HcclTest *hccl_test = nullptr;
    hccl_test = init_opbase_ptr(hccl_test);
    if(hccl_test == nullptr) {
        printf("hccl_test is null\n");
        ret = -1;
        goto hccltesterr3;
    }

    //解析命令行入参
    ret = hccl_test->parse_cmd_line(argc, argv);
    if (ret == 1) {
        //启动--help
        ret = 0;
        goto hccltesterr2;
    } else if(ret == -1) {
        //入参解析失败
        printf("This is an error in parse_cmd_line.\n");
        goto hccltesterr2;
    }

    begin = system_clock::now();
    //查找本host上的所有MPI拉起的进程
    ret = hccl_test->get_mpi_proc();
    if (ret != 0) {
        printf("This is an error in get_mpi_proc.\n");
        goto hccltesterr2;
    }
    diff = duration_cast<microseconds>(system_clock::now() - begin).count();
    HCCL_TEST_LOG("get_mpi_proc success, take time[%lld us].\n", diff);

    //校验命令行参数
    ret = hccl_test->check_cmd_line();
    if (ret != 0) {
        printf("This is an error in check_cmd_line.\n");
        goto hccltesterr2;
    }

    begin = system_clock::now();
    ret = hccl_test->device_init();
    if (ret != 0) {
        printf("This is an error in device_init.\n");
        goto hccltesterr2;
    }

    diff = duration_cast<microseconds>(system_clock::now() - begin).count();
    HCCL_TEST_LOG("device_init success, take time[%lld us].\n", diff);

    //获取hccltest的环境变量
    ret = hccl_test->get_env_resource();
    if (ret != 0) {
        printf("This is an error in get_env.\n");
        goto hccltesterr1;
    }
    ret = hccl_test->set_env_resource();
    if (ret != 0) {
        printf("This is an error in set_env.\n");
        goto hccltesterr1;
    }

    begin = system_clock::now();
    ret = hccl_test->start_test();
    diff = duration_cast<microseconds>(system_clock::now() - begin).count();
    HCCL_TEST_LOG("start_test success, take time[%lld us].\n", diff);

hccltesterr1:
    //销毁环境变量申请的资源
    ret = hccl_test->release_env_resource();
    if (ret != 0) {
        printf("This is an error in release_env_resource.\n");
    }
hccltesterr2:
    //删除构造器
    delete_opbase_ptr(hccl_test);
hccltesterr3:
    //设备去初始化
    ACLCHECK(aclFinalize());
    //释放MPI所用资源
    MPI_Finalize();
    return ret;
}
