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
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <hccl/hccl_types.h>
#include "hccl_alltoallv_rootinfo_test.h"
#include "hccl_opbase_rootinfo_base.h"
#include "hccl_check_buf_init.h"
using namespace hccl;

HcclTest* hccl::init_opbase_ptr(HcclTest* opbase)
{
    opbase = new HcclOpBaseAlltoallvTest();
    opbase->need_ranksize_alignment = true;

    return opbase;
}

void hccl::delete_opbase_ptr(HcclTest *&opbase)
{
    delete opbase;
    opbase = nullptr;
    return;
}

namespace hccl {
HcclOpBaseAlltoallvTest::HcclOpBaseAlltoallvTest() : HcclOpBaseTest()
{
    host_buf = nullptr;
    recv_buff_temp = nullptr;
    send_buff = nullptr;
    send_counts = nullptr;
    send_disp = nullptr;
    recv_buff = nullptr;
    recv_counts = nullptr;
    recv_disp = nullptr;
}

HcclOpBaseAlltoallvTest::~HcclOpBaseAlltoallvTest()
{
}

void HcclOpBaseAlltoallvTest::malloc_send_recv_buf()
{
    send_counts = (unsigned long long *)malloc(rank_size * sizeof(unsigned long long));
    if (send_counts == nullptr) {
        return;
    } 
    send_disp = (unsigned long long *)malloc(rank_size * sizeof(unsigned long long));
    if (send_disp == nullptr) {
        free(send_counts);
        return;
    }
    for(int i = 0; i < rank_size; ++i)
    {
        send_counts[i] = data->count / rank_size;
        send_disp[i] = i * data->count / rank_size;
    }

    recv_counts = (unsigned long long *)malloc(rank_size * sizeof(unsigned long long));
    if (recv_counts == nullptr) {
        return;
    } 
    recv_disp = (unsigned long long *)malloc(rank_size * sizeof(unsigned long long));
    if (recv_disp == nullptr) {
        free(recv_counts);
        return;
    }
    for(int i = 0; i < rank_size; ++i)
    {
        recv_counts[i] = data->count / rank_size;
        recv_disp[i] = i * data->count / rank_size;
    }
    return;
}

int HcclOpBaseAlltoallvTest::check_buf_result()
{
    //获取输出内存
    ACLCHECK(aclrtMallocHost((void**)&check_buf, malloc_kSize));
    ACLCHECK(aclrtMemcpy((void*)check_buf, malloc_kSize, (void*)recv_buff, malloc_kSize, ACL_MEMCPY_DEVICE_TO_HOST));

    int ret = 0;
    ret = hccl_alltoallv_check_result(check_buf, recv_counts, recv_disp, rank_id, rank_size, dtype);
    if(ret != 0)
    {
        check_err++;
    }
    return 0;
}

void HcclOpBaseAlltoallvTest::cal_execution_time(float time)
{
    double total_time_us = time * 1000;
    double average_time_us = total_time_us / iters;
    double algorithm_bandwith_GBytes_s = malloc_kSize / average_time_us * B_US_TO_GB_S;

    print_execution_time(average_time_us, algorithm_bandwith_GBytes_s);
    return;
}

void HcclOpBaseAlltoallvTest::free_send_recv_buf()
{
    free(send_counts);
    free(send_disp);
    free(recv_counts);
    free(recv_disp);
}

size_t HcclOpBaseAlltoallvTest::init_malloc_Ksize_by_data()
{
    u64 alignedData =
        data->count >= rank_size * granularity ? rank_size * granularity : rank_size;
    data->count = (data->count + alignedData - 1) / alignedData * alignedData;
    return data->count * data->type_size;
}

void HcclOpBaseAlltoallvTest::init_send_recv_size_by_data(size_t &send_bytes, size_t &recv_bytes)
{
    send_bytes = malloc_kSize;
    recv_bytes = malloc_kSize;
}
int HcclOpBaseAlltoallvTest::hccl_op_base_test() //主函数
{
    if (op_flag != 0 && rank_id == root_rank) {
        printf("Warning: The -o,--op <sum/prod/min/max> option does not take effect. Check the cmd parameter.\n");
    }

    is_initdata_overflow();
    //申请sendcounts和send_disp
    malloc_send_recv_buf();

    //初始化输入内存
    ACLCHECK(aclrtMallocHost((void**)&host_buf, malloc_kSize));
    hccl_host_buf_init((char*)host_buf, data->count, dtype, rank_id + 1);
    ACLCHECK(aclrtMemcpy((void*)send_buff, malloc_kSize, (void*)host_buf, malloc_kSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // 输入数据量，根据条件判断是否开启仅计算device执行时间
 	ACLCHECK(start_profile_device_time_if_needed(16*1024*1024));
    //执行集合通信操作
    for(int j = 0; j < warmup_iters; ++j) {
        HCCLCHECK(HcclAlltoAllV((void *)send_buff, send_counts, send_disp, (HcclDataType)dtype,\
            (void*)recv_buff, recv_counts, recv_disp, (HcclDataType)dtype, hccl_comm, stream));
    }

    ACLCHECK(aclrtRecordEvent(start_event, stream));

    for(int i = 0; i < iters; ++i) {
        HCCLCHECK(HcclAlltoAllV((void *)send_buff, send_counts, send_disp, (HcclDataType)dtype,\
            (void*)recv_buff, recv_counts, recv_disp, (HcclDataType)dtype, hccl_comm, stream));
    }
    //等待stream中集合通信任务执行完成
    ACLCHECK(aclrtRecordEvent(end_event, stream));
    ACLCHECK(end_profile_device_time_if_needed(16*1024*1024));
    ACLCHECK(aclrtSynchronizeStream(stream));

    float time;
    ACLCHECK(aclrtEventElapsedTime(&time, start_event, end_event));

    if (check == 1) {
        ACLCHECK(check_buf_result()); // 校验计算结果
    }

    cal_execution_time(time);
    free_send_recv_buf();
    return 0;
}
}