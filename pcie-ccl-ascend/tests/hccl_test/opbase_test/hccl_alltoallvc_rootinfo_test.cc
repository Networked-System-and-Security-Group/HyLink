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

#include "hccl_alltoallvc_rootinfo_test.h"
#include "hccl_check_buf_init.h"
#include <chrono>
#include <thread>
using namespace hccl;

HcclTest* hccl::init_opbase_ptr(HcclTest* opbase)
{
    opbase = new HcclOpBaseAlltoallvcTest();
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
HcclOpBaseAlltoallvcTest::HcclOpBaseAlltoallvcTest()
{
    send_count_matrix = nullptr;
    recv_counts = nullptr;
}

HcclOpBaseAlltoallvcTest::~HcclOpBaseAlltoallvcTest()
{
}

void HcclOpBaseAlltoallvcTest::malloc_send_recv_buf()
{
    send_count_matrix = (unsigned long long*)malloc(rank_size * rank_size * sizeof(unsigned long long));
    if (send_count_matrix == nullptr) {
        return;
    }
    recv_counts = (unsigned long long*)malloc(rank_size * sizeof(unsigned long long));
    if (recv_counts == nullptr) {
        free(send_count_matrix);
        return;
    }

    for (int i = 0; i < rank_size; ++i) {
        for (int j = 0; j < rank_size; ++j) {
            send_count_matrix[i * rank_size + j] = data->count / rank_size;
        }
    }

    for (int i = 0; i < rank_size; ++i) {
        recv_counts[i] = send_count_matrix[i * rank_size + rank_id];
    }
}

void HcclOpBaseAlltoallvcTest::free_send_recv_buf()
{
    free(send_count_matrix);
    free(recv_counts);
}

int HcclOpBaseAlltoallvcTest::check_buf_result()
{
    ACLCHECK(aclrtMallocHost((void**)&check_buf, malloc_kSize));
    ACLCHECK(aclrtMemcpy((void*)check_buf, malloc_kSize, (void*)recv_buff, malloc_kSize, ACL_MEMCPY_DEVICE_TO_HOST));
    // 构造 recv_disp
    u64 *recv_disp = (u64 *)malloc(sizeof(u64) * rank_size);
    if (recv_disp == nullptr) {
        return HCCL_E_PTR;
    } 
    recv_disp[0] = 0;
    for (int i = 1; i < rank_size; ++i) {
        recv_disp[i] = recv_disp[i - 1] + recv_counts[i - 1];
    }
    int ret = hccl_alltoallv_check_result(check_buf, recv_counts, recv_disp, rank_id, rank_size, dtype);
    if (ret != 0) {
        check_err++;
    }
    free(recv_disp);
    return 0;
}


size_t HcclOpBaseAlltoallvcTest::init_malloc_Ksize_by_data()
{
    u64 alignedData =
        data->count >= rank_size * granularity ? rank_size * granularity : rank_size;
    data->count = (data->count + alignedData - 1) / alignedData * alignedData;
    return data->count * data->type_size;
}

void HcclOpBaseAlltoallvcTest::init_send_recv_size_by_data(size_t &send_bytes, size_t &recv_bytes)
{
    send_bytes = malloc_kSize;
    recv_bytes = malloc_kSize;
}

void HcclOpBaseAlltoallvcTest::cal_execution_time(float time)
{
    double total_time_us = time * 1000;
    double average_time_us = total_time_us / iters;
    double algorithm_bandwith_GBytes_s = malloc_kSize / average_time_us * B_US_TO_GB_S;

    print_execution_time(average_time_us, algorithm_bandwith_GBytes_s);
    return;
}

int HcclOpBaseAlltoallvcTest::hccl_op_base_test()
{
    if (op_flag != 0 && rank_id == root_rank) {
        printf("Warning: The -o,--op <sum/prod/min/max> option does not take effect. Check the cmd parameter.\n");
    }

    is_initdata_overflow();
    malloc_send_recv_buf();

    ACLCHECK(aclrtMallocHost((void**)&host_buf, malloc_kSize));
    hccl_host_buf_init(host_buf, data->count, dtype, rank_id + 1);
    ACLCHECK(aclrtMemcpy((void*)send_buff, malloc_kSize, (void*)host_buf, malloc_kSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // 输入数据量，根据条件判断是否开启仅计算device执行时间
 	ACLCHECK(start_profile_device_time_if_needed(128*1024*1024));

    for (int j = 0; j < warmup_iters; ++j) {
        HCCLCHECK(HcclAlltoAllVC((void*)send_buff, send_count_matrix, (HcclDataType)dtype,
                                 (void*)recv_buff, (HcclDataType)dtype, hccl_comm, stream));
    }

    ACLCHECK(aclrtRecordEvent(start_event, stream));
    for (int i = 0; i < iters; ++i) {
        HCCLCHECK(HcclAlltoAllVC((void*)send_buff, send_count_matrix, (HcclDataType)dtype,
                                 (void*)recv_buff, (HcclDataType)dtype, hccl_comm, stream));
    }
    ACLCHECK(aclrtRecordEvent(end_event, stream));
    ACLCHECK(end_profile_device_time_if_needed(128*1024*1024));
    ACLCHECK(aclrtSynchronizeStream(stream));

    float time;
    ACLCHECK(aclrtEventElapsedTime(&time, start_event, end_event));

    if (check == 1) {
        ACLCHECK(check_buf_result());
    }

    cal_execution_time(time);
    free_send_recv_buf();
    return 0;
}
}