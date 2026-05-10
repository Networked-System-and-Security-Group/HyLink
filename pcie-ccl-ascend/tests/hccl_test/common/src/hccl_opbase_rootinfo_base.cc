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
#include <string>
#include <cmath>
#include <cstdint>
#include <hccl/hccl_types.h>
#include "hccl_opbase_rootinfo_base.h"

namespace hccl
{
HcclOpBaseTest::HcclOpBaseTest()
{
    host_buf = nullptr;
    recv_buff_temp = nullptr;
    check_buf = nullptr;
}

HcclOpBaseTest::~HcclOpBaseTest()
{
    destory_alloc_buf();
}

int HcclOpBaseTest::hccl_op_base_test()
{
    return 0;
}

void HcclOpBaseTest::init_data_count()
{
    switch(dtype)
    {
        case HCCL_DATA_TYPE_FP32:
            data->count = (data->data_size + sizeof(float) - 1)/sizeof(float); //count向上取整
            data->type_size = sizeof(float);
            break;
        case HCCL_DATA_TYPE_INT32:
            data->count = (data->data_size + sizeof(int) - 1)/sizeof(int);
            data->type_size = sizeof(int);
            break;
        case HCCL_DATA_TYPE_BFP16:
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_INT16:
            data->count = (data->data_size + sizeof(short) - 1)/sizeof(short);
            data->type_size = sizeof(short);
            break;
        case HCCL_DATA_TYPE_INT8:
            data->count = (data->data_size + sizeof(signed char) - 1)/sizeof(signed char);
            data->type_size = sizeof(signed char);
            break;
        case HCCL_DATA_TYPE_INT64:
        case HCCL_DATA_TYPE_FP64:
            data->count = (data->data_size + sizeof(long long) - 1)/sizeof(long long);
            data->type_size = sizeof(long long);
            break;
        case HCCL_DATA_TYPE_UINT64:
            data->count = (data->data_size + sizeof(unsigned long long) - 1)/sizeof(unsigned long long);
            data->type_size = sizeof(unsigned long long);
            break;
        case HCCL_DATA_TYPE_UINT8:
        case HCCL_DATA_TYPE_HIF8:
        case HCCL_DATA_TYPE_FP8E4M3:
        case HCCL_DATA_TYPE_FP8E5M2:
        case HCCL_DATA_TYPE_FP8E8M0:
        case HCCL_DATA_TYPE_MXFP8:
            data->count = (data->data_size + sizeof(unsigned char) - 1)/sizeof(unsigned char);
            data->type_size = sizeof(unsigned char);
            break;
        case HCCL_DATA_TYPE_UINT16:
            data->count = (data->data_size + sizeof(unsigned short) - 1)/sizeof(unsigned short);
            data->type_size = sizeof(unsigned short);
            break;
        case HCCL_DATA_TYPE_UINT32:
            data->count = (data->data_size + sizeof(unsigned int) - 1)/sizeof(unsigned int);
            data->type_size = sizeof(unsigned int);
            break;
        default:
            data->count = (data->data_size + sizeof(float) - 1)/sizeof(float);
            data->type_size = sizeof(float);
            break;
    }
    return;
}

int HcclOpBaseTest::init_buf_val()
{
    return 0;
}

int HcclOpBaseTest::check_buf_result()
{
    return 0;
}

void HcclOpBaseTest::no_verification()
{
    check = 0; //不进行校验
    if (rank_id == root_rank && print_dump) {
        printf("Warning: The calculation result overflows, No verification is performed.\n");
        print_dump = false;
    }
    return;
}

void HcclOpBaseTest::is_initdata_overflow()
{
    if((dtype == HCCL_DATA_TYPE_INT8 || dtype == HCCL_DATA_TYPE_UINT8 || dtype == HCCL_DATA_TYPE_HIF8
        || dtype == HCCL_DATA_TYPE_FP8E4M3 || dtype == HCCL_DATA_TYPE_FP8E5M2 || dtype == HCCL_DATA_TYPE_FP8E8M0 || dtype == HCCL_DATA_TYPE_MXFP8)
        && rank_size >= RANKSIZE_TH_FP32) {
        check = 0; //不进行校验
        if (rank_id == root_rank && print_dump) {
            printf("Warning: The initdata overflows, No verification is performed.\n");
            print_dump = false;
        }
    }
    return;
}

void HcclOpBaseTest::is_data_overflow()
{
    if (op_type == HCCL_REDUCE_PROD) {
        if (dtype == HCCL_DATA_TYPE_FP16 && rank_size >= RANKSIZE_TH_FP16) {
            no_verification();
        }
        if (dtype == HCCL_DATA_TYPE_FP32 && rank_size >= RANKSIZE_TH_FP32) {
            no_verification();
        }
        if ((dtype == HCCL_DATA_TYPE_INT8 || dtype == HCCL_DATA_TYPE_UINT8) &&
            rank_size >= RANKSIZE_TH_INT8) {
            no_verification();
        }
        if ((dtype == HCCL_DATA_TYPE_INT32 || dtype == HCCL_DATA_TYPE_UINT32)  &&
            rank_size >= RANKSIZE_TH_INT32) {
            no_verification();
        }
        if ((dtype == HCCL_DATA_TYPE_INT64 || dtype == HCCL_DATA_TYPE_UINT64) &&
            rank_size >= RANKSIZE_TH_INT64) {
            no_verification();
        }
    } else if (op_type == HCCL_REDUCE_SUM) {
        if((dtype == HCCL_DATA_TYPE_INT8 || dtype == HCCL_DATA_TYPE_UINT8) &&
            rank_size >= RANKSIZE_TH_INT64) {
            no_verification();
        }
    }

    return;
}

void HcclOpBaseTest::print_execution_time(double average_time_us, double algorithm_bandwith_GBytes_s)
{
    //不开启结果校验场景
    if (check == 0)
    {
        if (rank_id == root_rank) {
            if (print_header)
            {
                printf("%-15s | %-12s | %-18s | %s\n", data_size, aveg_time, alg_bandwidth, verification_result);
                print_header = false;
            }
            printf("%-17llu | %-14.2f | %-20.5f | NULL\n", data->data_size, average_time_us, algorithm_bandwith_GBytes_s);
        }
        return;
    }

    // 开启结果校验，部分rank结果校验失败场景
    bool check_result[rank_size];
    if (check_err != 0)
    {
        check_result[rank_id] = false; // 结果校验失败
        printf("local rank_id %d, check result failed, %-17llu | %-14.2f | %-20.5f | failed\n", rank_id, data->data_size, 
            average_time_us, algorithm_bandwith_GBytes_s);
    } else {
        check_result[rank_id] = true; // 结果校验成功
    }

    if (rank_id == root_rank)
    {
        if (print_header)
        {
            printf("%-15s | %-12s | %-18s | %s\n", data_size, aveg_time, alg_bandwidth, verification_result);
            print_header = false;
        }

        if (!check_result[root_rank])
        {
            printf("%-17llu | %-14.2f | %-20.5f | failed\n", data->data_size, average_time_us, algorithm_bandwith_GBytes_s);
        } else {
            printf("%-17llu | %-14.2f | %-20.5f | success\n", data->data_size, average_time_us, algorithm_bandwith_GBytes_s);
        }
    }
    return;
}

int HcclOpBaseTest::destory_alloc_buf()
{
    if (host_buf != nullptr) {
        ACLCHECK(aclrtFreeHost(host_buf));
        host_buf = nullptr;
    }
    if (recv_buff_temp != nullptr) {
        ACLCHECK(aclrtFreeHost(recv_buff_temp));
        recv_buff_temp = nullptr;
    }
    if (check_buf != nullptr) {
        ACLCHECK(aclrtFreeHost(check_buf));
        check_buf = nullptr;
    }
    return 0;
}
}
