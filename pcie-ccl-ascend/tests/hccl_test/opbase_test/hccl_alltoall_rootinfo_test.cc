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
#include "hccl_alltoall_rootinfo_test.h"
#include "hccl_opbase_rootinfo_base.h"
#include "hccl_check_buf_init.h"
#include <array>
#include <cstdint>
#include <thread>
using namespace hccl;

HcclTest* hccl::init_opbase_ptr(HcclTest* opbase)
{
    opbase = new HcclOpBaseAlltoallTest();
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
HcclOpBaseAlltoallTest::HcclOpBaseAlltoallTest()
    : HcclOpBaseTest(),
      sendCount_(0),
      recvCount_(0)
{
}

HcclOpBaseAlltoallTest::~HcclOpBaseAlltoallTest()
{
}

constexpr std::array<std::uint8_t, 8> pattern = {0x5a, 0x5a, 0x5a, 0x5a, 0xa5, 0xa5, 0xa5, 0xa5};
static int fill_alternating_pattern(void *send_buff, const std::size_t count)
{
    if (count == 0) {
        return 0;
    }
    const auto dst_buf = static_cast<std::uint8_t *>(send_buff);
    std::size_t copied = 0;
    if (count >= pattern.size()) {
        ACLCHECK(aclrtMemcpy((void *)dst_buf, pattern.size(), pattern.data(), pattern.size(), ACL_MEMCPY_HOST_TO_HOST));
        copied = pattern.size();
    } else {
        ACLCHECK(aclrtMemcpy((void *)dst_buf, count, pattern.data(), count, ACL_MEMCPY_HOST_TO_HOST));
        return 0;
    }
    while (copied < count) {
        std::size_t to_copy = copied;
        if (copied + to_copy > count) {
            to_copy = count - copied;
        }
        ACLCHECK(aclrtMemcpy((void *)(dst_buf + copied), to_copy, dst_buf, to_copy, ACL_MEMCPY_HOST_TO_HOST));
        copied += to_copy;
    }
    return 0;
}

int HcclOpBaseAlltoallTest::check_buf_result()
{
    //获取输出内存
    ACLCHECK(aclrtMallocHost((void**)&check_buf, malloc_kSize));
    ACLCHECK(aclrtMemcpy((void *)check_buf, malloc_kSize, (void *)recv_buff, malloc_kSize, ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<std::uint8_t> memortPattern(pattern.begin(), pattern.end());
    if (!hccl_alltoall_check_result(check_buf, malloc_kSize, rank_size, rank_id, memortPattern)) {
        check_err++;
    }
    return 0;
}

void HcclOpBaseAlltoallTest::cal_execution_time(float time)
{
    double total_time_us = time * 1000;
    double average_time_us = total_time_us / iters;
    double algorithm_bandwith_GBytes_s = malloc_kSize / average_time_us * B_US_TO_GB_S;

    print_execution_time(average_time_us, algorithm_bandwith_GBytes_s);
    return;
}

void HcclOpBaseAlltoallTest::free_send_recv_buf()
{
}

size_t HcclOpBaseAlltoallTest::init_malloc_Ksize_by_data()
{
    u64 alignedData = data->count >= rank_size * granularity || enable_zero_copy ? rank_size * granularity : rank_size;
    data->count = (data->count + alignedData - 1) / alignedData * alignedData;
    return data->count * data->type_size;
}

void HcclOpBaseAlltoallTest::init_send_recv_size_by_data(size_t &send_bytes, size_t &recv_bytes)
{
    send_bytes = malloc_kSize;
    recv_bytes = malloc_kSize;
}

int HcclOpBaseAlltoallTest::hccl_op_base_test() //主函数
{
    if (op_flag != 0 && rank_id == root_rank) {
        printf("Warning: The -o,--op <sum/prod/min/max> option does not take effect. Check the cmd parameter.\n");
    }

    is_initdata_overflow();
    //初始化输入内存
    ACLCHECK(aclrtMallocHost((void**)&host_buf, malloc_kSize));
    ACLCHECK(fill_alternating_pattern((char *)host_buf, malloc_kSize));
    ACLCHECK(aclrtMemcpy((void*)send_buff, malloc_kSize, (void*)host_buf, malloc_kSize, ACL_MEMCPY_HOST_TO_DEVICE));

    sendCount_ = data->count / rank_size;
    recvCount_ = data->count / rank_size;
    // 输入数据量，根据条件判断是否开启仅计算device执行时间
 	ACLCHECK(start_profile_device_time_if_needed(16*1024*1024));
    //执行集合通信操作
    for(int j = 0; j < warmup_iters; ++j) {
        HCCLCHECK(HcclAlltoAll((void *)send_buff, sendCount_, (HcclDataType)dtype,\
            (void*)recv_buff, recvCount_, (HcclDataType)dtype, hccl_comm, stream));
    }

    ACLCHECK(aclrtRecordEvent(start_event, stream));

    for(int i = 0; i < iters; ++i) {
        HCCLCHECK(HcclAlltoAll((void *)send_buff, sendCount_, (HcclDataType)dtype,\
            (void*)recv_buff, recvCount_, (HcclDataType)dtype, hccl_comm, stream));
    }
    //等待stream中集合通信任务执行完成
    ACLCHECK(aclrtRecordEvent(end_event, stream));
    ACLCHECK(end_profile_device_time_if_needed(16*1024*1024));
    ACLCHECK(aclrtSynchronizeStream(stream));

    float time;
    ACLCHECK(aclrtEventElapsedTime(&time, start_event, end_event));

    // 校验计算结果
    if (check == 1) {
        ACLCHECK(check_buf_result());
    }

    cal_execution_time(time);
    return 0;
}
}