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

#ifndef __HCCL_ALLREDUCE_ROOTINFO_TEST_H_
#define __HCCL_ALLREDUCE_ROOTINFO_TEST_H_
#include "hccl_test_common.h"
#include "mpi.h"
#include "hccl_check_common.h"
#include "hccl_opbase_rootinfo_base.h"
namespace hccl {
#define ALLREDUCE_SUM_MAX_RANKSIZE(HcclDataType) int64_t(((int64_t(1) << int(HcclDataTypePrecision::CONCAT(HcclDataType, ))) - 1) / 2)
#define ALLREDUCE_SUM_RESULE_OVERFLOW(rank_size,HcclDataType) bool(int64_t(rank_size) > ALLREDUCE_SUM_MAX_RANKSIZE(HcclDataType))
class HcclOpBaseAllreduceTest:public HcclOpBaseTest
{
public:
    HcclOpBaseAllreduceTest();
    virtual ~HcclOpBaseAllreduceTest();
    virtual int hccl_op_base_test(); //主函数
    void is_data_overflow() override;

protected:
    size_t init_malloc_Ksize_by_data() override;
    void init_send_recv_size_by_data(size_t &send_bytes, size_t &recv_bytes) override;

private:
    virtual int init_buf_val();  //（初始化host_buf，初始化check_buf，拷贝到send_buf） 其中需要调用hccl_host_buf_init
    virtual int check_buf_result();//（recv_buf拷贝到recvbufftemp,并且校验正确性）需要调用check_buf_init，校验正确性要调用check_buf_result_float
    void cal_execution_time(float time);//统计耗时
};
}
#endif
