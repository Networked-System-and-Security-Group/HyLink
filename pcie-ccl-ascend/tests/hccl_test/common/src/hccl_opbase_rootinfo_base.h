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

#ifndef __HCCL_OPBASE_ROOTINFO_BASE_H_
#define __HCCL_OPBASE_ROOTINFO_BASE_H_
#include "hccl_test_common.h"
#include "mpi.h"
#include "hccl_check_common.h"
#include <hccl/hccl_types.h>
#include "hccl_check_buf_init.h"

namespace hccl
{
constexpr double B_US_TO_GB_S = 1.0E6 / 1.0E9;
constexpr int RANKSIZE_TH_FP16 = 16;
constexpr int RANKSIZE_TH_FP32 = 128;
constexpr int RANKSIZE_TH_INT8 = 7;
constexpr int RANKSIZE_TH_INT32 = 31;
constexpr int RANKSIZE_TH_INT64 = 63;
/**
 * @brief HCCL data type Precision
 */
typedef enum {
    HCCL_DATA_TYPE_INT8_Precision = 7,    /**< int8 */
    HCCL_DATA_TYPE_INT16_Precision = 15,   /**< int16 */
    HCCL_DATA_TYPE_INT32_Precision = 31,   /**< int32 */
    HCCL_DATA_TYPE_INT64_Precision = 63,    /**< int64 */
    HCCL_DATA_TYPE_FP16_Precision = 10,    /**< fp16 */
    HCCL_DATA_TYPE_FP32_Precision = 23,    /**< fp32 */
    HCCL_DATA_TYPE_BFP16_Precision = 7,    /**< bfp16 */
} HcclDataTypePrecision;
#define CONCAT(x, y) x ## _Precision ## y

class HcclOpBaseTest:public HcclTest
{
public:
    HcclOpBaseTest();
    virtual ~HcclOpBaseTest();

    virtual int hccl_op_base_test(); //主函数
    virtual void init_data_count(); //初始化malloc_kSize
    virtual void no_verification();
    virtual void is_data_overflow();
    virtual void is_initdata_overflow();
    virtual void print_execution_time(double average_time_us, double algorithm_bandwith_GBytes_s); //打印耗时

    virtual int destory_alloc_buf() override;
public:
    void *host_buf;
    void *recv_buff_temp;
    void *check_buf;
    int check_err = 0;
    int val = 2; //校验参数

private:
    virtual int init_buf_val();  //（初始化host_buf，初始化check_buf，拷贝到send_buf） 其中需要调用hccl_host_buf_init
    virtual int check_buf_result();//（recv_buf拷贝到recvbufftemp,并且校验正确性）需要调用check_buf_init，校验正确性要调用check_buf_result_float
    const char *data_size = {"data_size(Bytes):"};
    const char *aveg_time = {"aveg_time(us):"};
    const char *alg_bandwidth = {"alg_bandwidth(GB/s):"};
    const char *verification_result = {"check_result:"};
};
}
#endif
