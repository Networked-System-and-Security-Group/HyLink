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

#ifndef _HCCL_CHECK_COMMOM_H_
#define _HCCL_CHECK_COMMOM_H_
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <hccl/hccl_types.h>
#include "hccl_test_common.h"

//浮点数计算精度，当前算误差百分比
#define HCCL_EPSION_FLOAT 0.000001

extern int check_buf_result_float(const void* result_buf, const void* check_buf, unsigned long long count);
extern int check_buf_result_int8(const void* result_buf, const void* check_buf, unsigned long long count);
extern int check_buf_result_half(const void* result_buf, const void* check_buf, unsigned long long count);
extern int check_buf_result_int32(const void* result_buf, const void* check_buf, unsigned long long count);
extern int check_buf_result_int64(const void* result_buf, const void* check_buf, unsigned long long count);
extern int check_buf_result_u64(const void* result_buf, const void* check_buf, unsigned long long count);

#endif