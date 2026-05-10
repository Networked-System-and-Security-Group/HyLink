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

#ifndef _HCCL_CHECK_BUF_INIT_H_
#define _HCCL_CHECK_BUF_INIT_H_
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
#include <map>

static inline float fp32_from_bits(uint32_t w)
{
    union {
        uint32_t as_bits;
        float as_value;
    } fp32 = { w };
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f)
{
    union {
        float as_value;
        uint32_t as_bits;
    } fp32 = { f };
    return fp32.as_bits;
}

static inline uint16_t fp16_ieee_from_fp32_value(float f)
{
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

static inline uint16_t fp32tobf16(float x){
	float y = x;
    int *p = (int *) &y;
    unsigned int exp, man;
    exp = *p & 0x7F800000u;
    man = *p & 0x007FFFFFu;
    if (exp == 0 && man == 0) {
        // zero
        return x; 
    }
    if (exp == 0x7F800000u) {
        // infinity or Nans
        return x;
    }
    // Normalized number
    // round to nearest
    float r = x;
    int *pr = (int *) &r;
    *pr &= 0xff800000; // r has the same exp as x
    r = r / 256;
    y = x + r;

    *p &= 0xffff0000;

    return y;
}

typedef void(*HostBufInitFunc)(void *, u64, int);
extern std::map<int,HostBufInitFunc> functionMap;

typedef void(*ReduceCheckBufInitFunc)(void *, u64, int, int, int);
extern std::map<int,ReduceCheckBufInitFunc> functionReduceMap;

typedef int(*AllToAllCheckResult)(const void*, u64*, u64*, int, int, int);
extern std::map<int,AllToAllCheckResult> functionAllToAllMap;

extern void hccl_host_buf_init(void *dst_buf, unsigned long long count, int dtype, int val);
extern void hccl_reduce_check_buf_init(
    void *dst_buf, unsigned long long count, int dtype, int op, int val, int rank_size);
extern int hccl_alltoallv_check_result(void *check_buf, unsigned long long *recv_counts, unsigned long long *recv_disp,
    int rank_id, int rank_size, int dtype);
extern bool hccl_alltoall_check_result(const void *recv_buff, const std::size_t count, const int nRanks,
    const int rank, const std::vector<std::uint8_t> &pattern);

#endif