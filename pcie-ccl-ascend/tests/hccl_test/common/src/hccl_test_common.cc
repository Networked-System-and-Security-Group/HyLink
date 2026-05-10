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
#include <vector>
#include "mpi.h"
#include "hccl_opbase_rootinfo_base.h"
#include "hccl_allgather_rootinfo_test.h"
#include "hccl_test_common.h"
#include <algorithm>
#include <arpa/inet.h>
#include <thread>
constexpr s32 HCCL_TEST_REDUCE_RESERVED = 4;
constexpr s32 HCCL_TEST_DATA_TYPE_RESERVED = 18;
constexpr s32 HCCL_TEST_ACCELERATOR_CONFIG_RESERVED = 8;
constexpr u32 HCCL_TEST_DATATYPE_BF16_SAT = 13;
HcclReduceOp test_ops[HCCL_TEST_REDUCE_RESERVED] = {
    HCCL_REDUCE_SUM, HCCL_REDUCE_PROD, HCCL_REDUCE_MAX, HCCL_REDUCE_MIN};
const char *test_opnames[HCCL_TEST_REDUCE_RESERVED] = {"sum", "prod", "max", "min"};
const char *test_accelerator_config[HCCL_TEST_ACCELERATOR_CONFIG_RESERVED] = {
    "default", "host_ts", "aicpu_ts", "aiv", "aiv_only", "ccu_ms", "ccu_sched", "aicpu"};

HcclDataType test_types[HCCL_TEST_DATA_TYPE_RESERVED] = {HCCL_DATA_TYPE_INT8, /**< int8 */
    HCCL_DATA_TYPE_INT16,                                                     /**< int16 */
    HCCL_DATA_TYPE_INT32,                                                     /**< int32 */
    HCCL_DATA_TYPE_FP16,                                                      /**< fp16 */
    HCCL_DATA_TYPE_FP32,                                                      /**< fp32 */
    HCCL_DATA_TYPE_INT64,                                                     /**< int64 */
    HCCL_DATA_TYPE_UINT64,                                                    /**< uint64 */
    HCCL_DATA_TYPE_UINT8,                                                     /**< uint8 */
    HCCL_DATA_TYPE_UINT16,                                                    /**< uint16 */
    HCCL_DATA_TYPE_UINT32,                                                    /**< uint32 */
    HCCL_DATA_TYPE_FP64,                                                      /**< fp64 */
    HCCL_DATA_TYPE_BFP16,                                                     /**< bfp16 */
    HCCL_DATA_TYPE_INT128,                                                    /**< int128 */
    HCCL_DATA_TYPE_HIF8,                                                      /**< hif8 */
    HCCL_DATA_TYPE_FP8E4M3,                                                   /**< fp8e4m3 */
    HCCL_DATA_TYPE_FP8E5M2,                                                   /**< fp8e5m2 */
    HCCL_DATA_TYPE_FP8E8M0,                                                   /**< fp8e8m0 */
    HCCL_DATA_TYPE_MXFP8                                                      /**< mxfp8 */
};
const char *test_typenames[HCCL_TEST_DATA_TYPE_RESERVED] = {
    "int8", "int16", "int32", "fp16", "fp32", "int64", "uint64", "uint8", "uint16", "uint32", "fp64", "bfp16",
    "int128", "hif8", "fp8e4m3", "fp8e5m2", "fp8e8m0", "mxfp8"};

int get_hccl_op_from_str(char *str)
{
    for (int op = 0; op < HCCL_TEST_REDUCE_RESERVED; op++) {
        if (strcmp(str, test_opnames[op]) == 0) {
            return op;
        }
    }

    return -1;
}

int get_hccl_dtype_from_str(char *str)
{
    for (int t = 0; t < HCCL_TEST_DATA_TYPE_RESERVED; t++) {
        if (strcmp(str, test_typenames[t]) == 0) {
            if (t >= HCCL_TEST_DATATYPE_BF16_SAT) {
                t += 1; // 由于HcclDataType跳过枚举值13，需要保证对应正确的枚举值
            }
            return t;
        }
    }
    return -1;
}

int get_accelerator_config_value_from_str(char *str)
{
    for (int t = 0; t < HCCL_TEST_ACCELERATOR_CONFIG_RESERVED; t++) {
        if (strcmp(str, test_accelerator_config[t]) == 0) {
            return t;
        }
    }
    return -1;
}

static void get_host_name(char *hostName, int maxlen)
{
    gethostname(hostName, maxlen);
    for (int i = 0; i < maxlen; i++) {
        if (hostName[i] == '.') {
            hostName[i] = '\0';
            return;
        }
    }
    return;
}

static uint64_t get_host_hash(const char *string)
{
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

static long parsesize(const char *value)
{
    long long int units;
    long size;
    char *size_lit;
    size = strtol(value, &size_lit, 0);
    if (strlen(size_lit) == 1) {
        switch (*size_lit) {
            case 'G':
            case 'g':
                units = GIGA;
                break;
            case 'M':
            case 'm':
                units = MEGA;
                break;
            case 'K':
            case 'k':
                units = KILO;
                break;
            default:
                return -1;
        }
    } else if (strlen(size_lit) == 0) {
        units = 1;
    } else {
        return -1;
    }

    return size * units;
}

u32 sal_str_len(const char *s, u32 maxLen = INT_MAX)
{
    return strnlen(s, maxLen);
}

int is_all_digit(const char *strNum)
{
    // 参数有效性检查
    if (strNum == NULL) {
        printf("Error: ptr [%s] is NULL\n", strNum);
        return -1;
    }

    u32 nLength = sal_str_len(strNum);
    for (u32 index = 0; index < nLength; index++) {
        if (!isdigit(strNum[index])) {
            printf("Error:In judge all digit, Check whether the value of [-i -n -r -w -c -p] is a positive integer.\n");
            return -1;
        }
    }
    return 0;
}

long strtol_alldigit(const char *optarg)
{
    long ret = is_all_digit(optarg);
    if (ret != 0) {
        return ret;
    }

    return strtol(optarg, NULL, 0);
}

aclrtMemAttr use_ddr_mem()
{
    constexpr char ASCEND910[] = "Ascend910";
    constexpr char ASCEND910B1[] = "Ascend910B1";
    constexpr char ASCEND910B2[] = "Ascend910B2";
    constexpr char ASCEND910B2C[] = "Ascend910B2C";
    constexpr char ASCEND910B3[] = "Ascend910B3";
    constexpr char ASCEND910B4[] = "Ascend910B4";
    constexpr char ASCEND910B4_1[] = "Ascend910B4-1";
    constexpr char ASCEND910_9391[] = "Ascend910_9391";
    constexpr char ASCEND910_9381[] = "Ascend910_9381";
    constexpr char ASCEND910_9392[] = "Ascend910_9392";
    constexpr char ASCEND910_9382[] = "Ascend910_9382";
    constexpr char ASCEND910_9372[] = "Ascend910_9372";
    constexpr char ASCEND910_9361[] = "Ascend910_9361";
    constexpr char ASCEND610[] = "Ascend610";
    constexpr char ASCEND610B1[] = "Ascend610B1";
    constexpr char ASCEND610B2[] = "Ascend610B2";
    constexpr char ASCEND610LITE[] = "Ascend610Lite";
    constexpr char BS9SX1A[] = "BS9SX1A";
    constexpr char BS9SX1AA[] = "BS9SX1AA";
    constexpr char BS9SX1AB[] = "BS9SX1AB";
    constexpr char BS9SX1AC[] = "BS9SX1AC";
    constexpr char BS9SX2A[] = "BS9SX2A";
    constexpr char BS9SX2AA[] = "BS9SX2AA";
    constexpr char BS9SX2AB[] = "BS9SX2AB";
    constexpr char MC61AM21A[] = "MC61AM21A";
    constexpr char MC61AM21AA[] = "MC61AM21AA";
    constexpr char MC61AM21AB[] = "MC61AM21AB";
    constexpr char ASCEND310P[] = "Ascend310P";
    constexpr char ASCEND310P1[] = "Ascend310P1";
    constexpr char ASCEND310P2[] = "Ascend310P2";
    constexpr char ASCEND310P3[] = "Ascend310P3";
    constexpr char ASCEND310P4[] = "Ascend310P4";
    constexpr char ASCEND310B1[] = "Ascend310B1";
    constexpr char ASCEND310B2[] = "Ascend310B2";
    constexpr char ASCEND310B3[] = "Ascend310B3";
    constexpr char ASCEND310B4[] = "Ascend310B4";
    constexpr char ASCEND310B[] = "Ascend310B";
    constexpr char AS31XM1X[] = "AS31XM1X";
    std::string socVersion = aclrtGetSocName();
    if (socVersion.compare(0, strlen(ASCEND310P), ASCEND310P) == 0 ||
        socVersion.compare(0, strlen(BS9SX1A), BS9SX1A) == 0 ||
        socVersion.compare(0, strlen(ASCEND610), ASCEND610) == 0 ||
        socVersion.compare(0, strlen(ASCEND610LITE), ASCEND610LITE) == 0 ||
        socVersion.compare(0, strlen(BS9SX2A), BS9SX2A) == 0 ||
        socVersion.compare(0, strlen(MC61AM21A), MC61AM21A) == 0 ||
        socVersion.compare(0, strlen(ASCEND310B), ASCEND310B) == 0 ||
        socVersion.compare(0, strlen(AS31XM1X), AS31XM1X) == 0) {
        return aclrtMemAttr::ACL_DDR_MEM_HUGE;
    }
    return aclrtMemAttr::ACL_HBM_MEM_HUGE;
}

int check_alloc_memory_size(size_t &alloc_size)
{
    size_t freeMem = 0;
    size_t totalMem = 0;
    size_t reservedMem;
    aclError ret = aclrtGetMemInfo(use_ddr_mem(), &freeMem, &totalMem);
    if (ret != ACL_ERROR_NONE) {
        printf("Get ddr memory info failed.");
        return HCCL_E_MEMORY;
    }
    if (alloc_size > freeMem) {
        printf("Alloc memory size is larger than free memory, allocSize: %zu, freeMem: %zu, totalMem: %zu\n",
            alloc_size,
            freeMem,
            totalMem);
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}

namespace hccl {
HcclTest::HcclTest()
{
    data = new DataSize;
    data->step_factor = 1;
    proc_rank = 0;
    proc_size = 0;
}

HcclTest::~HcclTest()
{
    delete data;
    data = nullptr;
}

struct option HcclTest::longopts[] = {{"op", required_argument, 0, 'o'},
    {"datatype", required_argument, 0, 'd'},
    {"minbytes", required_argument, 0, 'b'},
    {"maxbytes", required_argument, 0, 'e'},
    {"stepbytes", required_argument, 0, 'i'},
    {"stepfactor", required_argument, 0, 'f'},
    {"root", required_argument, 0, 'r'},
    {"iters", required_argument, 0, 'n'},
    {"warmup_iters", required_argument, 0, 'w'},
    {"check", required_argument, 0, 'c'},
    {"npus", required_argument, 0, 'p'},
    {"help", no_argument, 0, 'h'},
    {"zero_copy", required_argument, 0, 'z'},
    {"nslb", required_argument, 0, 's'},
    {"onlydevicetime", required_argument, 0, 't'},
    {"symmetric_memory", required_argument, 0, 'm'}
};

void HcclTest::print_help()
{
    printf("USAGE: ./test \n\t");
    if (IsSupport910_95()) {
        printf("[-a --accelerator <default/host_ts/aicpu_ts/aiv/aiv_only/ccu_ms/ccu_sched>] \n\t");
    }
    printf("[-b,--minbytes <min size in bytes>] \n\t"
           "[-e,--maxbytes <max size in bytes>] \n\t"
           "[-i,--stepbytes <increment size>] \n\t"
           "[-f,--stepfactor <increment factor>] \n\t"
           "[-n,--iters <iteration count>] \n\t"
           "[-o,--op <sum/prod/min/max>] \n\t"
           "[-d,--datatype <int8/int16/int32/fp16/fp32/int64/uint64/uint8/uint16/uint32/fp64/bfp16/int128/hif8/fp8e4m3/fp8e5m2/fp8e8m0/mxfp8>] \n\t"
           "[-r,--root <root>] \n\t"
           "[-w,--warmup_iters <warmup iteration count>] \n\t"
           "[-c,--check <result verification> 0:disabled 1:enabled.] \n\t"
           "[-p,--npus <npus used for one node>] \n\t"
           "[-z,--zero_copy  0:disabled 1:enabled.] \n\t"
           "[-m,--symmetric_memory  0:disabled 1:enabled.] \n\t"
           "[-s,--nslb  0:disabled 1:enabled.] \n\t"
           "[-t, --onlydevicetime 0:disabled 1:enabled. When -t is 1,-n and -w must be less than or equal to 100, not support aicpu_ts.] \n\t"
           "[-h,--help]\n");
    return;
}

int HcclTest::check_data_count()
{
    if (data_parsed_begin <= 0 || data_parsed_end <= 0) {
        printf("invalid size specified for [-b,--minbytes] or [-e,--maxbytes]\n");
        return -1;
    }
    data->min_bytes = (u64)data_parsed_begin;
    data->max_bytes = (u64)data_parsed_end;

    if (stepbytes_flag != 0 && temp_step_bytes < 0) {
        printf("Error: [-i,--stepbytes] must be greater than or equal to 0.\n");
        return -1;
    }

    u32 defaultStepFactor = 10;
    if (data->max_bytes < data->min_bytes) {
        printf("invalid option: maxbytes < minbytes, Check the [-b,--minbytes] and [-e,--maxbytes] options.\n");
        return -1;
    } else {
        if (stepbytes_flag != 0) {  // 用户配置了增量步长
            data->step_bytes = temp_step_bytes;
        } else {  // 用户未配置增量步长
            if (data->max_bytes == data->min_bytes) {
                data->step_bytes =
                    1;  // 用户配置数据量的起始值和结束值相同，但未配置增量步长，为防止进入死循环，设置增量步长为1
            }
            if (data->max_bytes > data->min_bytes) {
                data->step_bytes = (data->max_bytes - data->min_bytes) / defaultStepFactor;
            }
        }
    }

    // 数据量需要被ranksize整除
    if (need_ranksize_alignment) {
        // 计算对齐大小：rank_size * BUF_ALGIN_SIZE(512)
        u64 align_size = rank_size * BUF_ALGIN_SIZE;

        // 辅助函数：对齐字节数
        auto align_bytes = [&](u64 bytes) -> u64 {
            if (bytes < rank_size) {
                bytes = rank_size;
            }
            if (bytes > BUF_ALGIN_LINE) {
                // 大于BUF_ALGIN_LINE时，对齐到align_size的倍数
                bytes = (bytes / align_size) * align_size;
            } else {
                // 小于等于BUF_ALGIN_LINE时，对齐到rank_size的倍数
                bytes = (bytes / rank_size) * rank_size;
            }
            return bytes;
        };

        // 对齐min_bytes和max_bytes
        data->min_bytes = align_bytes(data->min_bytes);
        data->max_bytes = align_bytes(data->max_bytes);

        // 对齐step_bytes到rank_size的倍数（向上取整）
        data->step_bytes = ((data->step_bytes + rank_size - 1) / rank_size) * rank_size;
    }

    if (stepfactor_flag != 0 && data->step_factor <= 1.0) {
        printf("Error: [-f,--stepfactor] Must be greater than 1.0f, Start step mod.\n");
        return -1;
    }

    if (stepfactor_flag != 0 && stepbytes_flag != 0) {
        printf("Warning: [-f,--stepfactor] and [-i,--stepbytes] are set, [-f,--stepfactor] is enabled by default.\n");
    }

    return 0;
}

int HcclTest::check_only_device_exec_time()
{
    if (only_device_exec_time != 1 && only_device_exec_time != 0) {
        printf("Error: [-t,--onlydevicetime] is invalid, onlydevicetime should be 0 or 1\n");
        return -1;
    }

    if (only_device_exec_time == 1 && warmup_iters > 100) {
        printf("Error: [-w,--warmup_iters] is invalid, onlydevicetime is 1, but warmup_iters > 100, "
            "warmup_iters must be less than or equal to 100, now warmup_iters is %d\n", warmup_iters);
        return -1;
    }

    if (only_device_exec_time == 1 && iters > 100) {
        printf("Error: [-n,--iters] is invalid, onlydevicetime is 1, but iters > 100, "
            "iters must be less than or equal to 100, now iters is %d\n", iters);
        return -1;
    }

    if (only_device_exec_time == 1 && accelerator_config == 2) {
        printf("Error: [-a,--accelerator_config] is invalid, onlydevicetime is 1, but accelerator_config is aicpu_ts\n");
        return -1;
    }
    return 0;
}

int HcclTest::check_cmd_line()
{
    int ret = 0;
    ret = check_data_count();
    if (ret != 0) {
        return ret;
    }

    if (dtype == -1) {
        printf("Error: [-d,--datatype] is invalid, Use [-h,--help] to check the correct input parameter.\n");
        return -1;
    }

    if (op_type == -1) {
        printf("Error: [-o,--op] is invalid, Use [-h,--help] to check the correct input parameter.\n");
        return -1;
    }

    if (accelerator_config == -1) {
        printf("Error: [-a,--accelerator_config] is invalid, Use [-h,--help] to check the correct input parameter.\n");
        return -1;
    }

    if (warmup_iters < 0) {
        printf("Error: [-w,--warmup_iters] is invalid, warmup_iters must be greater than or equal to 0.\n");
        return -1;
    }

    if (iters < 0) {
        printf("Error: [-n,--iters] is invalid, iters must be greater than or equal to 0.\n");
        return -1;
    }

    if (root_rank >= rank_size || root_rank < 0)  // 如果指定的root rank大于等于rank_size
    {
        printf("Error: [-r,--root <root>] is invalid, root rank must be greater than or equal to 0 and less than or "
               "equal to %d.\n",
            rank_size - 1);
        return -1;
    }

    if (check != 1 && check != 0) {
        printf("Error: [-c,--check] is invalid, check should be 0 or 1\n");
        return -1;
    }

    if (dev_count == 0) {
        printf("Error: The number of device is 0.Check whether the package is correct.\n");
        return -1;
    }

    if (npus < 1 || npus > dev_count) {
        printf("Error: [-p,--npus <npus used for one node>] is invalid, npus must be greater than or equal to 1 and "
               "less than or equal to %d.\n",
            dev_count);
        return -1;
    }

    if (check_only_device_exec_time() == -1) {
        return -1;
    }

    if (enable_zero_copy && enable_symmetric_memory) {
        printf("Error: Zero-copy and symmetric memory cannot be enabled simultaneously.\n");
        return -1;
    }

    return 0;
}

int HcclTest::get_env_resource()
{
    // 支持profiling
    const char *profiling_env = getenv("HCCL_TEST_PROFILING");
    if (profiling_env != NULL) {
        profiling_flag = atoi(profiling_env);
        u32 nLength = sal_str_len(profiling_env);
        // 校验：入参为字符
        for (u32 index = 0; index < nLength; index++) {
            if (!isdigit(profiling_env[index])) {
                printf("Check whether HCCL_TEST_PROFILING is 0 or 1.\n");
                return -1;
            }
        }
        // 校验：入参非0非1
        if (profiling_flag != 0 && profiling_flag != 1) {
            printf("Check whether HCCL_TEST_PROFILING is 0 or 1.\n");
            return -1;
        }
    }

    // 获取HCCL BUFFER SIZE
    const char *cclBuffSize_env = getenv("HCCL_BUFFSIZE");
    if (cclBuffSize_env != NULL) {
        hccl_buffsize = atoi(cclBuffSize_env);
        u32 nLength = sal_str_len(cclBuffSize_env);
        // 校验：入参为字符
        for (u32 index = 0; index < nLength; index++) {
            if (!isdigit(cclBuffSize_env[index])) {
                printf("Check whether HCCL_BUFFSIZE is all digit.\n");
                return -1;
            }
        }

        // 校验：入参非0
        if (hccl_buffsize == 0) {
            printf("Check whether HCCL_BUFFSIZE is 0 or too big.\n");
            return -1;
        }

        if (only_device_exec_time == 1 && hccl_buffsize <= 100) {
            printf("Warning: -t is 1 but HCCL_BUFFSIZE <= 100MB , -t auto reset to 0!\n");
            only_device_exec_time = 0;
        }
    } else {
        hccl_buffsize = 200;
    }

    // 开启profiling
    if (profiling_flag == 1) {
        std::string prof_path = "/var/log/npu/profiling";
        const char *profiling_env_path = getenv("HCCL_TEST_PROFILING_PATH");
        if (profiling_env_path != NULL) {
            prof_path = profiling_env_path;
        }
        aclprofInit(prof_path.c_str(), prof_path.size());
        uint32_t profSwitch = ACL_PROF_ACL_API | ACL_PROF_TASK_TIME | ACL_PROF_AICORE_METRICS | ACL_PROF_AICPU |
                              ACL_PROF_HCCL_TRACE | ACL_PROF_MSPROFTX | ACL_PROF_RUNTIME_API;
        uint32_t deviceIdList = dev_id;
        int devNum = 1;
        profiling_config = aclprofCreateConfig(&deviceIdList, devNum, ACL_AICORE_PIPE_UTILIZATION, nullptr, profSwitch);
        ACLCHECK(aclprofStart(profiling_config));
    }
    return 0;
}

int HcclTest::set_env_resource()
{
    const char *retry_env = getenv("HCCL_OP_RETRY_ENABLE");
    if (retry_env != NULL) {
        return 0;
    } else {
        // 重执行开关
        int overwrite = 1; // 强制覆盖已存在的变量
        if(setenv("HCCL_OP_RETRY_ENABLE", "L0:0, L1:0, L2:0", overwrite) == -1) {
            printf("setenv HCCL_OP_RETRY_ENABLE failed\n");
            return -1;
        }
    }
    return 0;
}

int HcclTest::release_env_resource()
{
    if (profiling_flag == 1) {
        ACLCHECK(aclprofStop(profiling_config));
        aclprofFinalize();
    }

    return 0;
}

int HcclTest::parse_opt(int opt)
{
    switch (opt) {
        case 'a':
            accelerator_config = get_accelerator_config_value_from_str(optarg);
            break;
        case 'b':
            data_parsed_begin = parsesize(optarg);
            break;
        case 'e':
            data_parsed_end = parsesize(optarg);
            break;
        case 'i':
            stepbytes_flag++;
            temp_step_bytes = strtol_alldigit(optarg);
            break;
        case 'f':
            stepfactor_flag++;
            char *temp;
            data->step_factor = strtof(optarg, &temp);
            break;
        case 'n':
            iters = strtol_alldigit(optarg);
            break;
        case 'o':
            op_flag++;
            op_type = get_hccl_op_from_str(optarg);
            break;
        case 'd':
            dtype = get_hccl_dtype_from_str(optarg);
            break;
        case 'r':
            root_rank = strtol_alldigit(optarg);
            break;
        case 'w':
            warmup_iters = strtol_alldigit(optarg);
            break;
        case 'c':
            check = strtol_alldigit(optarg);
            break;
        case 'p':
            npus = strtol_alldigit(optarg);
            npus_flag = 1;
            break;
        case 'z':
            enable_zero_copy = std::stoi(optarg);
            break;
        case 'm':
            enable_symmetric_memory = std::stoi(optarg);
            break;
        case 's':
            nslb_flag = 1;
            break;
        case 't':
            only_device_exec_time = strtol_alldigit(optarg);
            break;
        case 'h':
            print_help();
            return 1;
        default:
            printf("invalid option \n");
            printf("Try [-h --help] for more information.\n");
            return -1;
    }
    return 0;
}

int HcclTest::parse_cmd_line(int argc, char *argv[])
{
    int opt = -1;
    int longindex = 0;
    int ret = 0;
    long parsed;
    while (-1 != (opt = getopt_long(argc, argv, "o:d:b:e:i:f:r:n:w:c:p:z:a:s:t:m:h", longopts, &longindex))) {
        ret = parse_opt(opt);
        if (ret != 0) {
            return ret;
        }
    }

    if (optind < argc) {
        printf("non-option ARGV-elements: ");
        while (optind < argc) {
            printf("%s ", argv[optind++]);
        }
        printf("\n");
        return -1;
    }

    return 0;
}

int HcclTest::getAviDevs(const char *devs, std::vector<int> &dev_ids)
{
    std::string use_devs(devs);
    std::string pattern = ",";
    std::string::size_type pos;
    use_devs += pattern;
    size_t val_size = use_devs.size();
    for (size_t i = 0; i < val_size; ++i) {
        pos = use_devs.find(pattern, i);
        if (pos < val_size) {
            std::string s = use_devs.substr(i, pos);
            int tmp_rank = atoi(s.c_str());
            dev_ids.push_back(tmp_rank);
            i = pos + pattern.size() - 1;
        }
    }

    return 0;
}

int HcclTest::get_mpi_proc()
{
    // 获取当前进程在所属进程组的编号
    MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

    ACLCHECK(aclrtGetDeviceCount(&dev_count));

    // 入参没有-p参数，npus默认值为device count
    if (npus_flag == 0) {
        npus = dev_count;
    }

    rank_id = proc_rank;
    rank_size = proc_size;

    const char *devs = getenv("HCCL_TEST_USE_DEVS");
    if (devs != NULL) {
        std::vector<int> dev_ids;
        int ret = getAviDevs(devs, dev_ids);
        sort(dev_ids.begin(), dev_ids.end());

        int local_rank;
        local_rank = proc_rank % dev_ids.size();
        for (int i = 0; i < dev_ids.size(); i++) {
            if (local_rank == i) {
                dev_id = dev_ids[i];
                break;
            }
        }
    } else {
        dev_id = proc_rank % npus;
    }

    return 0;
}

int HcclTest::hccl_op_base_test()
{
    return 0;
}

int HcclTest::destory_alloc_buf()
{
    return 0;
}

bool HcclTest::IsSupport910_95()
{
    std::string socVersion = aclrtGetSocName();
    if (socVersion.find("Ascend950") != std::string::npos) {
        return true;
    }
    return false;
}

int HcclTest::set_device_sat_mode()
{
    const char *soc_name_ptr = aclrtGetSocName();
    if (soc_name_ptr == nullptr) {
        printf("aclrtGetSocName failed");
        return -1;
    }

    std::string support_soc_name = "Ascend910B";
    if (support_soc_name.length() != strlen(soc_name_ptr) &&
        support_soc_name.compare(0, support_soc_name.length(), soc_name_ptr, 0, support_soc_name.length()) == 0) {
        ACLCHECK(aclrtSetDeviceSatMode(ACL_RT_OVERFLOW_MODE_INFNAN));
    }
    return 0;
}

size_t get_reseved_size(size_t send_bytes, size_t recv_bytes, size_t physicalGranularity)
{
   return (((send_bytes + physicalGranularity - 1) / physicalGranularity) * physicalGranularity) +
                       (((recv_bytes + physicalGranularity - 1) / physicalGranularity) * physicalGranularity);
}

int HcclTest::start_test()
{
    int ret = 0;
    size_t reserve_mem = 0;
    auto diff = 0L;
    auto begin = system_clock::now();
    if (enable_zero_copy)
    {
        // 获取send recv内存大小
        data->data_size = data->min_bytes;
        if (!(stepbytes_flag && !data->step_bytes)) {
            for (; data->data_size <= data->max_bytes;
                 (data->step_factor <= 1.0 ? data->data_size += data->step_bytes
                                           : data->data_size *= data->step_factor)) {
            }
        }

        size_t send_bytes = 0;
        size_t recv_bytes = 0;
        // 设置内存对齐大小 默认对齐128字节，开启zero_copy，默认对齐2M
        prop = {aclrtMemHandleType::ACL_MEM_HANDLE_TYPE_NONE,
            aclrtMemAllocationType::ACL_MEM_ALLOCATION_TYPE_PINNED,
            use_ddr_mem(),
            {dev_id, aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_DEVICE},
            0};
        auto status =
            aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &physicalGranularity);
        if (status != ACL_SUCCESS) {
            physicalGranularity = 2 * 1024 * 1024;
        }
        get_buff_size(send_bytes, recv_bytes);
        reserve_mem = get_reseved_size(send_bytes, recv_bytes, physicalGranularity);
        HCCLCHECK(static_cast<HcclResult>(check_alloc_memory_size(reserve_mem)));
        status = aclrtReserveMemAddress(&vir_ptr, reserve_mem, 0, NULL, 1);
        if (status != ACL_SUCCESS) {
            printf("[%s][%d] aclrtReserveMemAddress failed.\n", __FUNCTION__, __LINE__);
            goto error_device_init;
        }
    }

    begin = system_clock::now();
    ret = init_hcclComm();
    if (ret != 0) {
        goto error_reserve_memAddress;
    }

    diff = duration_cast<microseconds>(system_clock::now() - begin).count();
    HCCL_TEST_LOG("init_hcclComm success, take time[%lld us].\n", diff);

    if (enable_zero_copy) {
        auto status = HcclCommSetMemoryRange(hccl_comm, vir_ptr, reserve_mem, 0, 0);
        if (status != HCCL_SUCCESS) {
            printf("[%s][%d] HcclCommSetMemoryRange failed.\n", __FUNCTION__, __LINE__);
            goto error_reserve_lpcMemory;
        }
    }
    ret = opbase_test_by_data_size();

error_reserve_lpcMemory:
    if (enable_zero_copy) {
        ret = HcclCommUnsetMemoryRange(hccl_comm, vir_ptr);
    }
error_destory_comm:
    ret = HcclCommDestroy(hccl_comm);
error_reserve_memAddress:
    if (enable_zero_copy) {
        ret = aclrtReleaseMemAddress(vir_ptr);
    }
error_device_init:
    ret = destory_hcclComm();
    return ret;
}

int HcclTest::device_init()
{
    // 设备资源初始化
    ACLCHECK(aclInit(NULL));
    // 指定集合通信操作使用的设备
    ACLCHECK(aclrtSetDevice(dev_id));
    // 关闭溢出检测
    int ret = set_device_sat_mode();
    if (ret != 0) {
        printf("set_device_sat_mode execute failed, Detailed logs are stored at the default path: /root/ascend/log/\n");
        return ret;
    }
    ACLCHECK(aclrtCreateEvent(&start_event));
    ACLCHECK(aclrtCreateEvent(&end_event));
    // 创建任务stream
    ACLCHECK(aclrtCreateStream(&stream));
    if (only_device_exec_time) {
        ACLCHECK(aclrtCreateStream(&sync_stream));
        ACLCHECK(aclrtCreateEventWithFlag(&sync_event, ACL_EVENT_SYNC));
    }
    // 设置遇错即停
    ACLCHECK(aclrtSetStreamFailureMode(stream, 1));
    return 0;
}

int HcclTest::init_hcclComm_without_nslb()
{
    if (accelerator_config > 0) {
        HcclCommConfig config = {0};
        HcclCommConfigInit(&config);
        config.hcclWorldRankID = rank_id;
        config.hcclOpExpansionMode = accelerator_config;
        HCCLCHECK(HcclCommInitRootInfoConfig(rank_size, &comm_id, rank_id, &config, &hccl_comm));
    } else {
        HCCLCHECK(HcclCommInitRootInfo(rank_size, &comm_id, rank_id, &hccl_comm));
    }
    return 0;
}

int HcclTest::init_hcclComm()
{
    // 将root_info广播到通信域内的其他rank
    MPI_Request request;
    MPI_Status status;
    // 在root_rank获取root_info
    if (rank_id == root_rank) {
        printf("the minbytes is %llu, maxbytes is %llu, iters is %d, warmup_iters is %d\n",
            data->min_bytes,
            data->max_bytes,
            iters,
            warmup_iters);
        HcclResult getRootInfo = HcclGetRootInfo(&comm_id);
        if (getRootInfo == HCCL_SUCCESS) {
            // 将root_info广播到通信域内的其他rank
            MPI_Ibcast(&comm_id, HCCL_ROOT_INFO_BYTES, MPI_CHAR, root_rank, MPI_COMM_WORLD, &request);
        } else {
            // 通知其他rank获取root_info失败
            char send_str[] = "invalid";
            MPI_Ibcast(send_str, HCCL_ROOT_INFO_BYTES, MPI_CHAR, root_rank, MPI_COMM_WORLD, &request);
        }
        MPI_Wait(&request, &status);
        if (getRootInfo != HCCL_SUCCESS) {
            printf("Process %d HcclGetRootInfo failed, notify other process to exit\n", root_rank);
            return getRootInfo;
        }
    } else {
        // 非根进程接收数据
        MPI_Ibcast(&comm_id, HCCL_ROOT_INFO_BYTES, MPI_CHAR, root_rank, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, &status);
        if (strcmp(reinterpret_cast<const char *>(&comm_id), "invalid") == 0) {
            // 检测到无效数据
            printf("Process %d received invalid data from root process %d\n", rank_id, root_rank);
            return -1;
        }
    }
    // 初始化集合通信域
    if (nslb_flag == 1) {
        // HCCL_COMM_CONFIG_RESERVED 字段小于7，代表当前版本未适配DP场景
        if (HcclGetCommConfigCapability() < NSLBDP_SUPPORT_VERSION) {
            printf("The current version of hccl_test does not support the nslb-dp capability.\n");
            HCCLCHECK(HcclCommInitRootInfo(rank_size, &comm_id, rank_id, &hccl_comm));
            return 0;
        }
        // 配置nslb使能的场景 == 1 才生效
        HcclCommConfig config= { 0 };
        HcclCommConfigInit(&config);
        const char *ip = comm_id.internal;
        // 当前port字段目的只是提供一个拼接的id， 不具有实际意义
        uint32_t port = 11;
        uint32_t ipaddress = (uint32_t)inet_addr(ip);
        config.hcclWorldRankID = rank_id;
        config.hcclJobID = ((uint64_t)port << NSLBDP_PORT_OFFSET) | (uint64_t)ipaddress;
        config.hcclOpExpansionMode = accelerator_config;
        HCCLCHECK(HcclCommInitRootInfoConfig(rank_size, &comm_id, rank_id, &config, &hccl_comm));
    } else {
        init_hcclComm_without_nslb();
    }

    return 0;
}

int HcclTest::opbase_test_by_data_size()
{
    int ret = 0;
    CommSymWindow sym_win;
    HCCLCHECK(static_cast<HcclResult>(register_symmetric_memory(sym_win)));
    for (data->data_size = data->min_bytes; data->data_size <= data->max_bytes;
         (data->step_factor <= 1.0 ? data->data_size += data->step_bytes : data->data_size *= data->step_factor)) {
        size_t send_bytes = 0;
        size_t recv_bytes = 0;
        get_buff_size(send_bytes, recv_bytes);
        HCCLCHECK(static_cast<HcclResult>(prepare_zero_copy(send_bytes, recv_bytes)));
        HCCLCHECK(static_cast<HcclResult>(alloc_hccl_send_recv_buffer(send_buff, send_bytes, recv_buff, recv_bytes)));
        ret = hccl_op_base_test();
        HCCLCHECK(static_cast<HcclResult>(free_send_recv_buff_and_disable_local_buffer()));
        if (ret != 0) {
            printf("hccl_op_base execute failed, Detailed logs are stored at the default path: /root/ascend/log/\n");
            break;
        }
    }
    if (enable_zero_copy) {
        for (auto &&mem_handle : phy_alloc_mem_handle) {
            HCCLCHECK(HcclCommDeactivateCommMemory(hccl_comm, mem_handle.first));
            ACLCHECK(aclrtUnmapMem(mem_handle.first));
            ACLCHECK(aclrtFreePhysical(mem_handle.second));
        }
        phy_alloc_mem_handle.clear();
    }
    HCCLCHECK(static_cast<HcclResult>(deregister_symmetric_memory(sym_win)));
    return ret;
}

int HcclTest::destory_hcclComm()
{
    // 销毁任务流
    if (only_device_exec_time) {
        ACLCHECK(aclrtDestroyStream(sync_stream));
        ACLCHECK(aclrtDestroyEvent(sync_event));
    }
    ACLCHECK(aclrtDestroyStream(stream));
    ACLCHECK(aclrtDestroyEvent(start_event));
    ACLCHECK(aclrtDestroyEvent(end_event));
    // 重置设备
    ACLCHECK(aclrtResetDevice(dev_id));
    return 0;
}

void HcclTest::get_buff_size(size_t &send_bytes, size_t &recv_bytes)
{
    init_data_count();
    malloc_kSize = init_malloc_Ksize_by_data();
    init_send_recv_size_by_data(send_bytes, recv_bytes);
    return;
}

int HcclTest::prepare_zero_copy(const size_t &send_bytes, const size_t &recv_bytes)
{
    if (!enable_zero_copy || ((send_bytes + recv_bytes) == 0)) {
        return HCCL_SUCCESS;
    }
    static size_t current_alloc_phy_mem_byte = 0;
    auto malloc_mem = get_reseved_size(send_bytes, recv_bytes, physicalGranularity);
    if (current_alloc_phy_mem_byte >= malloc_mem) {
        return HCCL_SUCCESS;
    }
    int status = 0;
    aclrtDrvMemHandle mem_handle;
    status = aclrtMallocPhysical(&mem_handle, malloc_mem - current_alloc_phy_mem_byte, &prop, 0);
    if (status != ACL_SUCCESS) {
        printf("[%s][%d] aclrtMallocPhysical failed.\n", __FUNCTION__, __LINE__);
        goto enableerr0;
    }
    status = aclrtMapMem(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(vir_ptr) + current_alloc_phy_mem_byte),
        malloc_mem - current_alloc_phy_mem_byte,
        0,
        mem_handle,
        0);
    if (status != ACL_SUCCESS) {
        printf("[%s][%d] aclrtMapMem failed.\n", __FUNCTION__, __LINE__);
        goto enableerr0;
    }
    status = HcclCommActivateCommMemory(hccl_comm,
        reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(vir_ptr) + current_alloc_phy_mem_byte),
        malloc_mem - current_alloc_phy_mem_byte,
        0,
        mem_handle,
        0);
    if (status != 0) {
        printf("[%s][%d] HcclCommValidMemory failed.\n", __FUNCTION__, __LINE__);
        goto enableerr0;
    }
    phy_alloc_mem_handle.push_back(std::make_pair(
        reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(vir_ptr) + current_alloc_phy_mem_byte), mem_handle));
    current_alloc_phy_mem_byte = malloc_mem;
    return 0;

enableerr0:
    for (auto &&mem_handle : phy_alloc_mem_handle) {
        HCCLCHECK(HcclCommDeactivateCommMemory(hccl_comm, mem_handle.first));
        ACLCHECK(aclrtUnmapMem(mem_handle.first));
        ACLCHECK(aclrtFreePhysical(mem_handle.second));
    }
    phy_alloc_mem_handle.clear();
    return status;
}

int HcclTest::alloc_hccl_send_recv_buffer(
    void *&send_buff, const size_t &send_bytes, void *&recv_buff, const size_t &recv_bytes)
{
    if (enable_zero_copy) {
        // 2M对齐
        send_buff = vir_ptr;
        recv_buff = reinterpret_cast<void *>(
            reinterpret_cast<uintptr_t>(vir_ptr) +
            (((send_bytes + physicalGranularity - 1) / physicalGranularity) * physicalGranularity));
    } else if(enable_symmetric_memory) {
        send_buff = vir_ptr;
        recv_buff = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(vir_ptr) + send_bytes);
    } else {
        // 申请集合通信操作的内存
        if (send_bytes) {
            ACLCHECK(aclrtMalloc((void **)&send_buff, send_bytes, ACL_MEM_MALLOC_HUGE_ONLY));
        }
        if (recv_bytes) {
            ACLCHECK(aclrtMalloc((void **)&recv_buff, recv_bytes, ACL_MEM_MALLOC_HUGE_ONLY));
        }
    }
    return 0;
}

int HcclTest::free_send_recv_buff_and_disable_local_buffer()
{
    if (!enable_zero_copy && !enable_symmetric_memory) {
        // 申请集合通信操作的内存
        if (send_buff) {
            ACLCHECK(aclrtFree(send_buff));
        }
        if (recv_buff) {
            ACLCHECK(aclrtFree(recv_buff));
        }
    }

    int ret = destory_alloc_buf();
    if (ret != 0) {
        printf("hccl_op_base destory_alloc_buf failed, ret[%d]", ret);
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}

int HcclTest::hccl_mem_alloc(size_t size, void **ptr, aclrtDrvMemHandle *handle)
{
    if (ptr == nullptr || size == 0 || handle == nullptr) {
        printf("[%s][%d] Invalid parameter: ptr[%p], size[%zu].\n", __FUNCTION__, __LINE__, ptr, size);
        return HCCL_E_PARA;
    }

    int32_t deviceId;
    ACLCHECK(aclrtGetDevice(&deviceId));
    aclrtPhysicalMemProp prop;
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_HBM_MEM_HUGE;
    prop.location.id = deviceId;
    prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    prop.reserve = 0;

    size_t allocSize = size;
    size_t granularity = 0;
    aclError ret = aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED, &granularity);
    if (ret != ACL_SUCCESS || granularity == 0) {
        printf("[%s][%d] GetAllocationGranularity failed.\n", __FUNCTION__, __LINE__);
        return HCCL_E_RUNTIME;
    }
    allocSize = (allocSize + granularity - 1) / granularity * granularity;

    ret = aclrtReserveMemAddress(ptr, allocSize, 0, nullptr, 1);
    if (ret != ACL_SUCCESS) {
        printf("[%s][%d] aclrtReserveMemAddress failed.\n", __FUNCTION__, __LINE__);
        return HCCL_E_RUNTIME;
    }
    void *virPtr = *ptr;
    ret = aclrtMallocPhysical(handle, allocSize, &prop, 0);
    if(ret != ACL_SUCCESS) {
        printf("[%s][%d] aclrtMallocPhysical failed.\n", __FUNCTION__, __LINE__);
        aclrtReleaseMemAddress(virPtr);
        return HCCL_E_RUNTIME;
    }
    ret = aclrtMapMem(virPtr, allocSize, 0, *handle, 0);
    if(ret != ACL_SUCCESS) {
        printf("[%s][%d] aclrtMapMem failed.\n", __FUNCTION__, __LINE__);
        aclrtFreePhysical(*handle);
        aclrtReleaseMemAddress(virPtr);
        return HCCL_E_RUNTIME;
    }

    return HCCL_SUCCESS;
}

int HcclTest::hccl_mem_free(void *ptr, aclrtDrvMemHandle &handle)
{
    if (ptr == nullptr || handle == nullptr) {
        return HCCL_SUCCESS;
    }
    aclError ret = ACL_SUCCESS;
    ret = aclrtUnmapMem(ptr);
    if (ret != ACL_SUCCESS) {
        printf("[%s][%d] aclrtUnmapMem failed.\n", __FUNCTION__, __LINE__);
        return HCCL_E_RUNTIME;
    }
    ret = aclrtFreePhysical(handle);
    if (ret != ACL_SUCCESS) {
        printf("[%s][%d] aclrtFreePhysical failed.\n", __FUNCTION__, __LINE__);
        return HCCL_E_RUNTIME;
    }
    ret = aclrtReleaseMemAddress(ptr);
    if (ret != ACL_SUCCESS) {
        printf("[%s][%d] aclrtReleaseMemAddress failed.\n", __FUNCTION__, __LINE__);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

int HcclTest::register_symmetric_memory(CommSymWindow &sym_win)
{
    if (!enable_symmetric_memory) {
        return HCCL_SUCCESS;
    }
    // 获取send recv内存大小
    data->data_size = data->min_bytes;
    if (!(stepbytes_flag && !data->step_bytes)) {
        for (; data->data_size < data->max_bytes;
                (data->step_factor <= 1.0 ? data->data_size += data->step_bytes
                                        : data->data_size *= data->step_factor)) {
        }
    }
    size_t send_bytes = 0;
    size_t recv_bytes = 0;
    get_buff_size(send_bytes, recv_bytes);
    size_t reserve_size = send_bytes + recv_bytes;
    HCCLCHECK(static_cast<HcclResult>(hccl_mem_alloc(reserve_size, &vir_ptr, &symmetric_handle)));
    HCCLCHECK(HcclCommSymWinRegister(hccl_comm, vir_ptr, reserve_size, &sym_win, HCCL_WIN_COLL_SYMMETRIC));
    return 0;
}

int HcclTest::deregister_symmetric_memory(CommSymWindow &sym_win)
{
    if (enable_symmetric_memory) {
        HCCLCHECK(HcclCommSymWinDeregister(sym_win));
        MPI_Barrier(MPI_COMM_WORLD);
        HCCLCHECK(static_cast<HcclResult>(hccl_mem_free(vir_ptr, symmetric_handle)));
    }
    return 0;
}
// 由于ccu在大数据量场景下开启-t可能会卡死，-t参数对于ccu alltoallvc以外数据量大于等于 16MB的算子 -t不生效
// -t参数对于ccu alltoallvc 算子数据量大于等于 128MB的场景 -t不生效
aclError HcclTest::start_profile_device_time_if_needed(size_t data_size)
{
    bool isCcuSched = (accelerator_config == 0 || accelerator_config == 6) && IsSupport910_95();
    if (only_device_exec_time && !(isCcuSched && data->data_size >= data_size)) {
        ACLCHECK(aclrtStreamWaitEvent(stream, sync_event));
        ACLCHECK(aclrtResetEvent(sync_event, stream));
    }
    return ACL_SUCCESS;
}

aclError HcclTest::end_profile_device_time_if_needed(size_t data_size)
{
    bool isCcuSched = (accelerator_config == 0 || accelerator_config == 6) && IsSupport910_95();
    if (only_device_exec_time && !(isCcuSched && data->data_size >= data_size)) {
        int sleepTime = 50 + warmup_iters * 2 + iters * 2;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
        ACLCHECK(aclrtRecordEvent(sync_event, sync_stream));
    }
    return ACL_SUCCESS;
}

}  // namespace hccl
