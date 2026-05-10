#ifndef ACLRT_REDUCE_AICORE_PRELOAD_LAUNCH_H
#define ACLRT_REDUCE_AICORE_PRELOAD_LAUNCH_H

#include <acl/acl_rt.h>

#include <cstdint>

aclError LaunchPcclAicoreReduceFloat16Sum(void* dst, const void* src, uint64_t count_bytes,
                                           aclrtStream stream);
aclError LaunchPcclAicoreReduceFloat16Max(void* dst, const void* src, uint64_t count_bytes,
                                           aclrtStream stream);
aclError LaunchPcclAicoreReduceFloat16Min(void* dst, const void* src, uint64_t count_bytes,
                                           aclrtStream stream);
aclError LaunchPcclAicoreReduceBfloat16Sum(void* dst, const void* src, uint64_t count_bytes,
                                            aclrtStream stream);
aclError LaunchPcclAicoreReduceBfloat16Max(void* dst, const void* src, uint64_t count_bytes,
                                            aclrtStream stream);
aclError LaunchPcclAicoreReduceBfloat16Min(void* dst, const void* src, uint64_t count_bytes,
                                            aclrtStream stream);
aclError LaunchPcclAicoreReduceFloatSum(void* dst, const void* src, uint64_t count_bytes,
                                         aclrtStream stream);
aclError LaunchPcclAicoreReduceFloatMax(void* dst, const void* src, uint64_t count_bytes,
                                         aclrtStream stream);
aclError LaunchPcclAicoreReduceFloatMin(void* dst, const void* src, uint64_t count_bytes,
                                         aclrtStream stream);

#endif  // ACLRT_REDUCE_AICORE_PRELOAD_LAUNCH_H
