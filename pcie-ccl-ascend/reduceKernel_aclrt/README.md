# PCCL AICore Reduce

This component provides PCCL's explicit AICore implementation of local device-to-device elementwise reduce. It is not an `LD_PRELOAD` library and it does not define `aclrtReduceAsync`; HCCL and other runtime users keep their original runtime behavior.

## Supported AICore requests

`pcclAicoreReduceAsync` accepts device pointers on the current device and supports these combinations:

- `ACL_FLOAT16`, `ACL_BF16`, and `ACL_FLOAT`
- `ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM`, `ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX`, and `ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN`
- Nonzero byte counts divisible by the selected element size

All other requests return `ACL_ERROR_API_NOT_SUPPORT`. BF16 is converted to FP32 for the vector Add/Max/Min operation and converted back to BF16; the dav-2201 vector binary intrinsics do not accept BF16 directly. `devLaunchD2DReduce()` in PCCL is the single caller and falls back to native `aclrtReduceAsync` after emitting a WARN log. A selected AICore launch failure is returned directly and never silently redirected to SDMA.

For requests below 64 KiB the launcher uses one AIV; larger requests use two AIVs. The kernel models two L2 tiles and rotates each tile's q/r remainder across the active AIVs, so repeated L2 tails do not always execute on the same core. The final non-32-byte suffix is padded and follows the same disjoint, rotated ownership rule.

The direct launcher has no runtime-provided L2 slice id, so the two-slice loop is an explicit software tiling policy. Timing at a few microseconds is sensitive to launch and queueing overhead; device correctness and aggregate repeated-sample statistics should be used when evaluating this policy.

## Build as part of PCCL

The top-level build enables this component by default:

```bash
cmake -S . -B build -DPCCL_BUILD_TESTS=ON -DPCCL_BUILD_HCCL_TEST=OFF
cmake --build build -j
ctest --test-dir build -R aicore_reduce_device_correctness --output-on-failure
```

Use `-DPCCL_ENABLE_AICORE_REDUCE=OFF` only when building without the AscendC toolchain. In that configuration PCCL retains the original native reduce behavior.

## Standalone component build

The component can also be built and contract-tested independently:

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
cmake -S reduceKernel_aclrt -B /tmp/pccl-aicore-reduce-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DASCEND_PATH=/usr/local/Ascend/ascend-toolkit/latest
cmake --build /tmp/pccl-aicore-reduce-build -j
ctest --test-dir /tmp/pccl-aicore-reduce-build --output-on-failure
```

The artifact exports `pcclAicoreReduceAsync`; `check_reduce_symbols.sh` verifies that it does not export `aclrtReduceAsync`.
