# torch_npu HCCL integration test

This test validates the CANN `torch_npu` process-group path used by Megatron/MindSpeed.

## Symbol test

```bash
ADAPTIVE_CCL_ROOT=/workspace/Adaptive-CCL \
  tests/megatron_adapt/test_hccl_symbols.sh
```

It checks the exact uppercase CANN exports, including Config initialization and
`HcclCreateSubCommConfig`.

The runner also compiles and executes `test_hccl_pccl_dtype_mapping.cc`, which
checks the explicit HCCL-to-PCCL typed mapping and the public Typed PCCL
collective symbols.

## Runtime test

Build the HCCL hook first, then run on one process per NPU:

```bash
ADAPTIVE_CCL_ROOT=/workspace/Adaptive-CCL \
NPUS_PER_NODE=8 \
RUN_TORCH_NPU_TEST=1 \
  tests/megatron_adapt/run.sh
```

The runtime test uses small BF16 buffers and disables PCCL by default. Set
`AMPCCL_ENABLE_PCIE=1` to include PCCL initialization and exercise the Typed
PCCL path. Depending on the CANN
and torch_npu version, `new_group` may create groups through
`HcclCreateSubCommConfig` or through repeated `HcclCommInitRootInfoConfig`.
The test accepts both paths and checks group-domain registration plus all three
collective dispatch paths. With PCIe enabled, it also rejects the old untyped
4-byte-only guard message.

If the run is only intended to validate collective dtype routing, set
`TEST_SUBGROUPS=0` to avoid creating additional subgroup communicators.

## Feedback OpKey isolation test

This test alternates ReduceScatter and AllGather on the same 8-rank HCCL
domain. It checks that the first AllGather starts from its own default
feedback state instead of inheriting the ReduceScatter state. The runtime log
is written to `/tmp` by default:

```bash
ADAPTIVE_CCL_ROOT=/workspace/Adaptive-CCL \
  tests/megatron_adapt/run_feedback_opkey_isolation.sh
```

Set `AMPCCL_ENABLE_PCIE=0` when only the controller/SHM behavior is being
validated and PCIe resources are unavailable. The test uses a small BF16
buffer by default; set `TEST_BYTES_PER_RANK` to increase it.

For loader diagnosis, set `AMPCCL_INTERPOSE_DLSYM=0` to disable the explicit
`dlsym(libhccl.so, ...)` bridge. The default is enabled because torch_npu loads
HCCL entry points by handle-local `dlsym`.
