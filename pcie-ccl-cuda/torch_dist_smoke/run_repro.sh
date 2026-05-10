#!/usr/bin/env bash
set -uo pipefail
ROOT=/workspace/pcieccl-noctx
export CUDA_VISIBLE_DEVICES=0,2,4,6
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export LD_PRELOAD=$ROOT/build/libpccl.so
export LD_LIBRARY_PATH=$ROOT/build:${LD_LIBRARY_PATH:-}
export PCCL_IR_PATH=$ROOT/ir
export PCCL_EXECUTOR_TIMEOUT_MS=600000
export PCCL_LOG_LEVEL=${PCCL_LOG_LEVEL:-2}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29655
export ELEMS=${ELEMS:-8388608}
export ITERS=${ITERS:-20}
SCRIPT=${SCRIPT:-$ROOT/torch_dist_smoke/deadlock_repro.py}
python3 -m torch.distributed.run --nproc_per_node=4 --master_port=${MASTER_PORT} "$SCRIPT" "$@"
