#!/usr/bin/env bash
set -uo pipefail
ROOT=/workspace/pcieccl-noctx
cd /workspace/megatron
export CUDA_VISIBLE_DEVICES=0,2,4,6
export PYTHONNOUSERSITE=1
# PCCL needs multiple HW connections; the single-connection head-of-line block is
# an architectural incompatibility with its CPU-orchestrated multi-stream design.
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export LD_PRELOAD=$ROOT/build/libpccl.so
export LD_LIBRARY_PATH=$ROOT/build:${LD_LIBRARY_PATH:-}
export PCCL_IR_PATH=$ROOT/ir
export PCCL_EXECUTOR_TIMEOUT_MS=120000
export PCCL_LOG_LEVEL=${PCCL_LOG_LEVEL:-2}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29680}
ITERS=${ITERS:-2}
python3 -m torch.distributed.run --nproc_per_node=4 --master_port=${MASTER_PORT} \
  /workspace/megatron/pretrain_gpt.py \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 1 \
  --num-layers 2 \
  --hidden-size 4096 \
  --ffn-hidden-size 16384 \
  --num-attention-heads 32 \
  --seq-length 4096 \
  --max-position-embeddings 4096 \
  --micro-batch-size 1 \
  --global-batch-size 4 \
  --train-iters ${ITERS} \
  --lr 1e-4 \
  --min-lr 1e-5 \
  --lr-decay-style cosine \
  --lr-decay-iters 100 \
  --weight-decay 0.01 \
  --clip-grad 1.0 \
  --fp16 \
  --tokenizer-type NullTokenizer \
  --vocab-size 8192 \
  --mock-data \
  --sequence-parallel \
  --transformer-impl transformer_engine \
  --distributed-backend nccl \
  "$@"
