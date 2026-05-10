#!/usr/bin/env python3
"""Exercise alternating ReduceScatter and AllGather feedback on one domain.

The first ReduceScatter is deliberately completed before the first AllGather.
With a domain-global adaptive algorithm, the AllGather would inherit the
ReduceScatter alpha.  Each OpKey must instead start from the default alpha.
"""

import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


WORLD_SIZE = 8
DEFAULT_BYTES = 256 * 1024 * 1024


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def check_constant(tensor: torch.Tensor, expected: int, label: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"{label}: result contains NaN or Inf")
    if not bool(torch.all(tensor == expected).item()):
        raise RuntimeError(f"{label}: result does not equal {expected}")


def run_reduce_scatter(rank: int, count: int) -> None:
    dtype = torch.bfloat16
    send = torch.full(
        (WORLD_SIZE * count,), rank + 1, dtype=dtype, device="npu"
    )
    reduced = torch.empty(count, dtype=dtype, device="npu")
    dist.reduce_scatter_tensor(reduced, send)
    torch.npu.synchronize()
    check_constant(reduced, sum(range(1, WORLD_SIZE + 1)), "ReduceScatter")
    print(f"[OPKEY_TEST][rank={rank}] ReduceScatter PASS", flush=True)
    del send, reduced


def run_all_gather(rank: int, count: int) -> None:
    dtype = torch.bfloat16
    send = torch.full((count,), rank + 1, dtype=dtype, device="npu")
    gathered = torch.empty(WORLD_SIZE * count, dtype=dtype, device="npu")
    dist.all_gather_into_tensor(gathered, send)
    torch.npu.synchronize()
    for source in range(WORLD_SIZE):
        segment = gathered[source * count : (source + 1) * count]
        check_constant(segment, source + 1, f"AllGather source={source}")
    print(f"[OPKEY_TEST][rank={rank}] AllGather PASS", flush=True)
    del send, gathered


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"This test requires WORLD_SIZE={WORLD_SIZE}, got {world_size}")

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")

    bytes_per_rank = env_int("TEST_BYTES_PER_RANK", DEFAULT_BYTES)
    element_size = torch.tensor([], dtype=torch.bfloat16).element_size()
    if bytes_per_rank % element_size != 0:
        raise ValueError("TEST_BYTES_PER_RANK must align to bfloat16")
    count = bytes_per_rank // element_size

    # The order is intentional: the first AG must not inherit the first RS's
    # updated controller state. Repeat the sequence to exercise cache hits.
    for iteration in range(env_int("TEST_ITERS", 2)):
        run_reduce_scatter(rank, count)
        run_all_gather(rank, count)
        print(f"[OPKEY_TEST][rank={rank}] iteration={iteration} PASS", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
