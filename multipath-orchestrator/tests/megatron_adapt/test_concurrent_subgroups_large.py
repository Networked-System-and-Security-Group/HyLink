#!/usr/bin/env python3
"""Run large concurrent ReduceScatter/AllGather on two disjoint 4-rank groups.

The defaults intentionally reuse the buffer sizes observed in the Qwen3
training log.  Ranks 0..3 execute group A while ranks 4..7 execute group B;
the two groups therefore enter the same collective concurrently without any
single rank belonging to both groups.
"""

import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


WORLD_SIZE = 8
GROUP_SIZE = 4
DEFAULT_REDUCE_SCATTER_RECV_BYTES = 4095369728
DEFAULT_ALLGATHER_SEND_BYTES = 2047684864


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def selected_dtype() -> torch.dtype:
    name = os.environ.get("TEST_DTYPE", "bf16").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError("TEST_DTYPE must be bf16 or fp32")


def check_finite(tensor: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"{label}: result contains NaN or Inf")


def check_constant(tensor: torch.Tensor, expected: int, label: str) -> None:
    check_finite(tensor, label)
    if not bool(torch.all(tensor == expected).item()):
        flat = tensor.flatten()
        scan_elems = 16 * 1024 * 1024
        first_bad = -1
        first_value = None
        for start in range(0, flat.numel(), scan_elems):
            chunk = flat[start : start + scan_elems]
            mismatch = chunk != expected
            if bool(mismatch.any().item()):
                local_index = int(torch.nonzero(mismatch, as_tuple=False)[0].item())
                first_bad = start + local_index
                first_value = float(chunk[local_index].item())
                break
        raise RuntimeError(
            f"{label}: first_bad={first_bad}/{flat.numel()} "
            f"fraction={first_bad / flat.numel():.9f} value={first_value} expected={expected}"
        )


def run_reduce_scatter(group, group_ranks, rank: int, recv_bytes: int) -> None:
    dtype = selected_dtype()
    element_size = torch.tensor([], dtype=dtype).element_size()
    if recv_bytes % element_size != 0:
        raise ValueError("TEST_REDUCE_SCATTER_RECV_BYTES must align to bfloat16")

    recv_count = recv_bytes // element_size
    send = torch.empty(GROUP_SIZE * recv_count, dtype=dtype, device="npu")
    send.fill_(rank + 1)
    recv = torch.empty(recv_count, dtype=dtype, device="npu")

    dist.reduce_scatter_tensor(recv, send, group=group)
    torch.npu.synchronize()

    expected = sum(group_ranks) + len(group_ranks)
    check_constant(recv, expected, f"ReduceScatter group={group_ranks}")
    print(f"[CONCURRENT_TEST][rank={rank}] ReduceScatter group={group_ranks} PASS", flush=True)
    del send, recv
    torch.npu.empty_cache()


def run_all_gather(group, group_ranks, rank: int, send_bytes: int) -> None:
    dtype = selected_dtype()
    element_size = torch.tensor([], dtype=dtype).element_size()
    if send_bytes % element_size != 0:
        raise ValueError("TEST_ALLGATHER_SEND_BYTES must align to bfloat16")

    send_count = send_bytes // element_size
    gathered = torch.empty(GROUP_SIZE * send_count, dtype=dtype, device="npu")
    if os.environ.get("TEST_ALLGATHER_IN_PLACE", "0") == "1":
        gathered.fill_(float("nan"))
        local_rank = group_ranks.index(rank)
        send = gathered[local_rank * send_count : (local_rank + 1) * send_count]
    else:
        send = torch.empty(send_count, dtype=dtype, device="npu")
    send.fill_(rank + 1)

    dist.all_gather_into_tensor(gathered, send, group=group)
    torch.npu.synchronize()

    check_finite(gathered, f"AllGather group={group_ranks}")
    for index, source in enumerate(group_ranks):
        segment = gathered[index * send_count : (index + 1) * send_count]
        check_constant(segment, source + 1, f"AllGather group={group_ranks} source={source}")
    print(f"[CONCURRENT_TEST][rank={rank}] AllGather group={group_ranks} PASS", flush=True)
    del send, gathered
    torch.npu.empty_cache()


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ.get("WORLD_SIZE", WORLD_SIZE))
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"This test requires WORLD_SIZE={WORLD_SIZE}, got {world_size}")

    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    dist.barrier()

    group_a_ranks = list(range(0, GROUP_SIZE))
    group_b_ranks = list(range(GROUP_SIZE, WORLD_SIZE))
    group_a = dist.new_group(ranks=group_a_ranks, backend="hccl")
    group_b = dist.new_group(ranks=group_b_ranks, backend="hccl")
    dist.barrier()

    group_ranks = group_a_ranks if rank < GROUP_SIZE else group_b_ranks
    group = group_a if rank < GROUP_SIZE else group_b
    mode = os.environ.get("TEST_OP", "both").lower()
    active_group = os.environ.get("TEST_GROUP", "both").lower()
    if active_group not in ("a", "b", "both"):
        raise ValueError("TEST_GROUP must be a, b, or both")
    rank_group = "a" if rank < GROUP_SIZE else "b"
    run_group = active_group == "both" or active_group == rank_group

    for _ in range(env_int("TEST_ITERS", 1)):
        if run_group and mode in ("both", "reducescatter"):
            run_reduce_scatter(
                group,
                group_ranks,
                rank,
                env_int("TEST_REDUCE_SCATTER_RECV_BYTES", DEFAULT_REDUCE_SCATTER_RECV_BYTES),
            )
            dist.barrier()

        if run_group and mode in ("both", "allgather"):
            run_all_gather(
                group,
                group_ranks,
                rank,
                env_int("TEST_ALLGATHER_SEND_BYTES", DEFAULT_ALLGATHER_SEND_BYTES),
            )
            dist.barrier()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
