#!/usr/bin/env python3
"""Exercise torch_npu HCCL parent and subgroup communicators.

The runner starts one process per NPU. The test intentionally uses modest
buffers; this test validates communicator registration and dispatch, not the
256 MiB performance threshold.
"""

import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # registers the HCCL backend


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])

    def mark(group_index: int, event: str) -> None:
        print(f"[PYTEST][rank={rank}][group={group_index}] {event}", flush=True)

    torch.npu.set_device(local_rank)

    mark(-1, "init_process_group begin")
    dist.init_process_group(backend="hccl")
    mark(-1, "init_process_group end; world barrier begin")
    dist.barrier()
    mark(-1, "world barrier end")

    groups = [list(range(world_size))]
    if world_size >= 4 and os.environ.get("TEST_SUBGROUPS", "1") == "1":
        half = world_size // 2
        groups.extend([list(range(half)), list(range(half, world_size))])

    for group_index, group_ranks in enumerate(groups):
        mark(group_index, f"new_group begin ranks={group_ranks}")
        group = dist.new_group(ranks=group_ranks, backend="hccl")
        mark(group_index, "new_group end; creation barrier begin")
        # All ranks must finish creating a subgroup before any member starts
        # its subgroup collectives. Non-members participate in these world
        # barriers so the next subgroup is created in the same order.
        dist.barrier()
        mark(group_index, "creation barrier end")
        if rank not in group_ranks:
            mark(group_index, "non-member; exit group")
            dist.barrier()
            mark(group_index, "non-member exit barrier end")
            continue

        subgroup_rank = group_ranks.index(rank)
        value = torch.ones(512 * 1024, dtype=torch.bfloat16, device="npu")
        mark(group_index, "all_reduce begin")
        dist.all_reduce(value, group=group)
        mark(group_index, "all_reduce end; all_gather begin")

        gathered = torch.empty(len(group_ranks) * value.numel(), dtype=value.dtype, device="npu")
        dist.all_gather_into_tensor(gathered, value, group=group)
        mark(group_index, "all_gather end; reduce_scatter begin")

        reduced = torch.empty(value.numel(), dtype=value.dtype, device="npu")
        dist.reduce_scatter_tensor(reduced, gathered, group=group)
        assert torch.isfinite(reduced).all().item(), f"invalid result for subgroup rank {subgroup_rank}"
        mark(group_index, "reduce_scatter end; exit barrier begin")

        dist.barrier()
        mark(group_index, "exit barrier end")

    mark(-1, "final world barrier begin")
    dist.barrier()
    mark(-1, "final world barrier end; destroy begin")
    dist.destroy_process_group()
    mark(-1, "destroy end")


if __name__ == "__main__":
    main()
