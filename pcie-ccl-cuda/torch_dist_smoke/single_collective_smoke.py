import argparse
import datetime as dt
import os
import sys
import time

import torch
import torch.distributed as dist


def log(rank: int, msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] rank{rank}: {msg}", flush=True)


def check(rank: int, name: str, got: torch.Tensor, expected: torch.Tensor) -> None:
    expected = expected.to(device=got.device, dtype=got.dtype)
    if not torch.allclose(got, expected):
        log(rank, f"{name} mismatch got={got.detach().cpu()} expected={expected.detach().cpu()}")
        raise RuntimeError(f"{name} mismatch on rank {rank}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("op", choices=("allreduce", "allgather", "broadcast", "reducescatter"))
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--timeout-seconds", type=int, default=45)
    parser.add_argument("--elements", type=int, default=1024)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    log(rank, f"init backend={args.backend} op={args.op} world={world} local_rank={local_rank}")
    dist.init_process_group(
        backend=args.backend,
        timeout=dt.timedelta(seconds=args.timeout_seconds),
        device_id=device if args.backend == "nccl" else None,
    )
    log(rank, "init done")

    if args.op == "allreduce":
        x = torch.full((args.elements,), rank + 1.0, device=device)
        log(rank, f"all_reduce enter n={x.numel()}")
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        expected = torch.full((args.elements,), world * (world + 1) / 2, device=device)
        check(rank, "all_reduce", x, expected)
        log(rank, f"all_reduce done first={x[0].item():.1f}")
    elif args.op == "allgather":
        x = torch.full((args.elements,), rank, dtype=torch.float32, device=device)
        y = torch.empty((world * args.elements,), dtype=torch.float32, device=device)
        log(rank, f"all_gather_into_tensor enter n={x.numel()}")
        dist.all_gather_into_tensor(y, x)
        torch.cuda.synchronize(device)
        expected = torch.cat([torch.full((args.elements,), r) for r in range(world)])
        check(rank, "all_gather_into_tensor", y, expected)
        log(rank, f"all_gather_into_tensor done sum={y.sum().item():.1f}")
    elif args.op == "broadcast":
        x = torch.full((args.elements,), rank + 10.0, device=device)
        log(rank, f"broadcast enter n={x.numel()} root=0")
        dist.broadcast(x, src=0)
        torch.cuda.synchronize(device)
        check(rank, "broadcast", x, torch.full((args.elements,), 10.0, device=device))
        log(rank, f"broadcast done first={x[0].item():.1f}")
    else:
        x = torch.cat([torch.full((args.elements,), rank * 10.0 + r, dtype=torch.float32) for r in range(world)]).to(device)
        y = torch.empty((args.elements,), device=device)
        log(rank, f"reduce_scatter_tensor enter n={y.numel()}")
        dist.reduce_scatter_tensor(y, x, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        expected_value = sum(src_rank * 10.0 + rank for src_rank in range(world))
        check(rank, "reduce_scatter_tensor", y, torch.full((args.elements,), expected_value, device=device))
        log(rank, f"reduce_scatter_tensor done first={y[0].item():.1f}")

    log(rank, "barrier enter")
    dist.barrier(device_ids=[local_rank] if args.backend == "nccl" else None)
    log(rank, "barrier done")
    dist.destroy_process_group()
    log(rank, "destroy done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
