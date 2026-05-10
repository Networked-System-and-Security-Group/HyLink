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


def assert_close(rank: int, name: str, got: torch.Tensor, expected: torch.Tensor) -> None:
    expected = expected.to(dtype=got.dtype)
    if not torch.allclose(got.cpu(), expected.cpu()):
        log(rank, f"{name} mismatch got={got.detach().cpu()} expected={expected.detach().cpu()}")
        raise RuntimeError(f"{name} mismatch on rank {rank}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--elements", type=int, default=1024)
    parser.add_argument("--large-elements", type=int, default=1_048_576)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    log(rank, f"init backend={args.backend} world={world} local_rank={local_rank}")
    dist.init_process_group(
        backend=args.backend,
        timeout=dt.timedelta(seconds=args.timeout_seconds),
        device_id=device if args.backend == "nccl" else None,
    )
    log(rank, "init done")

    small = torch.tensor([float(rank + 1)], device=device)
    log(rank, "all_reduce scalar enter")
    dist.all_reduce(small, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    assert_close(rank, "all_reduce scalar", small, torch.tensor([world * (world + 1) / 2], device=device))
    log(rank, f"all_reduce scalar done value={small.item():.1f}")

    x = torch.full((args.elements,), rank + 1.0, device=device)
    log(rank, f"all_reduce vector enter n={x.numel()}")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    assert_close(rank, "all_reduce vector", x[:8], torch.full((8,), world * (world + 1) / 2, device=device))
    log(rank, f"all_reduce vector done first={x[0].item():.1f}")

    gather_in = torch.full((args.elements,), rank, dtype=torch.float32, device=device)
    gather_out = torch.empty((world * args.elements,), dtype=torch.float32, device=device)
    log(rank, f"all_gather_into_tensor enter n={gather_in.numel()}")
    dist.all_gather_into_tensor(gather_out, gather_in)
    torch.cuda.synchronize(device)
    expected = torch.cat([torch.full((args.elements,), r, dtype=torch.float32) for r in range(world)])
    assert_close(rank, "all_gather_into_tensor", gather_out, expected.to(device))
    log(rank, f"all_gather_into_tensor done sum={gather_out.sum().item():.1f}")

    rs_in = torch.cat([torch.full((args.elements,), rank * 10 + r, dtype=torch.float32) for r in range(world)]).to(device)
    rs_out = torch.empty((args.elements,), dtype=torch.float32, device=device)
    log(rank, f"reduce_scatter_tensor enter n={rs_out.numel()}")
    dist.reduce_scatter_tensor(rs_out, rs_in, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    rs_expected_value = sum(src_rank * 10 + rank for src_rank in range(world))
    assert_close(rank, "reduce_scatter_tensor", rs_out[:8], torch.full((8,), rs_expected_value, device=device))
    log(rank, f"reduce_scatter_tensor done first={rs_out[0].item():.1f}")

    bcast = torch.tensor([float(rank + 100)], device=device)
    log(rank, "broadcast enter root=0")
    dist.broadcast(bcast, src=0)
    torch.cuda.synchronize(device)
    assert_close(rank, "broadcast", bcast, torch.tensor([100.0], device=device))
    log(rank, f"broadcast done value={bcast.item():.1f}")

    large = torch.full((args.large_elements,), rank + 1.0, dtype=torch.float32, device=device)
    log(rank, f"all_reduce large enter bytes={large.numel() * large.element_size()}")
    dist.all_reduce(large, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    assert_close(rank, "all_reduce large", large[:8], torch.full((8,), world * (world + 1) / 2, device=device))
    log(rank, f"all_reduce large done first={large[0].item():.1f}")

    log(rank, "barrier enter")
    dist.barrier(device_ids=[local_rank] if args.backend == "nccl" else None)
    log(rank, "barrier done")
    dist.destroy_process_group()
    log(rank, "destroy done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
