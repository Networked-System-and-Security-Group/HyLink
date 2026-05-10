import argparse
import datetime as dt
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


def log(rank: int, msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] rank{rank}: {msg}", flush=True)


class TinyMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Linear(width, 16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--width", type=int, default=512)
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

    torch.manual_seed(1234)
    model = TinyMLP(args.width).to(device)
    ddp = DDP(model, device_ids=[local_rank], output_device=local_rank)
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-3)

    base_x = torch.randn(args.batch, args.width, device=device)
    base_y = torch.randint(0, 16, (args.batch,), device=device)

    losses = []
    for step in range(args.steps):
        x = base_x + float(rank) * 0.01 + float(step) * 0.001
        y = (base_y + rank + step) % 16
        opt.zero_grad(set_to_none=True)
        log(rank, f"step {step} forward enter")
        out = ddp(x)
        loss = F.cross_entropy(out, y)
        log(rank, f"step {step} backward enter loss={loss.item():.6f}")
        loss.backward()
        log(rank, f"step {step} optimizer enter")
        opt.step()
        torch.cuda.synchronize(device)
        losses.append(loss.detach())
        log(rank, f"step {step} done")

    loss_tensor = torch.stack(losses).mean()
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    loss_tensor /= world
    torch.cuda.synchronize(device)
    log(rank, f"mean_loss_all_ranks={loss_tensor.item():.6f}")

    log(rank, "barrier enter")
    dist.barrier(device_ids=[local_rank] if args.backend == "nccl" else None)
    log(rank, "barrier done")
    dist.destroy_process_group()
    log(rank, "destroy done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
