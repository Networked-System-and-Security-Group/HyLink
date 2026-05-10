"""Minimal reproducer for the PCCL pageable-D2H driver-lock deadlock.

Pattern: submit an async collective (user stream parks on a StreamWaitValue that
only the PCCL executor thread can satisfy), then immediately do a *pageable* D2H
(`.item()`) WITHOUT a full cuda sync first. The blocking pageable copy holds the
driver lock while the executor thread needs it to advance the collective.

Run under PCCL via LD_PRELOAD=.../libpccl.so, CUDA_DEVICE_MAX_CONNECTIONS=1.
"""
import datetime as dt
import os
import sys
import time

import torch
import torch.distributed as dist


def log(rank, msg):
    print(f"[{time.strftime('%H:%M:%S')}] rank{rank}: {msg}", flush=True)


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(backend="nccl",
                            timeout=dt.timedelta(seconds=30),
                            device_id=device)
    log(rank, "init done")

    n = int(os.environ.get("ELEMS", str(8 * 1024 * 1024)))  # ~32MB fp32
    iters = int(os.environ.get("ITERS", "20"))
    for it in range(iters):
        x = torch.full((n,), rank + 1.0, device=device)
        log(rank, f"iter{it} all_reduce enter")
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        # NO torch.cuda.synchronize() here -- this is the Megatron-like trigger.
        # .item() issues a blocking *pageable* D2H ordered behind the parked
        # collective-completion wait on the user stream.
        log(rank, f"iter{it} .item() enter (pageable D2H while collective parked)")
        v = x[0].item()
        log(rank, f"iter{it} .item() done v={v:.1f}")

    dist.destroy_process_group()
    log(rank, "destroy done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
