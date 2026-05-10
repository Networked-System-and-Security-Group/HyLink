"""Isolate the D2DReduce completion hang: tiny fp16 all_reduce with proper sync."""
import datetime as dt, os, sys, time
import torch, torch.distributed as dist

def log(r, m): print(f"[{time.strftime('%H:%M:%S')}] rank{r}: {m}", flush=True)

def main():
    rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr); dev = torch.device("cuda", lr)
    dist.init_process_group(backend="nccl", timeout=dt.timedelta(seconds=30), device_id=dev)
    log(rank, "init done")
    dtype = torch.float16 if os.environ.get("DT","fp16")=="fp16" else torch.float32
    n = int(os.environ.get("ELEMS","4"))
    for it in range(int(os.environ.get("ITERS","5"))):
        x = torch.full((n,), rank + 1.0, device=dev, dtype=dtype)
        log(rank, f"iter{it} all_reduce(n={n},dt={dtype}) enter")
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(dev)
        log(rank, f"iter{it} done sum0={x[0].item():.1f}")
    dist.destroy_process_group(); log(rank, "destroy done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
