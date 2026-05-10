#!/usr/bin/env python3
"""Merge msprof NPU hardware trace into a PCCL trace file.

Usage:
    python3 merge_msprof.py <pccl_trace.json> <msprof.json> [-o merged.json]

Extracts from msprof:
  - AscendCL@aclrtMemcpyAsync  (CPU-side DMA submissions)
  - MEMCPY_ASYNC               (NPU hardware DMA execution)
  - Computing                  (NPU kernel execution)
  - HostToDevice* flow events  (arrows linking CPU submission → NPU execution)

Shifts all timestamps onto the PCCL timeline using wall_epoch_us.
"""
import json, argparse, sys
from collections import defaultdict
from trace_common import load_events

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pccl", help="PCCL trace JSON")
    ap.add_argument("msprof", help="msprof JSON")
    ap.add_argument("-o", "--output", default="merged.json")
    args = ap.parse_args()

    print(f"Loading PCCL trace: {args.pccl}")
    _, pccl = load_events(args.pccl)

    wall_epoch_us = pccl.get("wall_epoch_us")
    if wall_epoch_us is None:
        print("ERROR: PCCL trace missing 'wall_epoch_us'. Rebuild with latest trace_writer.", file=sys.stderr)
        sys.exit(1)

    events = pccl["traceEvents"]
    print(f"  {len(events)} existing events, wall_epoch_us={wall_epoch_us}")

    print(f"Loading msprof trace: {args.msprof}")
    with open(args.msprof) as f:
        msprof = json.load(f)
    print(f"  {len(msprof)} total msprof events")

    # ── Allocate PIDs that won't collide with PCCL ──
    pccl_pids = {e.get("pid", 0) for e in events}
    base_pid = max(pccl_pids) + 1 if pccl_pids else 1
    cpu_pid = base_pid      # CPU-side DMA API calls
    npu_pid = base_pid + 1  # NPU hardware events

    events.append({"name": "process_name", "ph": "M", "pid": cpu_pid, "tid": 0,
                    "args": {"name": "CPU DMA Submit (msprof)"}})
    events.append({"name": "process_name", "ph": "M", "pid": npu_pid, "tid": 0,
                    "args": {"name": "NPU Hardware (msprof)"}})

    # ── Build remapping tables ──
    # msprof (pid, tid) → new tid under our cpu_pid / npu_pid
    cpu_threads = {}   # for CANN process events
    npu_threads = {}   # for Ascend Hardware events

    # Identify CANN pid(s) (the process hosting aclrtMemcpyAsync)
    cann_pids = {e["pid"] for e in msprof
                 if e.get("name") == "AscendCL@aclrtMemcpyAsync"}

    # Collect all msprof (pid,tid) that appear in flow events so we can remap them
    # Build a global (old_pid, old_tid) → (new_pid, new_tid) map
    def get_cpu_tid(old_pid, old_tid):
        key = (old_pid, old_tid)
        if key not in cpu_threads:
            tid = len(cpu_threads) + 1
            cpu_threads[key] = tid
            events.append({"name": "thread_name", "ph": "M", "pid": cpu_pid,
                           "tid": tid, "args": {"name": f"Thread {old_tid}"}})
        return cpu_threads[key]

    def get_npu_tid(old_pid, old_tid, label=None):
        key = (old_pid, old_tid)
        if key not in npu_threads:
            tid = len(npu_threads) + 1
            npu_threads[key] = tid
            events.append({"name": "thread_name", "ph": "M", "pid": npu_pid,
                           "tid": tid, "args": {"name": label or f"Stream {old_tid}"}})
        return npu_threads[key]

    counts = defaultdict(int)

    for e in msprof:
        name = e.get("name", "")
        ph = e.get("ph", "")

        # ── CPU-side DMA API calls ──
        if name == "AscendCL@aclrtMemcpyAsync" and ph == "X":
            tid = get_cpu_tid(e["pid"], e["tid"])
            events.append({
                "name": "aclrtMemcpyAsync",
                "cat": "acl_api",
                "ph": "X",
                "ts": float(e["ts"]) - wall_epoch_us,
                "dur": e.get("dur", 0),
                "pid": cpu_pid,
                "tid": tid,
                "args": e.get("args", {}),
            })
            counts["api"] += 1

        # ── NPU hardware DMA execution ──
        elif name == "MEMCPY_ASYNC" and ph == "X":
            stream = e.get("args", {}).get("Physic Stream Id", e["tid"])
            tid = get_npu_tid(e["pid"], e["tid"], f"DMA Stream {stream}")
            events.append({
                "name": "MEMCPY_ASYNC",
                "cat": "npu_hw",
                "ph": "X",
                "ts": float(e["ts"]) - wall_epoch_us,
                "dur": e.get("dur", 0),
                "pid": npu_pid,
                "tid": tid,
                "args": e.get("args", {}),
            })
            counts["dma"] += 1

        # ── NPU kernel execution ──
        elif name == "Computing" and ph == "X":
            tid = get_npu_tid(e["pid"], e["tid"], f"Compute")
            events.append({
                "name": "Computing",
                "cat": "npu_hw",
                "ph": "X",
                "ts": float(e["ts"]) - wall_epoch_us,
                "dur": e.get("dur", 0),
                "pid": npu_pid,
                "tid": tid,
                "args": e.get("args", {}),
            })
            counts["kernel"] += 1

        # ── Flow events: CPU submit → NPU execute ──
        elif ph == "s" and e.get("cat") == "HostToDevice":
            old_pid = e["pid"]
            if old_pid in cann_pids:
                tid = get_cpu_tid(old_pid, e["tid"])
                new_pid = cpu_pid
            else:
                tid = get_npu_tid(old_pid, e["tid"])
                new_pid = npu_pid
            events.append({
                "name": name,
                "ph": "s",
                "cat": "HostToDevice",
                "id": e["id"],
                "ts": float(e["ts"]) - wall_epoch_us,
                "pid": new_pid,
                "tid": tid,
            })
            counts["flow"] += 1

        elif ph == "f" and e.get("cat") == "HostToDevice":
            old_pid = e["pid"]
            if old_pid in cann_pids:
                tid = get_cpu_tid(old_pid, e["tid"])
                new_pid = cpu_pid
            else:
                tid = get_npu_tid(old_pid, e["tid"])
                new_pid = npu_pid
            events.append({
                "name": name,
                "ph": "f",
                "cat": "HostToDevice",
                "id": e["id"],
                "ts": float(e["ts"]) - wall_epoch_us,
                "pid": new_pid,
                "tid": tid,
                "bp": "e",
            })
            counts["flow"] += 1

    print(f"  Merged: {counts['api']} CPU API calls, {counts['dma']} DMA HW, "
          f"{counts['kernel']} kernels, {counts['flow']} flow arrows")
    print(f"  Tracks: {len(cpu_threads)} CPU threads, {len(npu_threads)} NPU streams")

    pccl["traceEvents"] = events
    with open(args.output, "w") as f:
        json.dump(pccl, f)
    print(f"Written: {args.output}")

if __name__ == "__main__":
    main()
