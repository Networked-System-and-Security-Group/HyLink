#!/usr/bin/env python3
"""Analyze H2H subchunk duration distribution from Chrome JSON trace files.

Detects preempted events (high wall-time vs low CPU-time) and reports
per-CPU breakdown of stragglers.
"""
import sys
import os
from trace_common import TraceEvents, percentile


def analyze(path, label):
    te = TraceEvents.from_file(path)
    h2h = te.h2h_sc

    if not h2h:
        print(f"[{label}] No H2H subchunk events found")
        return

    durs = sorted(e["dur"] for e in h2h)
    n = len(durs)

    # Preempted events: wall > 2ms and cpu_us/wall < 0.3
    preempted = []
    normal = []
    for e in h2h:
        args = e.get("args", {})
        cpu_us = args.get("cpu_us", e["dur"])
        dur = e["dur"]
        if dur > 2000 and cpu_us * 10 < dur * 3:
            preempted.append(e)
        else:
            normal.append(e)

    normal_durs = sorted(e["dur"] for e in normal)
    preempted_durs = sorted(e["dur"] for e in preempted)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total subchunks: {n}")
    print(f"  Preempted (wall>2ms, cpu/wall<0.3): {len(preempted)} ({100*len(preempted)/n:.1f}%)")
    print()
    print(f"  {'Percentile':<12} {'All (us)':>10} {'Normal (us)':>12} {'Preempted (us)':>15}")
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*15}")
    for p in [0, 50, 75, 90, 95, 99, 100]:
        print(f"  P{p:<11} {percentile(durs, p):>10} {percentile(normal_durs, p):>12} {percentile(preempted_durs, p):>15}")

    # Per-CPU breakdown of preempted events
    if preempted:
        cpu_counts = {}
        for e in preempted:
            cpu = e.get("args", {}).get("cpu_before", "?")
            cpu_counts[cpu] = cpu_counts.get(cpu, 0) + 1
        print(f"\n  Preempted events by CPU:")
        for cpu, cnt in sorted(cpu_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    CPU {cpu:>3}: {cnt} events")

    # Top stragglers
    print(f"\n  Top 5 slowest subchunks:")
    for e in sorted(h2h, key=lambda x: -x["dur"])[:5]:
        args = e.get("args", {})
        print(f"    {e['dur']:>8}us  cpu_us={args.get('cpu_us','?'):>6}  "
              f"cpu={args.get('cpu_before','?')}->{args.get('cpu_after','?')}  "
              f"bw={args.get('bw_gbps',0):.1f}GB/s  {e['name'][:50]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <trace.json> [label]")
        sys.exit(1)
    label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(sys.argv[1])
    analyze(sys.argv[1], label)
