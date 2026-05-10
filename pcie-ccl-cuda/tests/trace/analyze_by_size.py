#!/usr/bin/env python3
"""Analyze trace iterations grouped by data size.

Reports per-size-group latency, phase breakdown, achieved algorithm
bandwidth, and subchunk-level statistics.
"""
import sys
from collections import defaultdict
from trace_common import (
    TraceEvents, detect_iterations, iter_windows, events_in_window,
    span_us, percentiles_str, format_size,
)


def analyze(path):
    te = TraceEvents.from_file(path)

    iterations = detect_iterations(te.sched)
    if not iterations:
        print("No scheduler events found.")
        return
    windows = iter_windows(iterations, te.all_events)

    # ── Build per-iteration info ──
    iter_info = []
    for i, (start, end) in enumerate(windows):
        it_d2h = events_in_window(te.d2h, start, end)
        it_h2d = events_in_window(te.h2d, start, end)
        it_h2h = events_in_window(te.h2h_sc, start, end)
        it_spin = events_in_window(te.h2h_spin, start, end)

        d2h_bytes = sum(e.get("args", {}).get("bytes", 0) for e in it_d2h)

        iter_info.append({
            "idx": i,
            "total_us": end - start,
            "d2h_bytes": d2h_bytes,
            "d2h_span": span_us(it_d2h),
            "h2h_span": span_us(it_h2h),
            "h2d_span": span_us(it_h2d),
            "spin_span": span_us(it_spin),
            "spin_total": sum(e["dur"] for e in it_spin),
            "n_d2h": len(it_d2h), "n_h2h": len(it_h2h), "n_h2d": len(it_h2d),
            "d2h_events": it_d2h, "h2d_events": it_h2d,
            "h2h_events": it_h2h, "spin_events": it_spin,
        })

    # ── Group by D2H bytes ──
    by_size = defaultdict(list)
    for info in iter_info:
        by_size[info["d2h_bytes"]].append(info)

    for sz in sorted(by_size):
        iters = by_size[sz]
        n = len(iters)
        warmup = min(2, n)
        measured = iters[warmup:] or iters

        print(f"\n{'='*70}")
        print(f"  Data Size: {format_size(sz)}  ({n} iterations, {len(measured)} measured)")
        print(f"{'='*70}")

        avg = lambda key: sum(r[key] for r in measured) / len(measured)
        avg_total = avg("total_us") / 1000
        avg_d2h = avg("d2h_span") / 1000
        avg_h2h = avg("h2h_span") / 1000
        avg_h2d = avg("h2d_span") / 1000
        avg_spin_span = avg("spin_span") / 1000
        avg_spin_total = avg("spin_total") / 1000

        print(f"  Avg Total: {avg_total:.3f} ms")
        print(f"  Avg D2H span: {avg_d2h:.3f} ms")
        print(f"  Avg H2H span: {avg_h2h:.3f} ms")
        print(f"  Avg H2D span: {avg_h2d:.3f} ms")
        print(f"  Avg SpinWait span: {avg_spin_span:.3f} ms")
        print(f"  Avg SpinWait total: {avg_spin_total:.3f} ms")
        print(f"  Events per iter: D2H={measured[0]['n_d2h']}  H2H={measured[0]['n_h2h']}  H2D={measured[0]['n_h2d']}")

        if avg_total > 0:
            print(f"\n  Phase % of total: D2H={100*avg_d2h/avg_total:.0f}%  H2H={100*avg_h2h/avg_total:.0f}%  H2D={100*avg_h2d/avg_total:.0f}%")

        if avg_total > 0 and sz > 0:
            bw = sz / (avg_total / 1000) / 1e9
            print(f"  Achieved Alg BW: {bw:.2f} GB/s")

        # ── Per-iteration detail ──
        print(f"\n  {'Iter':>4}  {'Total':>8}  {'D2H':>8}  {'H2H':>8}  {'H2D':>8}  {'Spin':>8}  {'Gap':>8}")
        print(f"  {'':>4}  {'(ms)':>8}  {'(ms)':>8}  {'(ms)':>8}  {'(ms)':>8}  {'(ms)':>8}  {'D2H-H2H':>8}")
        for r in measured:
            gap = 0
            if r["d2h_events"] and r["h2h_events"]:
                first_d2h_end = min(e["ts"] + e["dur"] for e in r["d2h_events"])
                first_h2h_start = min(e["ts"] for e in r["h2h_events"])
                gap = (first_h2h_start - first_d2h_end) / 1000

            print(f"  {r['idx']:>4}  {r['total_us']/1000:>8.2f}  {r['d2h_span']/1000:>8.2f}  "
                  f"{r['h2h_span']/1000:>8.2f}  {r['h2d_span']/1000:>8.2f}  "
                  f"{r['spin_total']/1000:>8.2f}  {gap:>8.3f}")

        # ── H2H subchunk distributions ──
        all_h2h = [e for r in measured for e in r["h2h_events"]]
        if all_h2h:
            durs = [e["dur"] for e in all_h2h]
            bws = sorted(e.get("args", {}).get("bw_gbps", 0) for e in all_h2h if e.get("args", {}).get("bw_gbps", 0) > 0)
            print(f"\n  H2H subchunk dur: {percentiles_str(durs, [0, 50, 90, 99, 100])}")
            if bws:
                nb = len(bws)
                print(f"  H2H subchunk BW:  P0={bws[0]:.2f}  P50={bws[nb//2]:.2f}  P90={bws[int(nb*0.9)]:.2f}  P100={bws[-1]:.2f} GB/s")

        all_spin = [e for r in measured for e in r["spin_events"]]
        if all_spin:
            print(f"  SpinWait dur:     {percentiles_str([e['dur'] for e in all_spin], [0, 50, 90, 99, 100])}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <trace.json>")
        sys.exit(1)
    analyze(sys.argv[1])
