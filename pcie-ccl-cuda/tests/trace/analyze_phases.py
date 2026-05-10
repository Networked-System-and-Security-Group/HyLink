#!/usr/bin/env python3
"""Analyze trace phases per iteration to find bottlenecks.

Reports per-iteration D2H/H2H/H2D phase spans, SpinWait breakdown,
and pipeline overlap between phases.
"""
import sys
from collections import defaultdict
from trace_common import (
    TraceEvents, detect_iterations, iter_windows, events_in_window,
    span_us, percentiles_str,
)


def analyze(path):
    te = TraceEvents.from_file(path)

    d2h_dma = te.d2h
    h2d_dma = te.h2d
    h2h_sc = te.h2h_sc
    h2h_spin = te.h2h_spin

    # ── Event summary ──
    print("=== Event Summary ===")
    print(f"  D2H:          {len(d2h_dma)}")
    print(f"  H2D:          {len(h2d_dma)}")
    print(f"  H2H subchunks:{len(h2h_sc)}")
    print(f"  H2H SpinWait: {len(h2h_spin)}")
    print(f"  H2H outer:    {len(te.h2h_outer)}")
    print(f"  Scheduler:    {len(te.sched)}")
    print()

    # ── DMA size distributions ──
    for label, events in [("D2H", d2h_dma), ("H2D", h2d_dma)]:
        by_size = defaultdict(list)
        for e in events:
            sz = e.get("args", {}).get("bytes", 0)
            by_size[sz].append(e["dur"])
        if by_size:
            print(f"=== {label} DMA Transfers ===")
            for sz in sorted(by_size):
                durs = sorted(by_size[sz])
                print(f"  {sz/1024:.0f}KB: n={len(durs)}  {percentiles_str(durs, [0, 50, 90, 100])}")
            print()

    # ── H2H copy/reduce ──
    copies = [e for e in h2h_sc if "Copy" in e.get("name", "")]
    if copies:
        durs = [e["dur"] for e in copies]
        bws = [e.get("args", {}).get("bw_gbps", 0) for e in copies if e.get("args", {}).get("bw_gbps", 0) > 0]
        print(f"=== H2H Copy Subchunks ({len(copies)}) ===")
        print(f"  Duration: {percentiles_str(durs, [0, 50, 90, 99, 100])}")
        if bws:
            bws_s = sorted(bws)
            print(f"  BW: P0={bws_s[0]:.2f}  P50={bws_s[len(bws_s)//2]:.2f}  P100={bws_s[-1]:.2f} GB/s")
        print()

    # ── SpinWait ──
    if h2h_spin:
        durs = [e["dur"] for e in h2h_spin]
        print(f"=== SpinWait ({len(h2h_spin)}) ===")
        print(f"  Total: {sum(durs)/1000:.1f}ms across {len(durs)} events")
        print(f"  Duration: {percentiles_str(durs, [0, 50, 90, 99, 100])}")
        print()

    # ── Per-iteration analysis ──
    iterations = detect_iterations(te.sched)
    if not iterations:
        return
    windows = iter_windows(iterations, te.all_events)

    results = []
    for i, (start, end) in enumerate(windows):
        it_d2h = events_in_window(d2h_dma, start, end)
        it_h2h = events_in_window(h2h_sc, start, end)
        it_h2d = events_in_window(h2d_dma, start, end)
        it_spin = events_in_window(h2h_spin, start, end)

        results.append({
            "iter": i, "total": (end - start) / 1000,
            "d2h_span": span_us(it_d2h) / 1000,
            "h2h_span": span_us(it_h2h) / 1000,
            "h2d_span": span_us(it_h2d) / 1000,
            "spin_total": sum(e["dur"] for e in it_spin) / 1000,
            "spin_span": span_us(it_spin) / 1000,
            "n_d2h": len(it_d2h), "n_h2h": len(it_h2h), "n_h2d": len(it_h2d),
        })

    n_iter = len(results)
    print(f"=== Per-Iteration Timeline ({n_iter} iterations) ===")
    print(f"{'Iter':>4}  {'Total':>8}  {'D2H span':>9}  {'H2H span':>9}  {'H2D span':>9}  {'SpinSpan':>9}  {'SpinSum':>8}  {'#D2H':>5}  {'#H2H':>5}  {'#H2D':>5}")
    for r in results[-12:]:
        print(f"{r['iter']:>4}  {r['total']:>8.2f}  {r['d2h_span']:>9.2f}  {r['h2h_span']:>9.2f}  {r['h2d_span']:>9.2f}  {r['spin_span']:>9.2f}  {r['spin_total']:>8.2f}  {r['n_d2h']:>5}  {r['n_h2h']:>5}  {r['n_h2d']:>5}")

    # ── Averages (skip warmup) ──
    warmup = 5
    measured = results[warmup:]
    if measured:
        print()
        print(f"Averages (iterations {warmup}-{n_iter-1}, n={len(measured)}):")
        for key in ["total", "d2h_span", "h2h_span", "h2d_span", "spin_span", "spin_total"]:
            avg = sum(r[key] for r in measured) / len(measured)
            print(f"  {key:>12}: {avg:.2f} ms")

    # ── Pipeline overlap analysis ──
    print()
    print("=== Pipeline Overlap Analysis (last 10 measured iterations) ===")
    for r in results[-10:]:
        i = r["iter"]
        start, end = windows[i]
        it_d2h = events_in_window(d2h_dma, start, end)
        it_h2h = events_in_window(h2h_sc, start, end)
        it_h2d = events_in_window(h2d_dma, start, end)

        if not it_d2h or not it_h2h:
            continue

        d2h_start = min(e["ts"] for e in it_d2h)
        d2h_end = max(e["ts"] + e["dur"] for e in it_d2h)
        h2h_start = min(e["ts"] for e in it_h2h)
        h2h_end = max(e["ts"] + e["dur"] for e in it_h2h)

        if it_h2d:
            h2d_start = min(e["ts"] for e in it_h2d)
            h2d_end = max(e["ts"] + e["dur"] for e in it_h2d)
        else:
            h2d_start = h2d_end = 0

        d2h_h2h_overlap = max(0, min(d2h_end, h2h_end) - max(d2h_start, h2h_start)) / 1000

        print(f"  Iter {i}: D2H[{(d2h_start-start)/1000:.1f}-{(d2h_end-start)/1000:.1f}ms] "
              f"H2H[{(h2h_start-start)/1000:.1f}-{(h2h_end-start)/1000:.1f}ms] "
              f"H2D[{(h2d_start-start)/1000:.1f}-{(h2d_end-start)/1000:.1f}ms] "
              f"D2H||H2H={d2h_h2h_overlap:.1f}ms")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <trace.json>")
        sys.exit(1)
    analyze(sys.argv[1])
