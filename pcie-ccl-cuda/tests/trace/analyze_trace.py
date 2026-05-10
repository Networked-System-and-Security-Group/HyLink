#!/usr/bin/env python3
"""General trace analysis: transfer size distributions, H2H bandwidth,
SpinWait statistics, per-iteration phase timing, and per-rank timing.
"""
import sys
from collections import defaultdict
from trace_common import (
    TraceEvents, detect_iterations, iter_windows, events_in_window,
    span_us, percentiles_str, get_rank,
)


def analyze(path):
    te = TraceEvents.from_file(path)

    print("Event counts:")
    print(f"  D2H:    {len(te.d2h)}")
    print(f"  H2H:    {len(te.h2h_sc) + len(te.h2h_outer)}")
    print(f"  H2D:    {len(te.h2d)}")
    print(f"  Sched:  {len(te.sched)}")
    print(f"  Spin:   {len(te.h2h_spin)}")
    print()

    # ── DMA size distributions ──
    for label, events in [("D2H", te.d2h), ("H2D", te.h2d)]:
        by_size = defaultdict(list)
        for e in events:
            sz = e.get("args", {}).get("bytes", 0)
            by_size[sz].append(e)
        if by_size:
            print(f"{label} transfer sizes:")
            for sz in sorted(by_size):
                durs = [e["dur"] for e in by_size[sz]]
                avg = sum(durs) / len(durs)
                print(f"  {sz/1024:.0f}KB: {len(durs)} transfers, avg {avg:.0f}us, "
                      f"min {min(durs)}us, max {max(durs)}us")
            print()

    # ── H2H copy/reduce breakdown ──
    if te.h2h_sc:
        h2h_by_type = defaultdict(list)
        for e in te.h2h_sc:
            name = e.get("name", "")
            if "Copy" in name:
                h2h_by_type["copy"].append(e)
            elif "Reduce" in name:
                h2h_by_type["reduce"].append(e)

        for typ, evts in h2h_by_type.items():
            durs = [e["dur"] for e in evts]
            bws = [e.get("args", {}).get("bw_gbps", 0) for e in evts]
            print(f"H2H {typ}: {len(evts)} subchunks")
            print(f"  Duration: {percentiles_str(durs, [0, 50, 90, 99, 100])}")
            bws_pos = sorted(b for b in bws if b > 0)
            if bws_pos:
                print(f"  BW: P0={bws_pos[0]:.1f}  P50={bws_pos[len(bws_pos)//2]:.1f}  P100={bws_pos[-1]:.1f} GB/s")
            print()

    # ── SpinWait ──
    if te.h2h_spin:
        durs = [e["dur"] for e in te.h2h_spin]
        print(f"SpinWait: {len(durs)} events, total {sum(durs)/1000:.1f}ms")
        print(f"  {percentiles_str(durs, [0, 50, 90, 100])}")
        print()

    # ── Per-iteration phase analysis ──
    iterations = detect_iterations(te.sched)
    if not iterations:
        return

    windows = iter_windows(iterations, te.all_events)
    print(f"Detected {len(iterations)} iterations from scheduler events")
    print()

    if len(iterations) > 1:
        print("Per-iteration phase analysis (last 10 iterations):")
        print(f"{'Iter':>4}  {'Total(ms)':>10}  {'D2H span':>10}  {'H2H span':>10}  {'H2D span':>10}  {'SpinWait':>10}")
        print(f"{'----':>4}  {'----------':>10}  {'----------':>10}  {'----------':>10}  {'----------':>10}  {'----------':>10}")

        show_start = max(0, len(windows) - 10)
        for i in range(show_start, len(windows)):
            start, end = windows[i]
            d2h_s = span_us(events_in_window(te.d2h, start, end)) / 1000
            h2h_s = span_us(events_in_window(te.h2h_sc, start, end)) / 1000
            h2d_s = span_us(events_in_window(te.h2d, start, end)) / 1000
            spin_s = sum(e["dur"] for e in events_in_window(te.h2h_spin, start, end)) / 1000

            print(f"{i:>4}  {(end-start)/1000:>10.2f}  {d2h_s:>10.2f}  {h2h_s:>10.2f}  {h2d_s:>10.2f}  {spin_s:>10.2f}")

    # ── Per-rank timing ──
    print()
    print("Per-rank timing (across all iterations):")

    for label, events in [("D2H", te.d2h), ("H2D", te.h2d)]:
        rank_map = defaultdict(list)
        for e in events:
            r = get_rank(e)
            if r is not None:
                rank_map[r].append(e)
        if rank_map:
            print(f"  {label} per rank (avg duration):")
            for r in sorted(rank_map):
                durs = [e["dur"] for e in rank_map[r]]
                print(f"    Rank {r}: {len(durs)} transfers, avg {sum(durs)/len(durs):.0f}us, total {sum(durs)/1000:.1f}ms")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <trace.json>")
        sys.exit(1)
    analyze(sys.argv[1])
