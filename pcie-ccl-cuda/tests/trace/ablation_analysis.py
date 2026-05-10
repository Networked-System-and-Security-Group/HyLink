#!/usr/bin/env python3
"""
Precise per-phase ablation analysis: identify exactly where SP vs MP time differs.

For each iteration, measures wall-clock span of each pipeline phase:
  D2H DMA → H2H (Copy+Reduce) → H2D DMA → Barriers → Cleanup

All times in microseconds internally, displayed in milliseconds.
"""
import json, sys, os
from collections import defaultdict

def load_events(path):
    with open(path) as f:
        data = json.load(f)
    evts = data.get("traceEvents", []) if isinstance(data, dict) else data
    return [e for e in evts if isinstance(e, dict) and e.get("ph") == "X" and e.get("dur", 0) >= 0]

def classify(name):
    nl = name.lower()
    if "d2h dma" in nl: return "D2H_DMA"
    if "h2d dma" in nl: return "H2D_DMA"
    if "d2h batch" in nl or "d2h submit" in nl: return "D2H_BATCH"
    if "h2d batch" in nl: return "H2D_BATCH"
    if "barrier1" in nl: return "BARRIER1"
    if "barrier2" in nl: return "BARRIER2"
    if "cleanup" in nl: return "CLEANUP"
    if "h2h copy" in name and "#" in name and "sc=" not in name: return "H2H_COPY_OUTER"
    if "h2h reduce" in name and "#" in name and "sc=" not in name: return "H2H_REDUCE_OUTER"
    if "h2h copy" in name and "sc=" in name: return "H2H_COPY_SC"
    if "h2h reduce" in name and "sc=" in name: return "H2H_REDUCE_SC"
    if "spinwait" in nl: return "SPINWAIT"
    return None

def split_iterations(events, gap_us=5000):
    if not events:
        return []
    sorted_evts = sorted(events, key=lambda e: e["ts"])
    iters = []
    cur = [sorted_evts[0]]
    for e in sorted_evts[1:]:
        prev_end = cur[-1]["ts"] + cur[-1].get("dur", 0)
        if e["ts"] - prev_end > gap_us:
            iters.append(cur)
            cur = []
        cur.append(e)
    if cur:
        iters.append(cur)
    return iters

def phase_stats(events):
    """For a single iteration, compute per-phase: span, total_dur, count, per-event avg."""
    phases = {}
    for e in events:
        c = classify(e.get("name", ""))
        if not c:
            continue
        if c not in phases:
            phases[c] = {"count": 0, "total_dur": 0, "min_ts": float('inf'), "max_end": 0, "durs": []}
        p = phases[c]
        p["count"] += 1
        p["total_dur"] += e["dur"]
        p["min_ts"] = min(p["min_ts"], e["ts"])
        p["max_end"] = max(p["max_end"], e["ts"] + e["dur"])
        p["durs"].append(e["dur"])

    result = {}
    for c, p in phases.items():
        span = (p["max_end"] - p["min_ts"])  # us
        avg_dur = p["total_dur"] / p["count"] if p["count"] else 0
        result[c] = {
            "span_us": span,
            "total_dur_us": p["total_dur"],
            "count": p["count"],
            "avg_event_dur_us": avg_dur,
            "min_ts": p["min_ts"],
            "max_end": p["max_end"],
            "p50_dur": sorted(p["durs"])[len(p["durs"])//2] if p["durs"] else 0,
            "p99_dur": sorted(p["durs"])[int(len(p["durs"])*0.99)] if p["durs"] else 0,
        }

    # Iteration span
    all_ts = [e["ts"] for e in events]
    all_end = [e["ts"] + e.get("dur", 0) for e in events]
    result["_ITER"] = {"span_us": max(all_end) - min(all_ts), "min_ts": min(all_ts), "max_end": max(all_end)}
    return result

def analyze_sp(path):
    events = load_events(path)
    iters = split_iterations(events)
    # Last 10 iterations = 512MB measured
    measured = iters[-10:] if len(iters) >= 10 else iters
    print(f"SP: {len(iters)} total iters, using last {len(measured)}")

    # Aggregate
    all_phases = defaultdict(lambda: defaultdict(list))
    for it_evts in measured:
        ps = phase_stats(it_evts)
        for phase, stats in ps.items():
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    all_phases[phase][k].append(v)

    return all_phases

def analyze_mp(paths):
    """Analyze MP: per-rank, then aggregate as max across ranks per iteration."""
    rank_iters = []
    for path in sorted(paths):
        events = load_events(path)
        iters = split_iterations(events)
        measured = iters[-10:] if len(iters) >= 10 else iters
        rank_iters.append(measured)

    n_iters = min(len(ri) for ri in rank_iters)
    print(f"MP: {len(rank_iters)} ranks, {n_iters} measured iters")

    # For each iteration, compute per-rank phase stats, then take max span across ranks
    all_phases = defaultdict(lambda: defaultdict(list))
    for i in range(n_iters):
        # Collect all phases seen across any rank
        phase_names = set()
        rank_stats = []
        for ri in rank_iters:
            ps = phase_stats(ri[i])
            rank_stats.append(ps)
            phase_names.update(ps.keys())

        for phase in phase_names:
            # Max span across ranks (bottleneck rank determines iteration time)
            spans = [rs[phase]["span_us"] for rs in rank_stats if phase in rs]
            if spans:
                all_phases[phase]["span_us"].append(max(spans))
            for key in ["count", "total_dur_us", "avg_event_dur_us", "p50_dur", "p99_dur"]:
                vals = [rs[phase].get(key, 0) for rs in rank_stats if phase in rs]
                if vals:
                    if key == "count":
                        all_phases[phase][key].append(sum(vals))
                    else:
                        all_phases[phase][key].append(sum(vals)/len(vals))

    return all_phases

def print_comparison(sp_phases, mp_phases):
    # Phases in pipeline order
    phase_order = [
        "D2H_DMA", "D2H_BATCH",
        "H2H_COPY_OUTER", "H2H_COPY_SC", "H2H_REDUCE_OUTER", "H2H_REDUCE_SC",
        "SPINWAIT",
        "H2D_DMA", "H2D_BATCH",
        "BARRIER1", "CLEANUP", "BARRIER2",
        "_ITER"
    ]
    all_phases = set(list(sp_phases.keys()) + list(mp_phases.keys()))
    # Add any phases not in order
    for p in sorted(all_phases):
        if p not in phase_order:
            phase_order.insert(-1, p)  # before _ITER

    def avg(lst):
        return sum(lst)/len(lst) if lst else 0

    print(f"\n{'Phase':<22s} {'SP span(ms)':>12s} {'MP span(ms)':>12s} {'Δ(ms)':>8s} {'Δ%':>7s}  "
          f"{'SP p50(μs)':>10s} {'MP p50(μs)':>10s} {'SP p99(μs)':>10s} {'MP p99(μs)':>10s}  "
          f"{'SP cnt':>8s} {'MP cnt':>8s}")
    print("-" * 140)

    for phase in phase_order:
        sp = sp_phases.get(phase, {})
        mp = mp_phases.get(phase, {})
        sp_span = avg(sp.get("span_us", [0])) / 1000
        mp_span = avg(mp.get("span_us", [0])) / 1000
        delta = mp_span - sp_span
        pct = (delta / sp_span * 100) if sp_span > 0 else 0
        sp_p50 = avg(sp.get("p50_dur", [0]))
        mp_p50 = avg(mp.get("p50_dur", [0]))
        sp_p99 = avg(sp.get("p99_dur", [0]))
        mp_p99 = avg(mp.get("p99_dur", [0]))
        sp_cnt = avg(sp.get("count", [0]))
        mp_cnt = avg(mp.get("count", [0]))

        if sp_span == 0 and mp_span == 0:
            continue

        marker = " <<<" if abs(delta) > 0.5 else ""
        print(f"{phase:<22s} {sp_span:12.2f} {mp_span:12.2f} {delta:+8.2f} {pct:+6.1f}%  "
              f"{sp_p50:10.0f} {mp_p50:10.0f} {sp_p99:10.0f} {mp_p99:10.0f}  "
              f"{sp_cnt:8.0f} {mp_cnt:8.0f}{marker}")

def main():
    sp_file = "/workspace/pcieccl/ag_sp_8g.json"
    mp_files = sorted([f"/workspace/pcieccl/ag_mp_8g_rank{i}.json" for i in range(8)])

    # Verify files exist
    for f in [sp_file] + mp_files:
        if not os.path.exists(f):
            print(f"Missing: {f}")
            sys.exit(1)

    print("=" * 140)
    print("  PHASE-LEVEL SP vs MP COMPARISON (512MB AllGather, 8 GPU)")
    print("  span = wall-clock extent of phase (first event start → last event end)")
    print("  p50/p99 = per-event duration percentiles")
    print("=" * 140)

    sp_phases = analyze_sp(sp_file)
    mp_phases = analyze_mp(mp_files)
    print_comparison(sp_phases, mp_phases)

    # === Detailed D2H inter-event gap analysis ===
    print("\n" + "=" * 100)
    print("  D2H DMA INTER-EVENT GAP ANALYSIS (temporal clustering)")
    print("=" * 100)

    for label, path in [("SP", sp_file), ("MP rank0", mp_files[0])]:
        events = load_events(path)
        iters = split_iterations(events)
        measured = iters[-10:] if len(iters) >= 10 else iters

        all_gaps = []
        for it_evts in measured:
            d2h = sorted([e for e in it_evts if classify(e.get("name","")) == "D2H_DMA"],
                         key=lambda e: e["ts"])
            for i in range(1, len(d2h)):
                gap = d2h[i]["ts"] - (d2h[i-1]["ts"] + d2h[i-1]["dur"])
                all_gaps.append(gap)

        if all_gaps:
            all_gaps.sort()
            n = len(all_gaps)
            big_gaps = [g for g in all_gaps if g > 1000]  # > 1ms
            print(f"\n  {label}: {n} gaps total")
            print(f"    p50={all_gaps[n//2]:.0f}μs  p90={all_gaps[int(n*0.9)]:.0f}μs  "
                  f"p99={all_gaps[int(n*0.99)]:.0f}μs  max={all_gaps[-1]:.0f}μs")
            print(f"    Gaps > 1ms: {len(big_gaps)} ({len(big_gaps)/n*100:.1f}%)")
            if big_gaps:
                print(f"    Big gap p50={sorted(big_gaps)[len(big_gaps)//2]:.0f}μs  "
                      f"max={max(big_gaps):.0f}μs")

if __name__ == "__main__":
    main()
