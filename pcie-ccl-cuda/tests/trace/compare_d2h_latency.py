#!/usr/bin/env python3
"""Compare D2H DMA per-subchunk latency across trace files.

Focuses on 512MB AllGather iterations (64MB per rank = 64 × 1MB subchunks).
Reports per-subchunk latency percentiles, D2H span, and H2H overlap timing.

Usage:
    python3 compare_d2h_latency.py <trace1.json> <label1> [<trace2.json> <label2> ...]
"""
import json
import sys
import statistics

def load_trace(path):
    with open(path) as f:
        return json.load(f)

def extract_iterations(trace):
    """Extract per-iteration events, focusing on 64MB data size group (512MB AllGather)."""
    events = trace["traceEvents"]

    # Classify events
    d2h_events = []  # D2H DMA events
    h2h_events = []  # H2H copy/reduce subchunk events
    h2d_events = []  # H2D events (not needed for D2H analysis but useful)

    for e in events:
        name = e.get("name", "")
        ph = e.get("ph", "")
        if ph != "X":
            continue
        dur = e.get("dur", 0)
        ts = e.get("ts", 0)
        args = e.get("args", {})
        size = args.get("bytes", 0)

        if "D2H DMA" in name and size == 1048576:  # 1MB subchunks
            d2h_events.append({"ts": ts, "dur": dur, "size": size})
        elif "H2H Copy" in name or "H2H Reduce" in name:
            if "SpinWait" not in name:
                h2h_events.append({"ts": ts, "dur": dur})
        elif "H2D Batch" in name:
            h2d_events.append({"ts": ts, "dur": dur})

    # Group D2H events into iterations by detecting gaps > 5ms
    if not d2h_events:
        return []

    d2h_events.sort(key=lambda x: x["ts"])
    iterations = []
    current_iter = [d2h_events[0]]

    for e in d2h_events[1:]:
        if e["ts"] - (current_iter[-1]["ts"] + current_iter[-1]["dur"]) > 5000:
            iterations.append(current_iter)
            current_iter = [e]
        else:
            current_iter.append(e)
    iterations.append(current_iter)

    # Filter to 512MB iterations (should have ~64 D2H events of 1MB each)
    big_iters = [it for it in iterations if 50 <= len(it) <= 80]

    return big_iters, h2h_events

def percentile(data, p):
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    return s[f] + (k - f) * (s[c] - s[f])

def analyze_trace(path, label):
    trace = load_trace(path)
    big_iters, h2h_events = extract_iterations(trace)

    if not big_iters:
        print(f"  {label}: No 512MB iterations found")
        return None

    # Skip first 2 iterations (warmup)
    measured = big_iters[2:] if len(big_iters) > 2 else big_iters

    all_durs = []  # all subchunk durations in μs
    iter_spans = []  # per-iteration D2H span
    iter_d2h_starts = []  # per-iteration D2H start time

    for it in measured:
        durs = [e["dur"] for e in it]
        all_durs.extend(durs)

        span_start = min(e["ts"] for e in it)
        span_end = max(e["ts"] + e["dur"] for e in it)
        iter_spans.append(span_end - span_start)
        iter_d2h_starts.append(span_start)

    # Find H2H events overlapping with D2H
    h2h_during_d2h_count = 0
    h2h_during_d2h_dur = 0
    for it in measured:
        d2h_start = min(e["ts"] for e in it)
        d2h_end = max(e["ts"] + e["dur"] for e in it)
        for h in h2h_events:
            h_start = h["ts"]
            h_end = h["ts"] + h["dur"]
            overlap = min(d2h_end, h_end) - max(d2h_start, h_start)
            if overlap > 0:
                h2h_during_d2h_count += 1
                h2h_during_d2h_dur += overlap

    result = {
        "label": label,
        "n_iters": len(measured),
        "n_subchunks_per_iter": len(measured[0]) if measured else 0,
        "d2h_dur_p50": percentile(all_durs, 50),
        "d2h_dur_p90": percentile(all_durs, 90),
        "d2h_dur_p99": percentile(all_durs, 99),
        "d2h_dur_max": max(all_durs),
        "d2h_span_median": statistics.median(iter_spans),
        "d2h_span_mean": statistics.mean(iter_spans),
        "h2h_overlap_events": h2h_during_d2h_count // max(len(measured), 1),
        "h2h_overlap_dur_ms": h2h_during_d2h_dur / 1000.0 / max(len(measured), 1),
    }
    return result

def main():
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2 != 0:
        print(f"Usage: {sys.argv[0]} <trace.json> <label> [<trace.json> <label> ...]")
        sys.exit(1)

    pairs = [(args[i], args[i+1]) for i in range(0, len(args), 2)]
    results = []

    for path, label in pairs:
        print(f"Loading {label}: {path}")
        r = analyze_trace(path, label)
        if r:
            results.append(r)

    if not results:
        print("No results to compare")
        return

    # Print comparison table
    print()
    print("=" * 80)
    print("  D2H per-subchunk latency comparison (512MB AllGather, 1MB subchunks)")
    print("=" * 80)
    print()

    header = f"{'Metric':<30}"
    for r in results:
        header += f"  {r['label']:>14}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Iterations", "n_iters", "d"),
        ("Subchunks/iter", "n_subchunks_per_iter", "d"),
        ("D2H dur P50 (μs)", "d2h_dur_p50", ".0f"),
        ("D2H dur P90 (μs)", "d2h_dur_p90", ".0f"),
        ("D2H dur P99 (μs)", "d2h_dur_p99", ".0f"),
        ("D2H dur Max (μs)", "d2h_dur_max", ".0f"),
        ("D2H span median (ms)", "d2h_span_median", ".2f", 1e-3),
        ("D2H span mean (ms)", "d2h_span_mean", ".2f", 1e-3),
        ("H2H events during D2H", "h2h_overlap_events", "d"),
        ("H2H dur during D2H (ms)", "h2h_overlap_dur_ms", ".2f"),
    ]

    for row in rows:
        label_str, key, fmt = row[0], row[1], row[2]
        scale = row[3] if len(row) > 3 else 1
        line = f"{label_str:<30}"
        for r in results:
            val = r[key] * scale
            line += f"  {val:>14{fmt}}"
        print(line)

    # Print delta analysis
    if len(results) >= 2:
        print()
        print("Delta analysis:")
        base = results[0]
        for r in results[1:]:
            p50_delta = r["d2h_dur_p50"] - base["d2h_dur_p50"]
            p90_delta = r["d2h_dur_p90"] - base["d2h_dur_p90"]
            span_delta = (r["d2h_span_median"] - base["d2h_span_median"]) / 1000
            p50_pct = p50_delta / base["d2h_dur_p50"] * 100 if base["d2h_dur_p50"] else 0
            span_pct = span_delta / (base["d2h_span_median"]/1000) * 100 if base["d2h_span_median"] else 0
            print(f"  {r['label']} vs {base['label']}:")
            print(f"    P50: {p50_delta:+.0f}μs ({p50_pct:+.1f}%)")
            print(f"    P90: {p90_delta:+.0f}μs")
            print(f"    Span: {span_delta:+.2f}ms ({span_pct:+.1f}%)")
            h2h_delta = r["h2h_overlap_events"] - base["h2h_overlap_events"]
            print(f"    H2H events during D2H: {h2h_delta:+d}")

if __name__ == "__main__":
    main()
