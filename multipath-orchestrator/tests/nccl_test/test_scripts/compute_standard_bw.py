#!/usr/bin/env python3
"""Further process raw bandwidth CSV into a clean standard table with algbw and busbw.

Reads the raw CSV produced by process_compare_logs.py, applies trimming to multipath
samples, and computes both algorithm bandwidth (algbw) and bus bandwidth (busbw)
for every (collective, cards, size, mode) combination.

BusBwFactor per collective (from libampccl/core/shm_store.cc:30-41):
    AllReduce:     1.0
    AllGather:     n (card count)
    ReduceScatter: (n-1)/n
    AlltoAll:      (n-1)/n

IMPORTANT — AllToAll bandwidth convention correction:
    The AMPCCL code uses per-peer bytes (count*es) as op_key.bytes, so
    moved_bytes = per_peer_bytes and algbw = per_peer_bytes / time.
    The baseline test uses total per-rank bytes (n * per_peer_bytes),
    so baseline algbw = n * per_peer_bytes / time = n * AMPCCL algbw.
    This script multiplies AMPCCL AllToAll algbw and busbw by n to
    align with the baseline convention.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

COLLECTIVES = ("allgather", "allreduce", "alltoall", "reducescatter")
BASELINE_METRIC = "baseline_average_bandwidth"
MULTIPATH_ALG_METRIC = "multipath_alg_bandwidth"
MULTIPATH_TOTAL_METRIC = "multipath_total_bandwidth"


def bus_factor(collective: str, n_cards: int) -> float:
    """Return bus bandwidth factor: busbw = algbw * factor."""
    if n_cards <= 1:
        return 0.0
    c = collective.lower()
    if c == "allreduce":
        return 1.0
    elif c == "allgather":
        return float(n_cards)
    else:
        # reducescatter, alltoall
        return (n_cards - 1.0) / n_cards


def trimmed_mean(values: list[float], trim_n: int) -> tuple[float, int, int]:
    """Return (trimmed_mean, original_count, trimmed_count) after removing
    the `trim_n` lowest and `trim_n` highest values. If there aren't enough
    values to trim, falls back to plain mean."""
    if len(values) <= 2 * trim_n:
        # Not enough data to trim — use plain mean
        return mean(values), len(values), 0

    sorted_vals = sorted(values)
    trimmed = sorted_vals[trim_n:-trim_n]
    return mean(trimmed), len(values), 2 * trim_n


def process(raw_csv: Path, trim_n: int, output_csv: Path, skip_warmup: bool = True) -> list[dict]:
    """Read raw CSV, group samples, trim, compute algbw/busbw, return rows."""

    # Raw data grouped by (collective, cards, size_bytes, mode, metric)
    groups: defaultdict[tuple, list[float]] = defaultdict(list)

    with raw_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Optionally skip warmup samples for multipath metrics
            is_warmup = row.get("is_warmup", "").strip()
            if skip_warmup and is_warmup == "True":
                continue

            key = (
                row["collective"].strip().lower(),
                int(row["cards"]),
                int(row["size_bytes"]),
                row["mode"].strip(),
                row["metric"].strip(),
            )
            groups[key].append(float(row["bandwidth_gbps"]))

    # Build final table: one row per (collective, cards, size_bytes, mode)
    # Key: (collective, cards, size_bytes, mode) → {metric: trimmed_avg}
    result: dict[tuple, dict] = {}

    for (collective, cards, size_bytes, mode, metric), values in groups.items():
        key = (collective, cards, size_bytes, mode)
        if key not in result:
            result[key] = {
                "collective": collective,
                "cards": cards,
                "size_bytes": size_bytes,
                "mode": mode,
            }

        avg_val, num_samples, num_trimmed = trimmed_mean(values, trim_n)
        result[key][f"{metric}_avg"] = avg_val
        result[key][f"{metric}_samples"] = num_samples
        result[key][f"{metric}_trimmed"] = num_trimmed

    # Compute algbw and busbw for each row
    output_rows = []
    for key, data in sorted(result.items()):
        collective = data["collective"]
        cards = data["cards"]
        mode = data["mode"]
        bf = bus_factor(collective, cards)

        if mode == "baseline":
            # Baseline: algbw is directly measured, derive busbw
            algbw = data.get(f"{BASELINE_METRIC}_avg", None)
            if algbw is None:
                continue  # skip if no baseline data
            busbw = algbw * bf
            num_samples = data.get(f"{BASELINE_METRIC}_samples", 1)
            num_trimmed = 0
        else:
            # Multipath: prefer direct algbw from aggregate logs, else derive from busbw
            algbw_direct = data.get(f"{MULTIPATH_ALG_METRIC}_avg")
            busbw_direct = data.get(f"{MULTIPATH_TOTAL_METRIC}_avg")

            if algbw_direct is not None:
                algbw = algbw_direct
                busbw = algbw * bf
                num_samples = data.get(f"{MULTIPATH_ALG_METRIC}_samples", 0)
                num_trimmed = data.get(f"{MULTIPATH_ALG_METRIC}_trimmed", 0)
            elif busbw_direct is not None:
                busbw = busbw_direct
                algbw = busbw / bf if bf > 0 else 0.0
                num_samples = data.get(f"{MULTIPATH_TOTAL_METRIC}_samples", 0)
                num_trimmed = data.get(f"{MULTIPATH_TOTAL_METRIC}_trimmed", 0)
            else:
                continue  # skip if no multipath data

            # AllToAll correction: AMPCCL uses per-peer bytes for algbw,
            # but baseline convention uses total per-rank bytes (n * per-peer).
            # Multiply by n to align with the standard convention.
            if collective == "alltoall":
                algbw *= cards
                busbw *= cards

        output_rows.append({
            "collective": collective,
            "mode": mode,
            "cards": cards,
            "size_bytes": data["size_bytes"],
            "algbw_gbps": round(algbw, 6),
            "busbw_gbps": round(busbw, 6),
            "bus_factor": round(bf, 6),
            "num_samples": num_samples,
            "num_trimmed": num_trimmed,
        })

    return output_rows


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=script_dir / "csv",
                        help="Directory containing raw_bandwidth.csv and output destination")
    parser.add_argument("--raw-name", default="raw_bandwidth.csv",
                        help="Input raw CSV filename")
    parser.add_argument("--output-name", default="standard_bw.csv",
                        help="Output clean CSV filename")
    parser.add_argument("--trim-n", type=int, default=8,
                        help="Number of highest and lowest samples to trim (default: 8)")
    parser.add_argument("--no-skip-warmup", action="store_true",
                        help="Include warmup samples in trimming/averaging")
    args = parser.parse_args()

    raw_csv = args.csv_dir / args.raw_name
    if not raw_csv.is_file():
        print(f"ERROR: {raw_csv} not found. Run process_compare_logs.py first.")
        return 1

    output_rows = process(
        raw_csv=raw_csv,
        trim_n=args.trim_n,
        output_csv=args.csv_dir / args.output_name,
        skip_warmup=not args.no_skip_warmup,
    )

    output_csv = args.csv_dir / args.output_name
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "collective", "mode", "cards", "size_bytes",
        "algbw_gbps", "busbw_gbps", "bus_factor",
        "num_samples", "num_trimmed",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    # Print summary
    collectives_seen = sorted(set(r["collective"] for r in output_rows))
    modes_seen = sorted(set(r["mode"] for r in output_rows))
    card_counts_seen = sorted(set(r["cards"] for r in output_rows))
    print(f"output_rows={len(output_rows)} -> {output_csv}")
    print(f"collectives={collectives_seen}")
    print(f"modes={modes_seen}")
    print(f"card_counts={card_counts_seen}")
    print(f"trim_n={args.trim_n}")
    return 0 if output_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
