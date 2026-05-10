#!/usr/bin/env python3
"""Plot bandwidth comparison charts from the standardized bandwidth CSV.

Produces 4 charts (one per collective), each with 6 lines:
  baseline + multipath for 2, 4, and 8 GPU cards.

Design:
  - X-axis: message size (bytes), log scale, tick labels at every-other size
  - Y-axis: bus bandwidth (GB/s)
  - Markers: card-count → shape mapping (2: square, 4: circle, 8: star)
  - Colors: 6 distinct colors (one per line)
  - Line styles: baseline = dashed, multipath = solid
  - Title: "<CollectiveName> - Bus Bandwidth"
  - Legend: "<mode>_<cards>gpu"
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

COLLECTIVES = ("allgather", "allreduce", "alltoall", "reducescatter")
COLLECTIVE_LABELS = {
    "allgather": "AllGather",
    "allreduce": "AllReduce",
    "alltoall": "AlltoAll",
    "reducescatter": "ReduceScatter",
}

# Card count → marker shape
MARKER_MAP = {
    2: "s",   # square
    4: "o",   # circle
    8: "*",   # star
}

# 6 distinct colors (colorblind-friendly where possible)
COLOR_MAP = {
    ("baseline", 2): "#1f77b4",   # blue
    ("baseline", 4): "#ff7f0e",   # orange
    ("baseline", 8): "#2ca02c",   # green
    ("multipath", 2): "#d62728",  # red
    ("multipath", 4): "#9467bd",  # purple
    ("multipath", 8): "#8c564b",  # brown
}

LINE_STYLE_MAP = {
    "baseline": "--",   # dashed
    "multipath": "-",   # solid
}


def size_label(size_bytes: int) -> str:
    """Human-readable size label: use KB/MB/GB as appropriate."""
    if size_bytes >= 1073741824:
        return f"{size_bytes / 1073741824:.0f}GB"
    elif size_bytes >= 1048576:
        return f"{size_bytes / 1048576:.0f}MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.0f}KB"
    else:
        return f"{size_bytes}B"


def plot_collective(collective: str, data: list[dict], output_path: Path, bw_column: str = "busbw_gbps") -> None:
    """Create a single-collective bandwidth comparison chart."""
    # Group data by (mode, cards)
    groups: dict[tuple[str, str, int], dict] = {}
    for row in data:
        if row["collective"] != collective:
            continue
        mode = row["mode"]
        cards = int(row["cards"])
        key = (collective, mode, cards)
        if key not in groups:
            groups[key] = {"size": [], "bw": []}
        groups[key]["size"].append(int(row["size_bytes"]))
        groups[key]["bw"].append(float(row[bw_column]))

    if not groups:
        print(f"  WARNING: No data for {collective}")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot lines in consistent order: baseline first, then multipath; within each, 2, 4, 8
    plot_order = [
        ("baseline", 2), ("baseline", 4), ("baseline", 8),
        ("multipath", 2), ("multipath", 4), ("multipath", 8),
    ]

    all_sizes: set[int] = set()
    for mode, cards in plot_order:
        key = (collective, mode, cards)
        if key not in groups:
            continue
        g = groups[key]
        # Sort by size
        pairs = sorted(zip(g["size"], g["bw"]))
        sizes = [s for s, _ in pairs]
        bws = [b for _, b in pairs]
        all_sizes.update(sizes)

        color = COLOR_MAP.get((mode, cards), "#333333")
        marker = MARKER_MAP.get(cards, "x")
        linestyle = LINE_STYLE_MAP.get(mode, "-")
        label = f"{mode}_{cards}gpu"

        ax.plot(sizes, bws,
                color=color, marker=marker, linestyle=linestyle,
                linewidth=1.8, markersize=8, label=label,
                markerfacecolor="white" if mode == "baseline" else color)

    # X-axis: log scale, every-other tick label
    ax.set_xscale("log")
    ax.set_xlabel("Message Size", fontsize=12)

    # Build tick positions: one per size, but only label every other one
    sorted_sizes = sorted(all_sizes)
    ax.set_xticks(sorted_sizes)
    # Label every other tick
    tick_labels = []
    for i, sz in enumerate(sorted_sizes):
        if i % 2 == 0:
            tick_labels.append(size_label(sz))
        else:
            tick_labels.append("")
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)

    # Y-axis
    ax.set_ylabel("Bus Bandwidth (GB/s)", fontsize=12)
    ax.grid(True, alpha=0.3, which="both")

    # Title
    title = f"{COLLECTIVE_LABELS.get(collective, collective)} - Bus Bandwidth"
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Legend
    ax.legend(loc="best", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=script_dir / "csv",
                        help="Directory containing standard_bw.csv")
    parser.add_argument("--input-name", default="standard_bw.csv",
                        help="Input standardized bandwidth CSV")
    parser.add_argument("--plots-dir", type=Path, default=None,
                        help="Output directory for plots (default: csv/plots/)")
    parser.add_argument("--bw-column", default="busbw_gbps",
                        choices=["busbw_gbps", "algbw_gbps"],
                        help="Which bandwidth metric to plot (default: busbw_gbps)")
    parser.add_argument("--collectives", nargs="+", default=list(COLLECTIVES),
                        help="Collectives to plot (default: all four)")
    args = parser.parse_args()

    input_csv = args.csv_dir / args.input_name
    if not input_csv.is_file():
        print(f"ERROR: {input_csv} not found. Run compute_standard_bw.py first.")
        return 1

    plots_dir = args.plots_dir or (args.csv_dir / "plots")

    # Read all data
    with input_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Read {len(rows)} rows from {input_csv}")
    print(f"Plotting '{args.bw_column}' to {plots_dir}/")
    print()

    for collective in args.collectives:
        fname = f"{collective}_bw.png"
        plot_collective(collective, rows, plots_dir / fname, args.bw_column)

    print(f"\nDone. Plots saved to {plots_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
