import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INTERFERENCE_RE = re.compile(r"Interference ranges \(iter\):(.+)")
ALPHA_RANK0_RE = re.compile(r"\[Rank 0\].*AllGather before:.*alpha=([0-9.]+)")
BEGIN_FORMAL_RE = re.compile(r"\[rank0\] begin formal (\d+) iters")
TOTAL_BW_RE = re.compile(r"Total bw is ([0-9.]+) GB/s")


def parse_interference_ranges(line: str):
    """
    Parse interference ranges from log line:
    Format: Interference ranges (iter): [10,20] [30,40]
    Returns list [(10, 20), (30, 40)]
    """
    m = INTERFERENCE_RE.search(line)
    if not m:
        return []
    ranges = []
    rest = m.group(1)
    for part in rest.strip().split():
        part = part.strip()
        if not part.startswith("[") or not part.endswith("]"):
            continue
        try:
            a, b = part[1:-1].split(",")
            start = int(a)
            end = int(b)
            if end >= start:
                ranges.append((start, end))
        except Exception:
            continue
    return ranges


def parse_log_file(path: Path):
    """
    Parse a single log file, returns:
      - alphas: dynamic alpha (in formal iter order)
      - bws: total bandwidth (GB/s) per formal iter
      - interference_ranges: list of interference iter ranges

    Note: Total bw in log is "previous round" bandwidth:
      - bw after begin formal 0 is last warmup bandwidth (discard)
      - bw after begin formal k (>0) is formal iter k-1 bandwidth
      - alpha from Rank 0 AllGather before line, corresponds to current iter
    """
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    interference_ranges = []
    alphas = []
    bws = []

    last_alpha_rank0 = None
    current_formal_idx = None  # last seen "begin formal K"

    for line in text:
        # interference ranges
        if not interference_ranges:
            if "Interference ranges (iter):" in line:
                interference_ranges = parse_interference_ranges(line)

        # Rank0 alpha
        m_alpha = ALPHA_RANK0_RE.search(line)
        if m_alpha:
            try:
                last_alpha_rank0 = float(m_alpha.group(1))
            except ValueError:
                pass
            continue

        # formal iter start
        m_begin = BEGIN_FORMAL_RE.search(line)
        if m_begin:
            try:
                current_formal_idx = int(m_begin.group(1))
            except ValueError:
                current_formal_idx = None
            continue

        # Total bandwidth (previous round)
        m_bw = TOTAL_BW_RE.search(line)
        if m_bw and current_formal_idx is not None:
            # current_formal_idx == 0 means last warmup bandwidth, discard
            if current_formal_idx > 0 and last_alpha_rank0 is not None:
                try:
                    bw = float(m_bw.group(1))
                except ValueError:
                    bw = None
                if bw is not None:
                    bws.append(bw)
                    alphas.append(last_alpha_rank0)

    return {
        "alpha": np.array(alphas, dtype=float),
        "bw": np.array(bws, dtype=float),
        "interference_ranges": interference_ranges,
    }


def plot_figure(iters, bw_static, bw_dynamic, alpha_static, alpha_dynamic, interference_ranges, output_path: Path):
    """
    Plot similar to figure3-2.py:
      - Left axis: static/dynamic bandwidth
      - Right axis: static/dynamic alpha
      - Interference ranges: marked with axvspan
    """
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "grid.alpha": 0.15,
    })

    x = np.arange(len(iters))

    fig, ax1 = plt.subplots(figsize=(12, 7), dpi=150)

    # Paper-style vivid colors (high contrast, print-friendly)
    color_bw_static = "#0173B2"   # blue
    color_bw_dynamic = "#DE8F05"  # orange
    color_alpha_static = "#029E73"  # green (baseline)
    color_alpha_dynamic = "#CC78BC"  # purple-pink (AmpCCL)

    # Interference range shading (iter shown 1-based)
    for i, (start, end) in enumerate(interference_ranges):
        label = "PCIe Contention" if i == 0 else ""
        ax1.axvspan(start - 0.5, end + 0.5, color="#f8d7da", alpha=0.4, label=label, zorder=0)
        ax1.text((start + end) / 2.0, max(bw_static.max(), bw_dynamic.max()) * 1.05,
                 "Interference", color="#c0392b",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", alpha=0.8)

    # Bandwidth: line only (no points to avoid crowding when many iters)
    ax1.plot(x, bw_static, color=color_bw_static, linestyle="-", linewidth=1.8,
             label="Throughput (Static)", zorder=2)
    ax1.plot(x, bw_dynamic, color=color_bw_dynamic, linestyle="-", linewidth=1.8,
             label="Throughput (Dynamic)", zorder=2)

    ax1.set_xlabel("Iteration", fontsize=14, fontweight="bold", labelpad=12)
    ax1.set_ylabel("Throughput (GB/s)", fontsize=14, fontweight="bold", labelpad=12)

    # X-axis: show ~1/10 of iters (10, 20, 30, ...) and last (e.g. 48, 64), avoid crowding
    n = len(iters)
    if n <= 10:
        tick_indices = list(range(n))
    else:
        step = 100  # show every 10 iters
        tick_indices = list(range(step - 1, n, step))  # iter=10,20,30,...
        if (n - 1) not in tick_indices:
            tick_indices.append(n - 1)
            tick_indices.sort()
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels([iters[i] for i in tick_indices])

    max_bw = float(max(bw_static.max(initial=0), bw_dynamic.max(initial=0)))
    ax1.set_ylim(0, max_bw * 1.2 if max_bw > 0 else 1.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Alpha line (right axis): line only, no points
    ax2 = ax1.twinx()
    ax2.plot(x, alpha_static, color=color_alpha_static, linestyle="--", linewidth=1.5,
             label=r"Static $\alpha$ (Baseline)", zorder=3)
    ax2.plot(x, alpha_dynamic, color=color_alpha_dynamic, linestyle="-", linewidth=1.8,
             label=r"Dynamic $\alpha$ (AmpCCL)", zorder=4)

    max_alpha = float(max(max(alpha_static), max(alpha_dynamic)))
    ax2.set_ylabel(r"Path Splitting Ratio ($\alpha$)", fontsize=14, fontweight="bold",
                   color=color_alpha_dynamic, labelpad=12)
    ax2.set_ylim(0.0, min(1.0, max_alpha * 1.2 if max_alpha > 0 else 1.0))
    ax2.tick_params(axis="y", labelcolor=color_alpha_dynamic, labelsize=11)
    ax2.spines["right"].set_color(color_alpha_dynamic)
    ax2.spines["right"].set_linewidth(1.5)

    # Merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center",
               bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False,
               fontsize=12, columnspacing=1.5)

    plt.title("Experiment 3: Resilience to Dynamic PCIe Interference",
              fontsize=18, fontweight="bold", pad=60)
    plt.subplots_adjust(top=0.78, bottom=0.12, left=0.08, right=0.92)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Parse bandwidth/alpha from AMP-CCL log and generate Experiment 3 figure"
    )
    parser.add_argument("--dynamic-log", required=True,
                        help="Dynamic alpha log file (nccl_allgather_with_pcie_*.log)")
    parser.add_argument("--static-log", required=False,
                        help="Static baseline log (static.log), optional; if omitted only draws dynamic")
    parser.add_argument("--alpha-static", type=float, default=0.6,
                        help="Static algorithm alpha (constant for entire line), default 0.6")
    parser.add_argument("--output-fig", required=False,
                        help="Output figure path; default generated from dynamic log timestamp to ./figure/")

    args = parser.parse_args()

    dynamic_path = Path(args.dynamic_log).resolve()
    if not dynamic_path.is_file():
        raise FileNotFoundError(f"Dynamic log not found: {dynamic_path}")

    dyn = parse_log_file(dynamic_path)

    static_bw = None
    if args.static_log:
        static_path = Path(args.static_log).resolve()
        if not static_path.is_file():
            raise FileNotFoundError(f"Static log not found: {static_path}")
        sta = parse_log_file(static_path)
        static_bw = sta["bw"]

    # Use formal iters only, align lengths
    n_dyn = len(dyn["bw"])
    n_sta = len(static_bw) if static_bw is not None else n_dyn
    n = min(n_dyn, n_sta)
    if n == 0:
        raise RuntimeError("No formal iterations parsed from log.")

    bw_dynamic = dyn["bw"][:n]
    alpha_dynamic = dyn["alpha"][:n]

    if static_bw is not None:
        bw_static = static_bw[:n]
    else:
        # If no static log, use dynamic mean as baseline placeholder
        bw_static = np.full_like(bw_dynamic, bw_dynamic.mean())

    alpha_static = np.full_like(alpha_dynamic, args.alpha_static, dtype=float)

    iters = np.arange(1, n + 1, dtype=int)
    interference_ranges = dyn["interference_ranges"]

    if args.output_fig:
        out_path = Path(args.output_fig)
    else:
        # Extract timestamp from filename, output to ./figure/
        ts_match = re.search(r"nccl_allgather_with_pcie_(\\d{8}_\\d{6})", dynamic_path.name)
        if ts_match:
            ts = ts_match.group(1)
            out_name = f"nccl_allgather_with_pcie_{ts}.png"
        else:
            out_name = dynamic_path.stem + ".png"
        out_path = dynamic_path.parent.parent / "figure" / out_name

    plot_figure(iters, bw_static, bw_dynamic, alpha_static, alpha_dynamic,
                interference_ranges, out_path)


if __name__ == "__main__":
    main()

