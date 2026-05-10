import argparse
import os
import re
from pathlib import Path

from process_log_and_plot import parse_log_file, plot_figure
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Batch process nccl_allgather_with_pcie_*.log in logs dir, generate figures to ./figure/"
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Log directory (default ./logs)",
    )
    parser.add_argument(
        "--static-log",
        default="static.log",
        help="Static baseline log filename, relative to logs-dir (default static.log)",
    )
    parser.add_argument(
        "--alpha-static",
        type=float,
        default=0.6,
        help="Static algorithm alpha (constant for entire line), default 0.6",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    logs_dir = (base_dir / args.logs_dir).resolve()
    if not logs_dir.is_dir():
        raise NotADirectoryError(f"logs dir not found: {logs_dir}")

    fig_dir = (base_dir / "figure")
    fig_dir.mkdir(parents=True, exist_ok=True)

    static_path = logs_dir / args.static_log
    static_data = None
    if static_path.is_file():
        print(f"Parsing static baseline log: {static_path}")
        static_data = parse_log_file(static_path)
    else:
        print(f"Warning: static log not found: {static_path} (only dynamic curves will be drawn)")

    pattern = re.compile(r"nccl_allgather_with_pcie_(\d{8}_\d{6})\.log")

    for entry in sorted(logs_dir.glob("nccl_allgather_with_pcie_*.log")):
        m = pattern.match(entry.name)
        if not m:
            continue
        ts = m.group(1)
        print(f"Processing dynamic log: {entry.name}")

        dyn = parse_log_file(entry)

        n_dyn = len(dyn["bw"])
        if n_dyn == 0:
            print(f"  Skip {entry.name}: no formal iterations parsed.")
            continue

        if static_data is not None:
            n_sta = len(static_data["bw"])
            n = min(n_dyn, n_sta)
            bw_static = static_data["bw"][:n]
        else:
            n = n_dyn
            bw_static = np.full(n, dyn["bw"].mean())

        bw_dynamic = dyn["bw"][:n]
        alpha_dynamic = dyn["alpha"][:n]
        alpha_static = np.full_like(alpha_dynamic, args.alpha_static, dtype=float)
        iters = np.arange(1, n + 1, dtype=int)
        interference_ranges = dyn["interference_ranges"]

        out_path = fig_dir / f"nccl_allgather_with_pcie_{ts}.png"
        plot_figure(iters, bw_static, bw_dynamic, alpha_static, alpha_dynamic,
                    interference_ranges, out_path)


if __name__ == "__main__":
    main()

