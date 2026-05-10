#!/usr/bin/env python3
"""Validate and summarize the PCIeCCL x background-traffic AllGather ablation."""
import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean, pstdev

RANGES = ((100, 200), (300, 600), (700, 900))
CONDITIONS = {
    "pcie_on_bg_off": (1, 0),
    "pcie_on_bg_on": (1, 1),
    "pcie_off_bg_on": (0, 1),
    "pcie_off_bg_off": (0, 0),
}


def load_existing_parser():
    path = Path(__file__).with_name("process_results.py")
    spec = importlib.util.spec_from_file_location("inference_process_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select(points, active):
    return [p for p in points if any(a <= p[0] < b for a, b in RANGES) == active]


def stats(values):
    if not values:
        return {"count": 0, "mean": None, "std": None, "cv_percent": None}
    avg = mean(values)
    deviation = pstdev(values)
    return {"count": len(values), "mean": avg, "std": deviation,
            "cv_percent": 100.0 * deviation / avg if avg else math.nan}


def collect_log(path, condition, block, parser):
    text = path.read_text(errors="replace")
    header = text.splitlines()[0]
    fields = dict(re.findall(r"(\w+)=([^\s]+)", header))
    expected_route, expected_bg = CONDITIONS[condition]
    if int(fields.get("np", -1)) != 4 or int(fields.get("iters", -1)) != 1001:
        raise ValueError(f"{path}: expected 4 ranks and 1001 iterations")
    if int(fields.get("pcie_route_enabled", expected_route)) != expected_route:
        raise ValueError(f"{path}: PCIe route mismatch")
    if int(fields.get("background_enabled", expected_bg)) != expected_bg:
        raise ValueError(f"{path}: background setting mismatch")

    points = parser.parse_log(path)
    # Timer lines immediately precede each iteration's aggregate report. The
    # final timer pair before each iter line therefore belongs to that iter.
    fast_vals, pcie_vals, event_vals = [], [], []
    current_fast = current_pcie = None
    for line in text.splitlines():
        m = re.search(r"\(max\) fast_time=([\d.eE+-]+)s", line)
        if m:
            current_fast = float(m[1]) * 1000.0
        m = re.search(r"\(max\) pcie_time=([\d.eE+-]+)s", line)
        if m:
            current_pcie = float(m[1]) * 1000.0
        m = re.match(r"iter=(\d+) event_ms=([\d.eE+-]+)", line)
        if m:
            iteration = int(m[1])
            if iteration < len(points):
                if expected_route and (current_fast is None or current_pcie is None):
                    raise ValueError(f"{path}: missing path timers for iter {iteration}")
                if current_fast is not None:
                    fast_vals.append((iteration, current_fast))
                if current_pcie is not None:
                    pcie_vals.append((iteration, current_pcie))
                event_vals.append((iteration, float(m[2])))

    bg_lines = re.findall(r"^pcie_background (.+)$", text, re.M)
    if expected_bg:
        if len(bg_lines) != 4 * 2 * len(RANGES):
            raise ValueError(f"{path}: expected 24 background summaries, got {len(bg_lines)}")
        bg = []
        for line in bg_lines:
            values = dict(re.findall(r"(\w+)=([^\s]+)", line))
            if int(values["status"]) != 0 or int(values["attempts"]) != int(values["successes"]):
                raise ValueError(f"{path}: incomplete/failed background copy: {line}")
            bg.append(values)
        stage_rates = []
        for stage in range(len(RANGES)):
            stage_rows = [row for row in bg if int(row["stage"]) == stage]
            seconds = max(float(row["span_seconds"]) for row in stage_rows)
            bytes_total = sum(int(row["bytes"]) for row in stage_rows)
            stage_rates.append(bytes_total / seconds / 1e9)
        background_GBps = mean(stage_rates)
    else:
        if bg_lines:
            raise ValueError(f"{path}: found background workers in bg-off cell")
        background_GBps = 0.0

    active_points = select(points, True)
    inactive_points = select(points, False)
    active_ids = {p[0] for p in active_points}
    fast_active = [v for i, v in fast_vals if i in active_ids]
    pcie_active = [v for i, v in pcie_vals if i in active_ids]
    event_active = [v for i, v in event_vals if i in active_ids]
    fast_off = [v for i, v in fast_vals if i not in active_ids]
    pcie_off = [v for i, v in pcie_vals if i not in active_ids]
    event_off = [v for i, v in event_vals if i not in active_ids]
    bw_active = stats([p[2] for p in active_points])
    bw_off = stats([p[2] for p in inactive_points])
    return {
        "block": block, "condition": condition,
        "pcie_route_enabled": expected_route, "background_enabled": expected_bg,
        "bandwidth_active": bw_active, "bandwidth_off": bw_off,
        "fast_ms_active": stats(fast_active), "pcie_ms_active": stats(pcie_active),
        "event_ms_active": stats(event_active), "fast_ms_off": stats(fast_off),
        "pcie_ms_off": stats(pcie_off), "event_ms_off": stats(event_off),
        "pcie_bottleneck_share_active": (sum(p > f for p, f in zip(pcie_active, fast_active)) / len(pcie_active)
                                          if pcie_active and len(pcie_active) == len(fast_active) else None),
        "background_GBps": background_GBps,
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} LOG_ROOT")
    root = Path(sys.argv[1])
    parser = load_existing_parser()
    block_dirs = sorted((root / "blocks").glob("block_*"), key=lambda p: int(p.name.split("_")[-1]))
    if not block_dirs:
        raise SystemExit(f"No blocks found under {root / 'blocks'}")
    rows = []
    for block_dir in block_dirs:
        for condition in CONDITIONS:
            path = block_dir / condition / "static.log"
            if not path.is_file():
                raise ValueError(f"missing cell log: {path}")
            rows.append(collect_log(path, condition, block_dir.name, parser))

    csv_path = root / "factorial_summary.csv"
    columns = ["block", "condition", "pcie_route_enabled", "background_enabled",
               "bw_active_mean_GiBps", "bw_active_cv_percent", "bw_off_mean_GiBps",
               "bw_off_cv_percent", "fast_active_mean_ms", "pcie_active_mean_ms",
               "event_active_mean_ms", "pcie_bottleneck_share_active", "background_GBps"]
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "block": row["block"], "condition": row["condition"],
                "pcie_route_enabled": row["pcie_route_enabled"],
                "background_enabled": row["background_enabled"],
                "bw_active_mean_GiBps": row["bandwidth_active"]["mean"],
                "bw_active_cv_percent": row["bandwidth_active"]["cv_percent"],
                "bw_off_mean_GiBps": row["bandwidth_off"]["mean"],
                "bw_off_cv_percent": row["bandwidth_off"]["cv_percent"],
                "fast_active_mean_ms": row["fast_ms_active"]["mean"],
                "pcie_active_mean_ms": row["pcie_ms_active"]["mean"],
                "event_active_mean_ms": row["event_ms_active"]["mean"],
                "pcie_bottleneck_share_active": row["pcie_bottleneck_share_active"],
                "background_GBps": row["background_GBps"],
            })

    by_block = {row["block"]: {row2["condition"]: row2 for row2 in rows if row2["block"] == row["block"]}
                for row in rows}
    effects = []
    for block, cells in by_block.items():
        def delta(metric, left, right):
            return cells[left][metric]["mean"] - cells[right][metric]["mean"]
        effects.append({
            "block": block,
            "background_bw_effect_pcie_on_GiBps": delta("bandwidth_active", "pcie_on_bg_on", "pcie_on_bg_off"),
            "background_bw_effect_pcie_off_GiBps": delta("bandwidth_active", "pcie_off_bg_on", "pcie_off_bg_off"),
            "difference_in_differences_bw_GiBps": (
                delta("bandwidth_active", "pcie_on_bg_on", "pcie_on_bg_off") -
                delta("bandwidth_active", "pcie_off_bg_on", "pcie_off_bg_off")),
            "background_pcie_time_effect_ms": delta("pcie_ms_active", "pcie_on_bg_on", "pcie_on_bg_off"),
            "background_event_time_effect_ms_pcie_on": delta("event_ms_active", "pcie_on_bg_on", "pcie_on_bg_off"),
            "background_event_time_effect_ms_pcie_off": delta("event_ms_active", "pcie_off_bg_on", "pcie_off_bg_off"),
        })
    summary = {
        "design": "2x2 PCIeCCL route x continuous bidirectional background traffic",
        "active_windows": [list(r) for r in RANGES],
        "units": {"bandwidth": "GiB/s", "path_and_event_time": "ms", "background_rate": "aggregate decimal GB/s"},
        "rows": rows,
        "paired_effects_by_block": effects,
        "caveat": "Descriptive paired contrasts only; three blocks are not enough for a strong inferential claim.",
    }
    (root / "factorial_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({"rows": len(rows), "csv": str(csv_path),
                      "json": str(root / "factorial_summary.json"), "effects": effects}, indent=2))


if __name__ == "__main__":
    main()
