#!/usr/bin/env python3
"""Analyze raw 80-iteration logs with previous-round feedback alignment."""

import argparse
import csv
import re
from pathlib import Path
from statistics import mean, stdev, median
from PIL import Image, ImageDraw

from mpi_primitives.plot_pcie_allgather import parse_path_metrics
from plot_4card_allgather_regulation import get_font

RANGES = [(20, 40), (55, 65)]
INPUT_BYTES = 256 * 1024**2


def draw_raw(data, output):
    image = Image.new("RGB", (1500, 1540), "white")
    draw = ImageDraw.Draw(image)
    font, small = get_font(19), get_font(16)
    draw.text((100, 18), "4-card AllGather | 256 MiB input/rank | 80 iterations | raw measurements", fill="black", font=get_font(24, True))
    draw.text((100, 52), "Shaded: continuous 128 MiB D2H/rank in [20,40), [55,65). Bandwidth: 1 GiB output / time (decimal GB/s).", fill="black", font=small)
    blue, red = (30, 110, 210), (210, 55, 55)
    panels = [
        ("Enclosing ACL event bandwidth: max across ranks (80 measurements)", [(a, "event_output_gbps", c) for a, c in (("split_ratio", blue), ("tcp", red))]),
        ("Framework bandwidth: max of cross-rank path times (79 feedback records)", [(a, "algbw", c) for a, c in (("split_ratio", blue), ("tcp", red))]),
        ("PCIe split ratio alpha (previous-round feedback aligned)", [(a, "split_pcie", c) for a, c in (("split_ratio", blue), ("tcp", red))]),
        ("split_ratio: fast / PCIe / enclosing event time (ms)", [("split_ratio", k, c) for k, c in (("fast_time_ms", blue), ("pcie_time_ms", red), ("event_ms", (50, 150, 70)))]),
        ("tcp: fast / PCIe / enclosing event time (ms)", [("tcp", k, c) for k, c in (("fast_time_ms", blue), ("pcie_time_ms", red), ("event_ms", (50, 150, 70)))]),
    ]
    for index, (title, curves) in enumerate(panels):
        left, right, top, bottom = 100, 1450, 125 + index * 280, 345 + index * 280
        draw.text((left, top - 30), title, fill="black", font=font)
        values = [r[k] for a, k, c in curves for r in data[a] if r[k] is not None]
        ymax = max(values) * 1.18
        xp = lambda it: left + it / 80 * (right - left)
        yp = lambda v: bottom - v / ymax * (bottom - top)
        for a, b in RANGES:
            draw.rectangle((xp(a), top, xp(b), bottom), fill=(250, 235, 230))
        for tick in range(6):
            v = ymax * tick / 5
            draw.line((left, yp(v), right, yp(v)), fill=(225, 225, 225))
            draw.text((20, yp(v) - 8), f"{v:.2f}" if index == 2 else f"{v:.1f}", fill="black", font=small)
        for tick in range(0, 81, 10):
            draw.line((xp(tick), top, xp(tick), bottom), fill=(230, 230, 230))
            draw.text((xp(tick) - 8, bottom + 4), str(tick), fill="black", font=small)
        draw.rectangle((left, top, right, bottom), outline=(90, 90, 90))
        for ci, (algo, key, color) in enumerate(curves):
            points = [(xp(r["iter"]), yp(r[key])) for r in data[algo] if r[key] is not None]
            draw.line(points, fill=color, width=2)
            for x, y in points:
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
            label = algo if index < 3 else key
            lx = 660 + ci * 255
            draw.rectangle((lx - 5, top + 5, lx + 245, top + 28), fill="white")
            draw.line((lx, top + 17, lx + 25, top + 17), fill=color, width=3)
            draw.text((lx + 32, top + 6), label, fill=color, font=small)
    draw.text((650, 1520), "Formal iteration (0-based)", fill="black", font=small)
    image.save(output)


def read_records(path):
    text = path.read_text(errors="replace")
    assert "result: PASSED" in text and "global_mismatches=0" in text, path
    assert len(re.findall(r"rank [0-3]: PASSED", text)) == 4, path
    assert "size_bytes=134217728 ranges=[20,40) [55,65)" in text, path
    events = {
        int(it): float(ms)
        for it, ms in re.findall(r"iter=(\d+) event_ms=([\d.]+) bw_GBps=[\d.]+", text)
    }
    assert set(events) == set(range(80)), (path, sorted(events))
    paths = parse_path_metrics(path, expected_size_bytes=INPUT_BYTES, expected_ranks=4)
    assert set(paths) == set(range(79)), (path, sorted(paths))
    # The final feedback is not emitted: no following collective consumes it.
    records = []
    for it, ms in sorted(events.items()):
        p = paths.get(it)
        assert ms > 0, (path, it, ms)
        if p:
            assert p.bandwidth.total > 0 and p.timing is not None, (path, it)
        records.append({
            "iter": it,
            "algbw": p.bandwidth.total if p else None,
            "split_pcie": p.bandwidth.pcie / p.bandwidth.total if p else None,
            "fast_time_ms": p.timing.hccl_ms if p else None,
            "pcie_time_ms": p.timing.pcie_ms if p else None,
            "event_ms": ms,
            # Decimal GB/s; same output-byte numerator as the framework.
            "event_output_gbps": INPUT_BYTES * 4 / (ms / 1000) / 1e9,
            "is_contention": any(a <= it < b for a, b in RANGES),
        })
    return records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    args = parser.parse_args()
    args.image_dir.mkdir(parents=True, exist_ok=True)
    data = {algo: read_records(args.log_root / f"{algo}.log") for algo in ("split_ratio", "tcp")}
    with (args.log_root / "iterations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["algorithm"] + list(data["tcp"][0]))
        writer.writeheader()
        for algo, records in data.items():
            writer.writerows(dict(algorithm=algo, **r) for r in records)

    draw_raw(data, args.image_dir / "bandwidth_alpha_timings_raw.png")
    phases = {
        "all": lambda i: True,
        "normal": lambda i: not any(a <= i < b for a, b in RANGES),
        "contention": lambda i: any(a <= i < b for a, b in RANGES),
        "normal_0_20": lambda i: i < 20,
        "contention_20_40": lambda i: 20 <= i < 40,
        "recovery_40_55": lambda i: 40 <= i < 55,
        "contention_55_65": lambda i: 55 <= i < 65,
        "recovery_65_80": lambda i: i >= 65,
    }
    lines = ["# 4-card AllGather raw stability results", "",
             "Input: 256 MiB/rank; output: 1 GiB/rank; FP32; 80 formal + 5 warmup iterations.",
             "Continuous 128 MiB D2H interference/rank in [20,40), [55,65). No denoising.",
             "Both bandwidth metrics use 1 GiB output / time / 1e9 (decimal GB/s).",
             "Framework uses cross-rank max path time; event uses cross-rank max enclosing ACL event time.",
             "Feedback is emitted on the following collective: warmup excluded; framework has iters 0–78, event has 0–79.",
             "Event figure, CSV and statistics retain all 80 events. Path/alpha curves stop at 78.", "",
             "| Algorithm | Metric | Phase | N | Mean GB/s | Std | CV % | Min | Median | Dips <80% phase median |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    stats_rows = []
    for algo, rows in data.items():
        for metric in ("algbw", "event_output_gbps"):
            for phase, include in phases.items():
                values = [r[metric] for r in rows if include(r["iter"]) and r[metric] is not None]
                avg, sd, med = mean(values), stdev(values), median(values)
                dips = sum(v < 0.8 * med for v in values)
                stats_rows.append(dict(algorithm=algo, metric=metric, phase=phase, n=len(values),
                                       mean=avg, std=sd, cv_pct=100 * sd / avg,
                                       minimum=min(values), median=med, dips=dips))
                lines.append(f"| {algo} | {metric} | {phase} | {len(values)} | {avg:.2f} | {sd:.2f} | {100*sd/avg:.2f} | {min(values):.2f} | {med:.2f} | {dips} |")
    with (args.log_root / "statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats_rows[0]))
        writer.writeheader()
        writer.writerows(stats_rows)
    (args.log_root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
