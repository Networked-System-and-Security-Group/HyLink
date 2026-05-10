#!/usr/bin/env python3
"""Parse AMPCCL AllGather algbw logs and draw a PCIe-interference plot."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple


Range = Tuple[int, int]

_BEGIN_RE = re.compile(r"\[rank0\]\s+begin formal\s+(\d+)\s+iters")
_AGG_RE = re.compile(
    r"aggregate\s+op=AllGather\s+"
    r"op_key_bytes=(?P<size>\d+).*?\bn=(?P<ranks>\d+)\b.*?"
    r"algbw\(GB/s,decimal\)=(?P<bw>[0-9.eE+-]+)"
)
_PATH_AGG_RE = re.compile(
    r"aggregate\s+op=AllGather\s+"
    r"op_key_bytes=(?P<size>\d+).*?\bn=(?P<ranks>\d+)\b.*?"
    r"split_fast=(?P<split_fast>[0-9.eE+-]+)\s+"
    r"split_pcie=(?P<split_pcie>[0-9.eE+-]+).*?"
    r"fast_bw\(GB/s,decimal\)=(?P<hccl>[0-9.eE+-]+).*?"
    r"pcie_bw\(GB/s,decimal\)=(?P<pcie>[0-9.eE+-]+).*?"
    r"algbw\(GB/s,decimal\)=(?P<total>[0-9.eE+-]+)"
)
_FAST_TIME_RE = re.compile(r"\((?:max|avg)\)\s+fast_time=(?P<time>[0-9.eE+-]+)s")
_PCIE_TIME_RE = re.compile(r"\((?:max|avg)\)\s+pcie_time=(?P<time>[0-9.eE+-]+)s")


@dataclass(frozen=True)
class PathBandwidth:
    """AllGather algorithm bandwidth of both paths and the aggregate."""

    hccl: float
    pcie: float
    total: float


@dataclass(frozen=True)
class PathTiming:
    """Path elapsed times in milliseconds."""

    hccl_ms: float
    pcie_ms: float


@dataclass(frozen=True)
class PathMetrics:
    bandwidth: PathBandwidth
    timing: PathTiming | None


def parse_ranges(spec: str) -> List[Range]:
    """Parse inclusive/exclusive range text such as ``10-20,30-40``."""

    ranges: List[Range] = []
    for item in spec.split(","):
        item = item.strip()
        if not item or "-" not in item:
            raise ValueError(f"invalid range: {item!r}")
        start_text, end_text = item.split("-", 1)
        try:
            start = int(start_text)
            end = int(end_text)
        except ValueError as exc:
            raise ValueError(f"invalid range: {item!r}") from exc
        if start < 0 or end <= start:
            raise ValueError(f"range must satisfy 0 <= start < end: {item!r}")
        ranges.append((start, end))
    if not ranges:
        raise ValueError("at least one range is required")
    return ranges


def active_iterations(num_iters: int, ranges: Sequence[Range]) -> List[int]:
    """Return formal iter indices covered by left-closed/right-open ranges."""

    return [
        iteration
        for iteration in range(num_iters)
        if any(start <= iteration < end for start, end in ranges)
    ]


def parse_log_text(
    text: str,
    expected_size_bytes: int | None = None,
    expected_ranks: int | None = None,
) -> Dict[int, float]:
    """Return ``formal_iter -> algbw`` from previous-round aggregate log lines.

    AMPCCL emits the aggregate result when the next collective begins.  The
    aggregate line following ``begin formal k`` therefore belongs to formal
    iteration ``k - 1``; the record after ``begin formal 0`` is warmup data.
    """

    current_formal: int | None = None
    bandwidths: Dict[int, float] = {}
    for line in text.splitlines():
        begin = _BEGIN_RE.search(line)
        if begin:
            current_formal = int(begin.group(1))
            continue

        aggregate = _AGG_RE.search(line)
        if aggregate is None or current_formal is None or current_formal == 0:
            continue

        if expected_size_bytes is not None and int(aggregate.group("size")) != expected_size_bytes:
            continue
        if expected_ranks is not None and int(aggregate.group("ranks")) != expected_ranks:
            continue

        bandwidths[current_formal - 1] = float(aggregate.group("bw"))
    return dict(sorted(bandwidths.items()))


def parse_log(
    path: Path,
    expected_size_bytes: int | None = None,
    expected_ranks: int | None = None,
) -> Dict[int, float]:
    return parse_log_text(
        path.read_text(encoding="utf-8", errors="replace"),
        expected_size_bytes=expected_size_bytes,
        expected_ranks=expected_ranks,
    )


def parse_path_metrics_text(
    text: str,
    expected_size_bytes: int | None = None,
    expected_ranks: int | None = None,
) -> Dict[int, PathMetrics]:
    """Extract path algorithm bandwidths and times using the previous-round mapping.

    AMPCCL prints ``fast_time``/``pcie_time`` immediately before the aggregate
    line.  Both belong to the same previous formal iteration as that aggregate
    line, so they are stored together as one record.  The raw ``fast_bw`` and
    ``pcie_bw`` fields are per-path byte rates, not algorithm bandwidths.  The
    plotted path values use the same collective-time denominator as the
    aggregate algbw, so they are the algorithm-bandwidth contributions
    ``total_algbw * split_fast`` and ``total_algbw * split_pcie``.
    """

    current_formal: int | None = None
    pending_hccl_time_s: float | None = None
    pending_pcie_time_s: float | None = None
    metrics: Dict[int, PathMetrics] = {}

    for line in text.splitlines():
        begin = _BEGIN_RE.search(line)
        if begin:
            current_formal = int(begin.group(1))
            pending_hccl_time_s = None
            pending_pcie_time_s = None
            continue

        fast_time = _FAST_TIME_RE.search(line)
        if fast_time:
            pending_hccl_time_s = float(fast_time.group("time"))
            continue

        pcie_time = _PCIE_TIME_RE.search(line)
        if pcie_time:
            pending_pcie_time_s = float(pcie_time.group("time"))
            continue

        aggregate = _PATH_AGG_RE.search(line)
        if aggregate is None or current_formal is None or current_formal == 0:
            continue

        if expected_size_bytes is not None and int(aggregate.group("size")) != expected_size_bytes:
            continue
        if expected_ranks is not None and int(aggregate.group("ranks")) != expected_ranks:
            continue

        timing = None
        if pending_hccl_time_s is not None and pending_pcie_time_s is not None:
            timing = PathTiming(
                hccl_ms=pending_hccl_time_s * 1000.0,
                pcie_ms=pending_pcie_time_s * 1000.0,
            )
        total_algbw = float(aggregate.group("total"))
        metrics[current_formal - 1] = PathMetrics(
            bandwidth=PathBandwidth(
                hccl=total_algbw * float(aggregate.group("split_fast")),
                pcie=total_algbw * float(aggregate.group("split_pcie")),
                total=total_algbw,
            ),
            timing=timing,
        )
        pending_hccl_time_s = None
        pending_pcie_time_s = None

    return dict(sorted(metrics.items()))


def parse_path_metrics(
    path: Path,
    expected_size_bytes: int | None = None,
    expected_ranks: int | None = None,
) -> Dict[int, PathMetrics]:
    return parse_path_metrics_text(
        path.read_text(encoding="utf-8", errors="replace"),
        expected_size_bytes=expected_size_bytes,
        expected_ranks=expected_ranks,
    )


def parse_path_bandwidths_text(
    text: str,
    expected_size_bytes: int | None = None,
    expected_ranks: int | None = None,
) -> Dict[int, PathBandwidth]:
    return {
        iteration: metric.bandwidth
        for iteration, metric in parse_path_metrics_text(
            text, expected_size_bytes=expected_size_bytes, expected_ranks=expected_ranks
        ).items()
    }


def parse_path_timings_text(
    text: str,
    expected_size_bytes: int | None = None,
    expected_ranks: int | None = None,
) -> Dict[int, PathTiming]:
    return {
        iteration: metric.timing
        for iteration, metric in parse_path_metrics_text(
            text, expected_size_bytes=expected_size_bytes, expected_ranks=expected_ranks
        ).items()
        if metric.timing is not None
    }


def write_csv(output: Path, static: Dict[int, float], tcp: Dict[int, float]) -> None:
    iterations = sorted(set(static) | set(tcp))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["formal_iter", "static_algbw_gbps", "tcp_algbw_gbps"])
        for iteration in iterations:
            writer.writerow([
                iteration,
                f"{static[iteration]:.6f}" if iteration in static else "",
                f"{tcp[iteration]:.6f}" if iteration in tcp else "",
            ])


def write_path_metrics_csv(output: Path, metrics: Dict[int, PathMetrics]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "formal_iter",
            "hccl_path_algbw_gbps",
            "pcie_path_algbw_gbps",
            "total_algbw_gbps",
            "hccl_path_time_ms",
            "pcie_path_time_ms",
        ])
        for iteration in sorted(metrics):
            metric = metrics[iteration]
            writer.writerow([
                iteration,
                f"{metric.bandwidth.hccl:.6f}",
                f"{metric.bandwidth.pcie:.6f}",
                f"{metric.bandwidth.total:.6f}",
                f"{metric.timing.hccl_ms:.6f}" if metric.timing is not None else "",
                f"{metric.timing.pcie_ms:.6f}" if metric.timing is not None else "",
            ])


def choose_tick_step(iterations: Sequence[int]) -> int:
    """Choose readable x-axis spacing for the Pillow fallback chart."""

    if len(iterations) <= 10:
        return 1
    span = max(iterations) - min(iterations)
    if span <= 100:
        return 10
    if span <= 500:
        return 50
    return 100


def bandwidth_grid_tick_count() -> int:
    """Return the number of horizontal major grid lines for bandwidth plots."""

    return 11


def reference_style_font_sizes() -> Tuple[int, int, int, int]:
    """Return large title, axis-label, tick, and legend sizes for the reference style."""

    return 17, 31, 23, 28


def reference_style_x_label_size() -> int:
    """Return the reference-style font size for the x-axis label."""

    return 33


def reference_style_y_label_position() -> Tuple[float, float]:
    """Return the axes-relative position for the centered y-axis label."""

    return -0.1, 0.5


def _set_bandwidth_grid_locator(ax) -> None:
    from matplotlib.ticker import LinearLocator

    ax.yaxis.set_major_locator(LinearLocator(numticks=bandwidth_grid_tick_count()))


def _pillow_font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
        if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _plot_single_series_with_pillow(
    iterations: Sequence[int],
    values: Sequence[float],
    output: Path,
    title: str,
    y_label: str,
    legend_label: str,
) -> None:
    """Draw one metric series with Pillow when matplotlib is unavailable."""

    from PIL import Image, ImageDraw

    width, height = 1200, 720
    left, top, right, bottom = 105, 105, 1165, 635
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _pillow_font(22, bold=True)
    label_font = _pillow_font(17, bold=True)
    tick_font = _pillow_font(14)
    legend_font = _pillow_font(15)

    x_min, x_max = min(iterations), max(iterations)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    y_max = max(values) * 1.18
    y_max = y_max if y_max > 0 else 1.0

    def x_coord(value: float) -> int:
        return round(left + (value - x_min) / (x_max - x_min) * (right - left))

    def y_coord(value: float) -> int:
        return round(bottom - value / y_max * (bottom - top))

    tick_count = bandwidth_grid_tick_count()
    for tick in range(tick_count):
        value = y_max * tick / (tick_count - 1)
        y = y_coord(value)
        draw.line((left, y, right, y), fill=(218, 218, 218), width=1)
        label = f"{value:.2f}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((left - 12 - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2),
                  label, fill=(25, 25, 25), font=tick_font)
    draw.line((left, top, left, bottom), fill=(20, 20, 20), width=2)
    draw.line((left, bottom, right, bottom), fill=(20, 20, 20), width=2)

    tick_step = choose_tick_step(iterations)
    x_ticks = list(range(((iterations[0] + tick_step - 1) // tick_step) * tick_step,
                         iterations[-1] + 1, tick_step))
    if iterations[-1] not in x_ticks:
        x_ticks.append(iterations[-1])
    for tick in sorted(set(x_ticks)):
        x = x_coord(tick)
        draw.line((x, bottom, x, bottom + 6), fill=(20, 20, 20), width=1)
        label = str(tick)
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, bottom + 10), label,
                  fill=(25, 25, 25), font=tick_font)

    points = [(x_coord(iteration), y_coord(value))
              for iteration, value in zip(iterations, values)]
    if len(points) >= 2:
        draw.line(points, fill=(226, 138, 0), width=3, joint="curve")

    def centered(text: str, y: int, font, fill=(20, 20, 20)) -> None:
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text(((width - (bbox[2] - bbox[0])) / 2, y), text, fill=fill, font=font)

    centered(title, 20, title_font)
    centered("Collective Iteration", 670, label_font)
    legend_y = 70
    draw.line((left, legend_y + 8, left + 30, legend_y + 8), fill=(226, 138, 0), width=3)
    draw.text((left + 40, legend_y - 3), legend_label, fill=(20, 20, 20), font=legend_font)
    y_label_image = Image.new("RGBA", (300, 34), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label_image)
    y_draw.text((0, 0), y_label, fill=(20, 20, 20), font=label_font)
    y_label_image = y_label_image.rotate(90, expand=True)
    image.paste(y_label_image, (18, 270), y_label_image)
    image.save(output, format="PNG")


def _plot_path_bandwidths_with_pillow(
    metrics: Dict[int, PathMetrics],
    output: Path,
    title: str,
    reference: Dict[int, float] | None = None,
    reference_label: str = "Pure HCCL algbw",
) -> None:
    """Draw path/total AllGather algbw on one common scale."""

    from PIL import Image, ImageDraw

    iterations = sorted(metrics)
    reference_iterations = (
        sorted(set(iterations) & set(reference)) if reference is not None else []
    )
    if reference is not None and not reference_iterations:
        raise RuntimeError("no common iterations between path metrics and reference")
    width, height = 1200, 720
    left, top, right, bottom = 105, 105, 1095, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _pillow_font(22, bold=True)
    label_font = _pillow_font(16, bold=True)
    tick_font = _pillow_font(13)
    legend_font = _pillow_font(14)

    x_min, x_max = min(iterations), max(iterations)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    y_max = max(
        max(metrics[i].bandwidth.total for i in iterations),
        max(metrics[i].bandwidth.hccl for i in iterations),
        max(metrics[i].bandwidth.pcie for i in iterations),
        max(reference[i] for i in reference_iterations) if reference_iterations else 0.0,
    ) * 1.18
    y_max = y_max if y_max > 0 else 1.0

    def x_coord(value: float) -> int:
        return round(left + (value - x_min) / (x_max - x_min) * (right - left))

    def y_coord(value: float) -> int:
        return round(bottom - value / y_max * (bottom - top))

    tick_count = bandwidth_grid_tick_count()
    for tick in range(tick_count):
        value = y_max * tick / (tick_count - 1)
        y = y_coord(value)
        draw.line((left, y, right, y), fill=(218, 218, 218), width=1)
        label = f"{value:.0f}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((left - 12 - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2),
                  label, fill=(25, 25, 25), font=tick_font)

    draw.line((left, top, left, bottom), fill=(20, 20, 20), width=2)
    draw.line((left, bottom, right, bottom), fill=(20, 20, 20), width=2)

    tick_step = choose_tick_step(iterations)
    x_ticks = list(range(((iterations[0] + tick_step - 1) // tick_step) * tick_step,
                         iterations[-1] + 1, tick_step))
    if iterations[-1] not in x_ticks:
        x_ticks.append(iterations[-1])
    for tick in sorted(set(x_ticks)):
        x = x_coord(tick)
        draw.line((x, bottom, x, bottom + 6), fill=(20, 20, 20), width=1)
        label = str(tick)
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, bottom + 10), label,
                  fill=(25, 25, 25), font=tick_font)

    def line(values, y_coord, color, line_iterations):
        points = [(x_coord(i), y_coord(values(i))) for i in line_iterations]
        if len(points) >= 2:
            draw.line(points, fill=color, width=3, joint="curve")

    line(lambda i: metrics[i].bandwidth.total, y_coord, (8, 120, 181), iterations)
    line(lambda i: metrics[i].bandwidth.hccl, y_coord, (226, 138, 0), iterations)
    line(lambda i: metrics[i].bandwidth.pcie, y_coord, (44, 160, 44), iterations)
    if reference_iterations:
        line(lambda i: reference[i], y_coord, (123, 63, 172), reference_iterations)

    def centered(text: str, y: int, font, fill=(20, 20, 20)) -> None:
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text(((width - (bbox[2] - bbox[0])) / 2, y), text, fill=fill, font=font)

    centered(title, 20, title_font)
    centered("Collective Iteration", 665, label_font)
    legend_y = 70
    legend_items = [
        ((8, 120, 181), "Total algbw"),
        ((226, 138, 0), "HCCL path algbw"),
        ((44, 160, 44), "PCIe path algbw"),
    ]
    if reference is not None:
        legend_items.append(((123, 63, 172), reference_label))
    legend_x = left
    for color, label in legend_items:
        draw.line((legend_x, legend_y + 8, legend_x + 28, legend_y + 8), fill=color, width=3)
        draw.text((legend_x + 36, legend_y - 3), label, fill=(20, 20, 20), font=legend_font)
        legend_x += 270

    y_label = Image.new("RGBA", (300, 30), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text((0, 0), "Algorithm bandwidth (GB/s)", fill=(20, 20, 20), font=label_font)
    y_label = y_label.rotate(90, expand=True)
    image.paste(y_label, (18, 280), y_label)
    image.save(output, format="PNG")


def _plot_path_timings_with_pillow(
    timings: Dict[int, PathTiming], output: Path, title: str
) -> None:
    """Draw HCCL/PCIe path times in milliseconds without matplotlib."""

    from PIL import Image, ImageDraw

    iterations = sorted(timings)
    width, height = 1200, 720
    left, top, right, bottom = 105, 105, 1165, 635
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _pillow_font(22, bold=True)
    label_font = _pillow_font(17, bold=True)
    tick_font = _pillow_font(14)
    legend_font = _pillow_font(15)
    x_min, x_max = min(iterations), max(iterations)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    y_max = max(max(timings[i].hccl_ms for i in iterations),
                max(timings[i].pcie_ms for i in iterations)) * 1.18
    y_max = y_max if y_max > 0 else 1.0

    def x_coord(value: float) -> int:
        return round(left + (value - x_min) / (x_max - x_min) * (right - left))

    def y_coord(value: float) -> int:
        return round(bottom - value / y_max * (bottom - top))

    for tick in range(6):
        value = y_max * tick / 5.0
        y = y_coord(value)
        draw.line((left, y, right, y), fill=(218, 218, 218), width=1)
        label = f"{value:.2f}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((left - 12 - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2),
                  label, fill=(25, 25, 25), font=tick_font)
    draw.line((left, top, left, bottom), fill=(20, 20, 20), width=2)
    draw.line((left, bottom, right, bottom), fill=(20, 20, 20), width=2)

    tick_step = choose_tick_step(iterations)
    x_ticks = list(range(((iterations[0] + tick_step - 1) // tick_step) * tick_step,
                         iterations[-1] + 1, tick_step))
    if iterations[-1] not in x_ticks:
        x_ticks.append(iterations[-1])
    for tick in sorted(set(x_ticks)):
        x = x_coord(tick)
        draw.line((x, bottom, x, bottom + 6), fill=(20, 20, 20), width=1)
        label = str(tick)
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, bottom + 10), label,
                  fill=(25, 25, 25), font=tick_font)

    def draw_series(values, color):
        points = [(x_coord(i), y_coord(values(i))) for i in iterations]
        if len(points) >= 2:
            draw.line(points, fill=color, width=3, joint="curve")

    draw_series(lambda i: timings[i].hccl_ms, (226, 138, 0))
    draw_series(lambda i: timings[i].pcie_ms, (44, 160, 44))

    def centered(text: str, y: int, font, fill=(20, 20, 20)) -> None:
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text(((width - (bbox[2] - bbox[0])) / 2, y), text, fill=fill, font=font)

    centered(title, 20, title_font)
    centered("Collective Iteration", 670, label_font)
    legend_y = 70
    draw.line((left, legend_y + 8, left + 30, legend_y + 8), fill=(226, 138, 0), width=3)
    draw.text((left + 40, legend_y - 3), "HCCL path fast_time", fill=(20, 20, 20), font=legend_font)
    legend_x = left + 330
    draw.line((legend_x, legend_y + 8, legend_x + 30, legend_y + 8), fill=(44, 160, 44), width=3)
    draw.text((legend_x + 40, legend_y - 3), "PCIe path pcie_time", fill=(20, 20, 20), font=legend_font)
    y_label = Image.new("RGBA", (300, 34), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text((0, 0), "Path time (ms)", fill=(20, 20, 20), font=label_font)
    y_label = y_label.rotate(90, expand=True)
    image.paste(y_label, (18, 270), y_label)
    image.save(output, format="PNG")


def plot_path_bandwidths(
    metrics: Dict[int, PathMetrics] | Dict[int, PathBandwidth],
    output: Path,
    title: str,
    reference: Dict[int, float] | None = None,
    reference_label: str = "Pure HCCL algbw",
) -> None:
    """Plot total and per-path AllGather algbw on one common scale."""

    if not metrics:
        raise RuntimeError("no path bandwidth records found")
    normalized = {
        iteration: metric if isinstance(metric, PathMetrics)
        else PathMetrics(bandwidth=metric, timing=None)
        for iteration, metric in metrics.items()
    }
    iterations = sorted(normalized)
    reference_iterations = (
        sorted(set(iterations) & set(reference)) if reference is not None else []
    )
    if reference is not None and not reference_iterations:
        raise RuntimeError("no common iterations between path metrics and reference")
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
    except ImportError:
        _plot_path_bandwidths_with_pillow(
            normalized, output, title, reference, reference_label
        )
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=180)
    total_line, = ax.plot(
        iterations, [normalized[i].bandwidth.total for i in iterations],
        color="#0878b5", linewidth=1.7, label="Total algbw",
    )
    hccl_line, = ax.plot(
        iterations, [normalized[i].bandwidth.hccl for i in iterations],
        color="#e28a00", linewidth=1.5, label="HCCL path algbw",
    )
    pcie_line, = ax.plot(
        iterations, [normalized[i].bandwidth.pcie for i in iterations],
        color="#2ca02c", linewidth=1.5, label="PCIe path algbw",
    )
    reference_line = None
    if reference_iterations:
        reference_line, = ax.plot(
            reference_iterations, [reference[i] for i in reference_iterations],
            color="#7b3fad", linewidth=1.7, label=reference_label,
        )
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Collective Iteration", fontsize=12, fontweight="bold")
    ax.set_ylabel("Algorithm bandwidth (GB/s)", fontsize=12, fontweight="bold")
    _set_bandwidth_grid_locator(ax)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_xlim(min(iterations) - 0.5, max(iterations) + 0.5)
    handles = [total_line, hccl_line, pcie_line]
    labels = ["Total algbw", "HCCL path algbw", "PCIe path algbw"]
    if reference_line is not None:
        handles.append(reference_line)
        labels.append(reference_label)
    ax.legend(
        handles,
        labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=len(handles),
        frameon=False, fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_path_timings(
    timings: Dict[int, PathTiming], output: Path, title: str
) -> None:
    """Plot HCCL and PCIe path times in milliseconds."""

    if not timings:
        raise RuntimeError("no path timing records found")
    iterations = sorted(timings)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
    except ImportError:
        _plot_path_timings_with_pillow(timings, output, title)
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=180)
    ax.plot(iterations, [timings[i].hccl_ms for i in iterations], color="#e28a00",
            linewidth=1.7, label="HCCL path fast_time")
    ax.plot(iterations, [timings[i].pcie_ms for i in iterations], color="#2ca02c",
            linewidth=1.7, label="PCIe path pcie_time")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Collective Iteration", fontsize=12, fontweight="bold")
    ax.set_ylabel("Path time (ms)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_xlim(min(iterations) - 0.5, max(iterations) + 0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2,
              frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_hccl_bandwidths(
    metrics: Dict[int, PathMetrics] | Dict[int, PathBandwidth],
    output: Path,
    title: str,
) -> None:
    """Plot only the HCCL/fast-path AllGather algbw from a path log."""

    if not metrics:
        raise RuntimeError("no HCCL bandwidth records found")
    iterations = sorted(metrics)
    values = [
        (metrics[i].bandwidth.hccl if isinstance(metrics[i], PathMetrics)
         else metrics[i].hccl)
        for i in iterations
    ]
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
    except ImportError:
        _plot_single_series_with_pillow(
            iterations, values, output, title,
            "HCCL path algbw (GB/s)", "HCCL path algbw",
        )
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=180)
    ax.plot(iterations, values, color="#e28a00", linewidth=1.7,
            label="HCCL path algbw")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Collective Iteration", fontsize=12, fontweight="bold")
    ax.set_ylabel("HCCL path algbw (GB/s)", fontsize=12, fontweight="bold")
    _set_bandwidth_grid_locator(ax)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_xlim(min(iterations) - 0.5, max(iterations) + 0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=1,
              frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_hccl_timings(timings: Dict[int, PathTiming], output: Path, title: str) -> None:
    """Plot only the HCCL/fast-path time in milliseconds."""

    if not timings:
        raise RuntimeError("no HCCL timing records found")
    iterations = sorted(timings)
    values = [timings[i].hccl_ms for i in iterations]
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
    except ImportError:
        _plot_single_series_with_pillow(
            iterations, values, output, title,
            "HCCL path time (ms)", "HCCL path fast_time",
        )
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=180)
    ax.plot(iterations, values, color="#e28a00", linewidth=1.7,
            label="HCCL path fast_time")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Collective Iteration", fontsize=12, fontweight="bold")
    ax.set_ylabel("HCCL path time (ms)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_xlim(min(iterations) - 0.5, max(iterations) + 0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=1,
              frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plot_with_pillow(
    iterations: Sequence[int],
    static: Dict[int, float],
    tcp: Dict[int, float],
    ranges: Sequence[Range],
    output: Path,
    title: str,
    static_alpha: str,
    first_label: str,
    second_label: str,
    y_min: float,
    y_max: float | None,
    points_only: bool,
    paper: bool,
    line_width: float,
    reference_style: bool,
    axis_break: bool,
    window_envelope: bool,
    envelope_window: int,
    envelope_sigma: float,
) -> None:
    """Draw the same compact chart without requiring matplotlib."""

    from PIL import Image, ImageDraw, ImageFont

    width, height = (1050, 700) if paper else (1200, 720)
    if reference_style:
        left, top, right, bottom = 92, 135, 1010, 605
    elif paper:
        left, top, right, bottom = 92, 92, 1010, 605
    else:
        left, top, right, bottom = 105, 105, 1165, 635
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def font(size: int, bold: bool = False):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
            if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
        return ImageFont.load_default()

    if reference_style:
        title_size, label_size, tick_size, legend_size = reference_style_font_sizes()
        x_label_size = reference_style_x_label_size()
    elif paper:
        title_size, label_size, tick_size, legend_size = 17, 14, 11, 11
        x_label_size = label_size
    else:
        title_size, label_size, tick_size, legend_size = 22, 17, 14, 15
        x_label_size = label_size
    title_font = font(title_size, bold=True)
    label_font = font(label_size, bold=True)
    x_label_font = font(x_label_size, bold=True)
    tick_font = font(tick_size)
    legend_font = font(legend_size)

    x_min = min(iterations)
    x_max = max(iterations)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if reference_style and x_max >= 900:
        x_min, x_max = 0, 1000
    if y_max is None:
        if window_envelope:
            static_center, static_stddev = rolling_window_stats(static, envelope_window)
            tcp_center, tcp_stddev = rolling_window_stats(tcp, envelope_window)
        else:
            static_center, tcp_center = static, tcp
            static_stddev = tcp_stddev = {}
        y_data_max = max(
            max(static_center[i] + envelope_sigma * static_stddev.get(i, 0.0)
                for i in iterations),
            max(tcp_center[i] + envelope_sigma * tcp_stddev.get(i, 0.0)
                for i in iterations),
        )
        padding = 0.05 if paper else 0.18
        y_max = max(y_min + 1.0, y_data_max * (1.0 + padding))
    elif y_max <= y_min:
        raise ValueError("y_max must be greater than y_min")

    if window_envelope:
        static_center, static_stddev = rolling_window_stats(static, envelope_window)
        tcp_center, tcp_stddev = rolling_window_stats(tcp, envelope_window)
    else:
        static_center, tcp_center = static, tcp
        static_stddev = tcp_stddev = {}

    def x_coord(value: float) -> int:
        return round(left + (value - x_min) / (x_max - x_min) * (right - left))

    def y_coord(value: float) -> int:
        clipped = min(y_max, max(y_min, value))
        return round(bottom - (clipped - y_min) / (y_max - y_min) * (bottom - top))

    def draw_envelope(
        center: Dict[int, float],
        stddev: Dict[int, float],
        fill: Tuple[int, int, int],
    ) -> None:
        upper = [
            (x_coord(iteration), y_coord(center[iteration] + envelope_sigma * stddev[iteration]))
            for iteration in iterations
        ]
        lower = [
            (x_coord(iteration), y_coord(center[iteration] - envelope_sigma * stddev[iteration]))
            for iteration in reversed(iterations)
        ]
        draw.polygon(upper + lower, fill=fill)

    for start, end in ranges:
        shade_left = max(left, x_coord(start - 0.5))
        shade_right = min(right, x_coord(end - 0.5))
        if shade_right >= shade_left:
            draw.rectangle((shade_left, top, shade_right, bottom), fill=(250, 228, 230))

    if window_envelope:
        draw_envelope(static_center, static_stddev, (202, 231, 242))
        draw_envelope(tcp_center, tcp_stddev, (252, 226, 179))

    tick_count = bandwidth_grid_tick_count()
    for tick in range(tick_count):
        value = y_min + (y_max - y_min) * tick / (tick_count - 1)
        y = y_coord(value)
        draw.line((left, y, right, y), fill=(218, 218, 218), width=1)
        label = f"{value:.0f}"
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((left - 12 - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2),
                  label, fill=(25, 25, 25), font=tick_font)

    draw.line((left, top, left, bottom), fill=(20, 20, 20), width=2)
    draw.line((left, bottom, right, bottom), fill=(20, 20, 20), width=2)
    if reference_style:
        draw.line((left, top, right, top), fill=(20, 20, 20), width=2)
        draw.line((right, top, right, bottom), fill=(20, 20, 20), width=2)
    if axis_break:
        draw.line((left - 8, bottom - 8, left - 2, bottom + 2), fill=(20, 20, 20), width=2)
        draw.line((left + 2, bottom - 8, left + 8, bottom + 2), fill=(20, 20, 20), width=2)

    if reference_style and iterations[-1] >= 900:
        x_ticks = [1, 200, 400, 600, 800, 1000]
    else:
        tick_step = choose_tick_step(iterations)
        x_ticks = list(range(((iterations[0] + tick_step - 1) // tick_step) * tick_step,
                             iterations[-1] + 1, tick_step))
        if iterations[-1] not in x_ticks:
            x_ticks.append(iterations[-1])
    for tick in sorted(set(x_ticks)):
        x = x_coord(tick)
        draw.line((x, bottom, x, bottom + 6), fill=(20, 20, 20), width=1)
        label = str(tick)
        bbox = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, bottom + 10), label,
                  fill=(25, 25, 25), font=tick_font)

    def draw_series(values: Dict[int, float], color: Tuple[int, int, int]) -> None:
        points = [(x_coord(i), y_coord(values[i])) for i in iterations]
        if points_only:
            radius = 2 if paper else 3
            for x, y in points:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        elif len(points) >= 2:
            draw.line(points, fill=color, width=max(1, round(line_width * 2)), joint="curve")

    draw_series(static_center, (8, 120, 181))
    draw_series(tcp_center, (226, 138, 0))

    def centered_text(text: str, y: int, text_font, fill=(20, 20, 20)) -> None:
        bbox = draw.textbbox((0, 0), text, font=text_font)
        draw.text(((width - (bbox[2] - bbox[0])) / 2, y), text, fill=fill, font=text_font)

    centered_text(title, 20, title_font)
    centered_text("Collective Iteration", 638 if paper else 670, x_label_font)

    y_label_text = "Throughput (GB/s)" if reference_style else "Algorithm bandwidth (GB/s)"
    if reference_style:
        text_bbox = draw.textbbox((0, 0), y_label_text, font=label_font)
        padding = 2
        y_label = Image.new(
            "RGBA",
            (
                text_bbox[2] - text_bbox[0] + 2 * padding,
                text_bbox[3] - text_bbox[1] + 2 * padding,
            ),
            (255, 255, 255, 0),
        )
        text_position = (padding - text_bbox[0], padding - text_bbox[1])
    else:
        y_label = Image.new("RGBA", (360, 34), (255, 255, 255, 0))
        text_position = (0, 0)
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text(
        text_position,
        y_label_text,
        fill=(20, 20, 20),
        font=label_font,
    )
    y_label = y_label.rotate(90, expand=True)
    y_label_y = round((top + bottom - y_label.height) / 2)
    image.paste(y_label, (8 if paper else 18, y_label_y), y_label)

    legend_y = 45 if reference_style else 70
    draw.line((left, legend_y + 8, left + 30, legend_y + 8), fill=(8, 120, 181), width=3)
    draw.text((left + 40, legend_y - 3), first_label,
              fill=(20, 20, 20), font=legend_font)
    if reference_style:
        second_x, second_y = left, legend_y + 31
    else:
        second_x, second_y = left + 320, legend_y
    draw.line((second_x, second_y + 8, second_x + 30, second_y + 8),
              fill=(226, 138, 0), width=3)
    draw.text((second_x + 40, second_y - 3), second_label,
              fill=(20, 20, 20), font=legend_font)
    legend_x = left + (540 if reference_style else 540)
    draw.rectangle((legend_x, legend_y, legend_x + 24, legend_y + 16), fill=(250, 228, 230))
    draw.text((legend_x + 34, legend_y - 3), "PCIe Contention",
              fill=(20, 20, 20), font=legend_font)

    image.save(output, format="PNG")


def repair_isolated_spikes(
    values: Dict[int, float],
    deviation_threshold: float,
    window: int = 3,
    side_tolerance: float | None = None,
    max_passes: int = 5,
) -> Tuple[Dict[int, float], List[int]]:
    """Replace isolated spikes using the three samples on each side.

    A point is repaired only when the left and right local means describe the
    same plateau and the point differs from their midpoint by more than the
    requested threshold.  Iteration gaps are treated as boundaries so a real
    transition cannot be smoothed across missing data.
    """

    if deviation_threshold <= 0:
        raise ValueError("deviation_threshold must be positive")
    if window <= 0:
        raise ValueError("window must be positive")
    if side_tolerance is not None and side_tolerance <= 0:
        raise ValueError("side_tolerance must be positive")

    iterations = sorted(values)
    repaired = dict(values)
    all_replaced: List[int] = []
    if len(iterations) < 2 * window + 1:
        return repaired, all_replaced

    # Keep the two local sides sufficiently close; otherwise this is likely a
    # real slope/step rather than an isolated measurement spike.
    effective_side_tolerance = (
        side_tolerance if side_tolerance is not None
        else max(5.0, deviation_threshold * 0.75)
    )

    for _ in range(max_passes):
        pass_replaced: List[int] = []
        for position in range(window, len(iterations) - window):
            iteration = iterations[position]
            left_iterations = iterations[position - window:position]
            right_iterations = iterations[position + 1:position + window + 1]
            if left_iterations != list(range(iteration - window, iteration)):
                continue
            if right_iterations != list(range(iteration + 1, iteration + window + 1)):
                continue

            left_mean = mean(repaired[item] for item in left_iterations)
            right_mean = mean(repaired[item] for item in right_iterations)
            if abs(left_mean - right_mean) > effective_side_tolerance:
                continue

            local_baseline = (left_mean + right_mean) / 2.0
            if abs(repaired[iteration] - local_baseline) <= deviation_threshold:
                continue
            repaired[iteration] = local_baseline
            pass_replaced.append(iteration)

        if not pass_replaced:
            break
        all_replaced.extend(pass_replaced)

    return repaired, all_replaced


def smooth_value_ranges(
    values: Dict[int, float],
    ranges: Sequence[Range],
    window: int = 3,
) -> Tuple[Dict[int, float], List[int]]:
    """Interpolate explicitly selected concentrated-noise blocks.

    For each left-closed/right-open range, the three samples immediately
    before and after the block define the endpoints.  Samples inside the
    block are replaced by a linear interpolation between those two local
    means.  A range is skipped when its context is incomplete.
    """

    if window <= 0:
        raise ValueError("window must be positive")

    repaired = dict(values)
    replaced: List[int] = []
    for start, end in ranges:
        if start < 0 or end <= start:
            raise ValueError(f"invalid smoothing range: {(start, end)!r}")
        target = list(range(start, end))
        left = list(range(start - window, start))
        right = list(range(end, end + window))
        if not all(item in values for item in target + left + right):
            continue

        left_mean = mean(values[item] for item in left)
        right_mean = mean(values[item] for item in right)
        count = len(target)
        for offset, iteration in enumerate(target, start=1):
            fraction = offset / (count + 1)
            repaired[iteration] = left_mean + fraction * (right_mean - left_mean)
            replaced.append(iteration)

    return repaired, replaced


def rolling_window_stats(
    values: Dict[int, float],
    window: int,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Return a centered rolling mean and population standard deviation.

    The window is measured in available samples, not iteration distance.  At
    the two ends of a series, the window is clipped to the available samples.
    An odd window keeps the center symmetric around each sample.
    """

    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd number")

    iterations = sorted(values)
    radius = window // 2
    means: Dict[int, float] = {}
    stddevs: Dict[int, float] = {}
    for index, iteration in enumerate(iterations):
        start = max(0, index - radius)
        end = min(len(iterations), index + radius + 1)
        samples = [values[item] for item in iterations[start:end]]
        means[iteration] = mean(samples)
        stddevs[iteration] = pstdev(samples)
    return means, stddevs


def _normalize_thresholds(thresh: float | Sequence[float] | str | None) -> List[float]:
    if thresh is None:
        return []
    if isinstance(thresh, (int, float)):
        return [float(thresh)]
    if isinstance(thresh, str):
        return [float(x.strip()) for x in thresh.split(",") if x.strip()]
    return [float(x) for x in thresh]


def plot_bandwidths(
    static: Dict[int, float],
    tcp: Dict[int, float],
    ranges: Sequence[Range],
    output: Path,
    title: str,
    static_alpha: str = "0.08",
    second_label: str = "Throughput(TCP)",
    y_min: float = 0.0,
    y_max: float | None = None,
    points_only: bool = False,
    static_spike_threshold: float | Sequence[float] | str | None = None,
    second_spike_threshold: float | Sequence[float] | str | None = None,
    second_smooth_ranges: Sequence[Range] | None = None,
    spike_side_tolerance: float | None = None,
    paper: bool = False,
    line_width: float = 1.7,
    reference_style: bool = False,
    axis_break: bool = False,
    window_envelope: bool = False,
    envelope_window: int = 11,
    envelope_sigma: float = 1.0,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    if y_min < 0:
        raise ValueError("y_min must be non-negative")
    if y_max is not None and y_max <= y_min:
        raise ValueError("y_max must be greater than y_min")
    if line_width <= 0:
        raise ValueError("line_width must be positive")
    if envelope_sigma < 0:
        raise ValueError("envelope_sigma must be non-negative")
    iterations = sorted(set(static) & set(tcp))
    if not iterations:
        raise RuntimeError("no common formal iterations found in the two logs")

    plot_static = dict(static)
    plot_second = dict(tcp)
    first_label = f"Throughput(Static, α={static_alpha})"
    if reference_style:
        first_label = "Throughput(Fixed Splitting)"
        second_label = "Throughput(Multipath Orchestrator)"
    for thresh in _normalize_thresholds(static_spike_threshold):
        plot_static, _ = repair_isolated_spikes(
            plot_static,
            deviation_threshold=thresh,
            side_tolerance=spike_side_tolerance,
        )
    for thresh in _normalize_thresholds(second_spike_threshold):
        plot_second, _ = repair_isolated_spikes(
            plot_second,
            deviation_threshold=thresh,
            side_tolerance=spike_side_tolerance,
        )
    if second_smooth_ranges:
        plot_second, _ = smooth_value_ranges(plot_second, second_smooth_ranges)
    if window_envelope:
        rolling_window_stats(plot_static, envelope_window)
        rolling_window_stats(plot_second, envelope_window)

    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
    except ImportError:
        _plot_with_pillow(
            iterations,
            plot_static,
            plot_second,
            ranges,
            output,
            title,
            static_alpha,
            first_label,
            second_label,
            y_min,
            y_max,
            points_only,
            paper,
            line_width,
            reference_style,
            axis_break,
            window_envelope,
            envelope_window,
            envelope_sigma,
        )
        return plot_static, plot_second

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if reference_style:
        fig, ax = plt.subplots(figsize=(3.45, 2.35), dpi=300)
        title_size, label_size, tick_size, legend_size = reference_style_font_sizes()
    elif paper:
        fig, ax = plt.subplots(figsize=(3.45, 2.35), dpi=300)
        title_size, label_size, tick_size, legend_size = 8.5, 8.0, 7.0, 6.8
    else:
        fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=180)
        title_size, label_size, tick_size, legend_size = 15, 12, 10, 10

    if window_envelope:
        static_center, static_stddev = rolling_window_stats(plot_static, envelope_window)
        second_center, second_stddev = rolling_window_stats(plot_second, envelope_window)
    else:
        static_center, second_center = plot_static, plot_second
        static_stddev = second_stddev = {}

    if y_max is None:
        y_data_max = max(
            max(static_center[i] + envelope_sigma * static_stddev.get(i, 0.0)
                for i in iterations),
            max(second_center[i] + envelope_sigma * second_stddev.get(i, 0.0)
                for i in iterations),
        )
        padding = 0.05 if paper else 0.18
        y_max = max(y_min + 1.0, y_data_max * (1.0 + padding))

    for index, (start, end) in enumerate(ranges):
        ax.axvspan(start - 0.5, end - 0.5, color="#f3cfd3", alpha=0.48,
                   label="PCIe Contention" if index == 0 else None, zorder=0)

    if window_envelope:
        ax.fill_between(
            iterations,
            [static_center[i] - envelope_sigma * static_stddev[i] for i in iterations],
            [static_center[i] + envelope_sigma * static_stddev[i] for i in iterations],
            color="#0878b5", alpha=0.18, linewidth=0, zorder=1,
        )
        ax.fill_between(
            iterations,
            [second_center[i] - envelope_sigma * second_stddev[i] for i in iterations],
            [second_center[i] + envelope_sigma * second_stddev[i] for i in iterations],
            color="#e28a00", alpha=0.18, linewidth=0, zorder=1,
        )
        ax.plot(iterations, [static_center[i] for i in iterations], color="#0878b5",
                linewidth=line_width, label=first_label, zorder=2)
        ax.plot(iterations, [second_center[i] for i in iterations], color="#e28a00",
                linewidth=line_width, label=second_label, zorder=2)
    elif points_only:
        marker_size = 8 if paper else 14
        ax.scatter(iterations, [plot_static[i] for i in iterations], color="#0878b5",
                   s=marker_size, alpha=0.72,
                   label=first_label, zorder=2)
        ax.scatter(iterations, [plot_second[i] for i in iterations], color="#e28a00",
                   s=marker_size, alpha=0.72, label=second_label, zorder=2)
    else:
        ax.plot(iterations, [static_center[i] for i in iterations], color="#0878b5",
                linewidth=line_width, label=first_label, zorder=2)
        ax.plot(iterations, [second_center[i] for i in iterations], color="#e28a00",
                linewidth=line_width, label=second_label, zorder=2)

    ax.set_title(title, fontsize=title_size, fontweight="bold",
                 pad=5 if paper or reference_style else None)
    x_label_size = reference_style_x_label_size() if reference_style else label_size
    ax.set_xlabel("Collective Iteration", fontsize=x_label_size, fontweight="bold")
    ax.set_ylabel(
        "Throughput (GB/s)" if reference_style else "Algorithm bandwidth (GB/s)",
        fontsize=label_size,
        fontweight="bold",
        labelpad=8,
    )
    ax.xaxis.label.set_verticalalignment("top")
    ax.yaxis.label.set_verticalalignment("center")
    if reference_style:
        ax.yaxis.set_label_coords(*reference_style_y_label_position())
    ax.tick_params(axis="both", labelsize=tick_size)
    if reference_style:
        ax.tick_params(axis="both", which="major", length=4, width=1.1)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.3)
        if axis_break:
            break_kwargs = dict(
                transform=ax.transAxes,
                color="black",
                linewidth=1.3,
                clip_on=False,
            )
            ax.plot([-0.018, 0.002], [-0.015, 0.015], **break_kwargs)
            ax.plot([0.002, 0.022], [-0.015, 0.015], **break_kwargs)
    ax.set_ylim(y_min, y_max)
    _set_bandwidth_grid_locator(ax)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    if reference_style and max(iterations) >= 900:
        ax.set_xlim(0, 1000)
        ax.set_xticks([1, 200, 400, 600, 800, 1000])
    else:
        ax.set_xlim(min(iterations) - 0.5, max(iterations) + 0.5)
    ax.legend(loc="upper center",
              bbox_to_anchor=(0.5, 1.12 if reference_style else 1.14),
              ncol=2 if reference_style else 3,
              frameon=False, fontsize=legend_size, markerscale=0.9)
    if reference_style:
        fig.subplots_adjust(left=0.21, right=0.98, bottom=0.22, top=0.66)
    elif paper:
        fig.subplots_adjust(left=0.17, right=0.99, bottom=0.20, top=0.75)
    else:
        fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return plot_static, plot_second


def _size_label(size_bytes: int) -> str:
    units = ((1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB"))
    for unit_size, unit_name in units:
        if size_bytes % unit_size == 0:
            return f"{size_bytes // unit_size}{unit_name}"
    return f"{size_bytes}B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-log", type=Path)
    parser.add_argument("--tcp-log", type=Path)
    parser.add_argument("--path-log", type=Path,
                        help="extract per-path algbw, total algbw and path times from one log")
    parser.add_argument("--reference-log", type=Path,
                        help="overlay aggregate algbw from a reference log in path-log mode")
    parser.add_argument("--reference-label", default="Pure HCCL algbw",
                        help="legend label for --reference-log")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--time-output", type=Path,
                        help="write a two-line path-time plot in path-log mode")
    parser.add_argument("--hccl-only", action="store_true",
                        help="plot only the HCCL/fast path in path-log mode")
    parser.add_argument("--ranges")
    parser.add_argument("--size-bytes", type=int)
    parser.add_argument("--ranks", type=int)
    parser.add_argument("--static-alpha", default="0.08")
    parser.add_argument("--second-label", default="Throughput(TCP)",
                        help="legend label for the second comparison series")
    parser.add_argument("--y-min", type=float, default=0.0,
                        help="lower limit of the bandwidth axis in GB/s")
    parser.add_argument("--y-max", type=float,
                        help="upper limit of the bandwidth axis in GB/s")
    parser.add_argument("--points-only", action="store_true",
                        help="draw one marker per iteration without connecting lines")
    parser.add_argument("--second-spike-threshold",
                        help="repair isolated spikes in the second series above this local deviation (supports single or comma-separated values like 20,12)")
    parser.add_argument("--static-spike-threshold",
                        help="repair isolated spikes in the static series above this local deviation (supports single or comma-separated values like 20,12)")
    parser.add_argument("--spike-threshold",
                        help="repair isolated spikes in both series above this local deviation (supports single or comma-separated values like 20,12)")
    parser.add_argument("--spike-side-tolerance", type=float,
                        help="maximum allowed difference between left and right local means")
    parser.add_argument("--denoised-csv-output", type=Path,
                        help="write repaired/smoothed series to CSV")
    parser.add_argument("--second-smooth-ranges",
                        help="explicit left-closed/right-open ranges to smooth in the second series")
    parser.add_argument("--paper", action="store_true",
                        help="use compact single-column paper dimensions and typography")
    parser.add_argument("--line-width", type=float, default=1.7,
                        help="line width in points for comparison curves")
    parser.add_argument("--reference-style", action="store_true",
                        help="match the reference figure labels, frame, fonts and layout")
    parser.add_argument("--axis-break", action="store_true",
                        help="draw a broken-axis mark at the lower y-axis boundary")
    parser.add_argument("--window-envelope", action="store_true",
                        help="draw rolling mean curves with standard-deviation envelopes")
    parser.add_argument("--envelope-window", type=int, default=11,
                        help="odd number of samples in the rolling envelope window")
    parser.add_argument("--envelope-sigma", type=float, default=1.0,
                        help="number of rolling standard deviations in the envelope")
    parser.add_argument("--title", default="AllGather Throughput under PCIe Interference")
    args = parser.parse_args()

    if args.path_log is not None:
        if args.size_bytes is None or args.ranks is None:
            parser.error("--path-log requires --size-bytes and --ranks")
        metrics = parse_path_metrics(args.path_log, args.size_bytes, args.ranks)
        if not metrics:
            raise RuntimeError("could not parse path bandwidth records from the log")
        reference = None
        if args.reference_log is not None:
            reference = parse_log(args.reference_log, args.size_bytes, args.ranks)
            if not reference:
                raise RuntimeError("could not parse reference algbw records from the log")
        timings = {
            iteration: metric.timing
            for iteration, metric in metrics.items()
            if metric.timing is not None
        }
        if args.csv_output is not None:
            write_path_metrics_csv(args.csv_output, metrics)
        if args.hccl_only:
            plot_hccl_bandwidths(metrics, args.output, args.title)
        else:
            plot_path_bandwidths(
                metrics,
                args.output,
                args.title,
                reference=reference,
                reference_label=args.reference_label,
            )
        if args.time_output is not None:
            if not timings:
                raise RuntimeError("could not parse path timing records from the log")
            time_title = args.title.replace("Bandwidth", "Path Time")
            if args.hccl_only:
                plot_hccl_timings(timings, args.time_output, time_title)
            else:
                plot_path_timings(timings, args.time_output, time_title)
        print(f"path_records={len(metrics)} timing_records={len(timings)}")
        print(f"plot={args.output}")
        if args.time_output is not None:
            print(f"time_plot={args.time_output}")
        if args.csv_output is not None:
            print(f"csv={args.csv_output}")
        return

    if args.static_log is None or args.tcp_log is None or args.ranges is None:
        parser.error("comparison mode requires --static-log, --tcp-log and --ranges")
    if args.size_bytes is None or args.ranks is None:
        parser.error("comparison mode requires --size-bytes and --ranks")
    ranges = parse_ranges(args.ranges)
    second_smooth_ranges = (
        parse_ranges(args.second_smooth_ranges)
        if args.second_smooth_ranges else None
    )
    static = parse_log(args.static_log, args.size_bytes, args.ranks)
    tcp = parse_log(args.tcp_log, args.size_bytes, args.ranks)
    if not static or not tcp:
        raise RuntimeError("could not parse formal algbw records from both logs")

    static_spike_threshold = args.static_spike_threshold
    second_spike_threshold = args.second_spike_threshold
    if args.spike_threshold is not None:
        if static_spike_threshold is None:
            static_spike_threshold = args.spike_threshold
        if second_spike_threshold is None:
            second_spike_threshold = args.spike_threshold

    if args.csv_output is not None:
        write_csv(args.csv_output, static, tcp)
    repaired_static, repaired_tcp = plot_bandwidths(
        static,
        tcp,
        ranges,
        args.output,
        args.title,
        args.static_alpha,
        args.second_label,
        args.y_min,
        args.y_max,
        args.points_only,
        static_spike_threshold,
        second_spike_threshold,
        second_smooth_ranges,
        args.spike_side_tolerance,
        args.paper,
        args.line_width,
        args.reference_style,
        args.axis_break,
        args.window_envelope,
        args.envelope_window,
        args.envelope_sigma,
    )
    if args.denoised_csv_output is not None:
        write_csv(args.denoised_csv_output, repaired_static, repaired_tcp)
    print(f"static_records={len(static)} tcp_records={len(tcp)}")
    print(f"plot={args.output}")
    if args.csv_output is not None:
        print(f"csv={args.csv_output}")
    if args.denoised_csv_output is not None:
        print(f"denoised_csv={args.denoised_csv_output}")


if __name__ == "__main__":
    main()
