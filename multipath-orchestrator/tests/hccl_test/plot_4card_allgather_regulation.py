#!/usr/bin/env python3
"""Parse and plot 4-card 256MB AllGather split_ratio vs tcp feedback regulation with PCIe interference."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont


def get_font(size: int, bold: bool = False):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in font_paths:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def parse_algo_log(log_path: Path, ranges: List[Tuple[int, int]]) -> List[Dict]:
    """Parse iteration records from mpi_primitives log."""
    records = []
    current_iter = None
    fast_time = None
    pcie_time = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_iter = re.search(r"\[rank0\]\s+begin formal\s+(\d+)\s+iters", line)
            if m_iter:
                current_iter = int(m_iter.group(1))
                fast_time = None
                pcie_time = None
                continue

            m_fast = re.search(r"\(max\)\s+fast_time=([0-9.eE+-]+)s", line)
            if m_fast:
                fast_time = float(m_fast.group(1)) * 1000.0  # to ms
                continue

            m_pcie = re.search(r"\(max\)\s+pcie_time=([0-9.eE+-]+)s", line)
            if m_pcie:
                pcie_time = float(m_pcie.group(1)) * 1000.0  # to ms
                continue

            m_agg = re.search(
                r"aggregate\s+op=AllGather.*?"
                r"split_fast=([0-9.]+)\s+split_pcie=([0-9.]+)\s*\|.*?"
                r"algbw\(GB/s,decimal\)=([0-9.eE+-]+)",
                line,
            )
            if m_agg and current_iter is not None:
                split_fast = float(m_agg.group(1))
                split_pcie = float(m_agg.group(2))
                algbw = float(m_agg.group(3))

                ft = fast_time if fast_time is not None else 0.0
                pt = pcie_time if pcie_time is not None else 0.0
                step_time = max(ft, pt) if (ft > 0 and pt > 0) else (ft if ft > 0 else pt)

                is_contention = any(start <= current_iter < end for start, end in ranges)

                records.append({
                    "iter": current_iter,
                    "algbw": algbw,
                    "split_fast": split_fast,
                    "split_pcie": split_pcie,
                    "fast_time_ms": ft,
                    "pcie_time_ms": pt,
                    "step_time_ms": step_time,
                    "is_contention": is_contention,
                })
                current_iter = None
                fast_time = None
                pcie_time = None

    return records


def draw_comparison_plot(
    split_records: List[Dict],
    tcp_records: List[Dict],
    output_path: Path,
    ranges: List[Tuple[int, int]],
):
    """Draw a clean 3-panel comparison figure: Bandwidth, Alpha Regulation, Path Timings."""
    width, height = 1300, 1000
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_title = get_font(20, bold=True)
    font_panel = get_font(15, bold=True)
    font_label = get_font(13, bold=True)
    font_tick = get_font(11)
    font_annot = get_font(12, bold=True)

    left = 110
    right = width - 60
    plot_width = right - left

    panels = [
        {"top": 80, "bottom": 330, "title": "AllGather Algorithm Bandwidth (algbw, GB/s)"},
        {"top": 390, "bottom": 640, "title": "Adaptive Split Ratio Alpha (PCIe Path Share)"},
        {"top": 700, "bottom": 940, "title": "split_ratio Path Timings (Fast / PCIe Path, ms)"},
    ]

    title_text = "4-Card 256MB AllGather: split_ratio vs tcp Under 128MB PCIe Interference"
    draw.text((left, 25), title_text, fill=(20, 20, 20), font=font_title)

    all_iters = [r["iter"] for r in split_records] + [r["iter"] for r in tcp_records]
    if not all_iters:
        print("No iterations to plot.")
        return
    x_min = 0
    x_max = max(all_iters)
    if x_max == x_min:
        x_max += 1

    def x_pixel(it: float) -> int:
        return int(left + (it - x_min) / (x_max - x_min) * plot_width)

    # Draw shaded interference zones
    for p in panels:
        p_top = p["top"]
        p_bottom = p["bottom"]
        for r_start, r_end in ranges:
            x0 = x_pixel(r_start)
            x1 = x_pixel(r_end)
            draw.rectangle([x0, p_top, x1, p_bottom], fill=(245, 240, 240))
            draw.line([(x0, p_top), (x0, p_bottom)], fill=(220, 160, 160), width=1)
            draw.line([(x1, p_top), (x1, p_bottom)], fill=(220, 160, 160), width=1)

    # Shaded label on top panel
    for r_start, r_end in ranges:
        x0 = x_pixel(r_start)
        x1 = x_pixel(r_end)
        mid_x = (x0 + x1) // 2
        draw.text((mid_x - 55, panels[0]["top"] + 8), "128MB PCIe Jam", fill=(180, 70, 70), font=font_annot)

    def draw_axes_and_grid(p, y_min: float, y_max: float, n_ticks: int = 5, fmt: str = "%.1f"):
        p_top = p["top"]
        p_bottom = p["bottom"]
        p_height = p_bottom - p_top

        draw.rectangle([left, p_top, right, p_bottom], outline=(180, 180, 180), width=1)
        draw.text((left, p_top - 24), p["title"], fill=(40, 40, 40), font=font_panel)

        for i in range(n_ticks + 1):
            val = y_min + i * (y_max - y_min) / n_ticks
            y_pix = int(p_bottom - (val - y_min) / (y_max - y_min) * p_height)
            draw.line([(left, y_pix), (right, y_pix)], fill=(235, 235, 235), width=1)
            lbl = fmt % val
            draw.text((left - 45, y_pix - 7), lbl, fill=(100, 100, 100), font=font_tick)

        for x_val in range(0, x_max + 1, 10):
            x_pix = x_pixel(x_val)
            draw.line([(x_pix, p_bottom), (x_pix, p_bottom + 5)], fill=(120, 120, 120), width=1)
            if p == panels[-1]:
                draw.text((x_pix - 7, p_bottom + 8), str(x_val), fill=(80, 80, 80), font=font_tick)

    # Panel 1: Bandwidth comparison
    p1 = panels[0]
    bw_vals = [r["algbw"] for r in split_records] + [r["algbw"] for r in tcp_records]
    y1_min = 0.0
    y1_max = (max(bw_vals) * 1.15) if bw_vals else 100.0
    draw_axes_and_grid(p1, y1_min, y1_max, n_ticks=5, fmt="%.0f")

    # Legend for Panel 1
    leg_x = right - 320
    leg_y = p1["top"] + 12
    draw.line([(leg_x, leg_y), (leg_x + 25, leg_y)], fill=(30, 120, 220), width=3)
    draw.text((leg_x + 32, leg_y - 7), "split_ratio (Ours)", fill=(20, 20, 20), font=font_label)
    draw.line([(leg_x + 180, leg_y), (leg_x + 205, leg_y)], fill=(220, 50, 50), width=2)
    draw.text((leg_x + 212, leg_y - 7), "tcp (AIMD)", fill=(20, 20, 20), font=font_label)

    # Plot split_ratio algbw
    if split_records:
        pts = [(x_pixel(r["iter"]), int(p1["bottom"] - (r["algbw"] - y1_min) / (y1_max - y1_min) * (p1["bottom"] - p1["top"]))) for r in split_records]
        draw.line(pts, fill=(30, 120, 220), width=3)
        for pt in pts:
            draw.ellipse([pt[0]-2, pt[1]-2, pt[0]+2, pt[1]+2], fill=(30, 120, 220))

    # Plot tcp algbw
    if tcp_records:
        pts_tcp = [(x_pixel(r["iter"]), int(p1["bottom"] - (r["algbw"] - y1_min) / (y1_max - y1_min) * (p1["bottom"] - p1["top"]))) for r in tcp_records]
        draw.line(pts_tcp, fill=(220, 50, 50), width=2)
        for pt in pts_tcp:
            draw.ellipse([pt[0]-2, pt[1]-2, pt[0]+2, pt[1]+2], fill=(220, 50, 50))

    # Panel 2: Split Ratio (Alpha)
    p2 = panels[1]
    alpha_vals = [r["split_pcie"] for r in split_records] + [r["split_pcie"] for r in tcp_records]
    y2_min = 0.0
    y2_max = max(0.40, (max(alpha_vals) * 1.25) if alpha_vals else 0.40)
    draw_axes_and_grid(p2, y2_min, y2_max, n_ticks=4, fmt="%.2f")

    # Legend for Panel 2
    draw.line([(leg_x, p2["top"] + 12), (leg_x + 25, p2["top"] + 12)], fill=(30, 120, 220), width=3)
    draw.text((leg_x + 32, p2["top"] + 5), "split_ratio alpha", fill=(20, 20, 20), font=font_label)
    draw.line([(leg_x + 180, p2["top"] + 12), (leg_x + 205, p2["top"] + 12)], fill=(220, 50, 50), width=2)
    draw.text((leg_x + 212, p2["top"] + 5), "tcp alpha", fill=(20, 20, 20), font=font_label)

    if split_records:
        pts_a = [(x_pixel(r["iter"]), int(p2["bottom"] - (r["split_pcie"] - y2_min) / (y2_max - y2_min) * (p2["bottom"] - p2["top"]))) for r in split_records]
        draw.line(pts_a, fill=(30, 120, 220), width=3)
        for pt in pts_a:
            draw.ellipse([pt[0]-2, pt[1]-2, pt[0]+2, pt[1]+2], fill=(30, 120, 220))

    if tcp_records:
        pts_ta = [(x_pixel(r["iter"]), int(p2["bottom"] - (r["split_pcie"] - y2_min) / (y2_max - y2_min) * (p2["bottom"] - p2["top"]))) for r in tcp_records]
        draw.line(pts_ta, fill=(220, 50, 50), width=2)
        for pt in pts_ta:
            draw.ellipse([pt[0]-2, pt[1]-2, pt[0]+2, pt[1]+2], fill=(220, 50, 50))

    # Panel 3: split_ratio Path Timings
    p3 = panels[2]
    time_vals = [r["fast_time_ms"] for r in split_records] + [r["pcie_time_ms"] for r in split_records]
    y3_min = 0.0
    y3_max = (max(time_vals) * 1.25) if time_vals else 20.0
    draw_axes_and_grid(p3, y3_min, y3_max, n_ticks=4, fmt="%.1f")

    # Legend for Panel 3
    draw.line([(leg_x, p3["top"] + 12), (leg_x + 25, p3["top"] + 12)], fill=(40, 160, 60), width=2)
    draw.text((leg_x + 32, p3["top"] + 5), "Fast Path (ms)", fill=(20, 20, 20), font=font_label)
    draw.line([(leg_x + 180, p3["top"] + 12), (leg_x + 205, p3["top"] + 12)], fill=(230, 140, 20), width=2)
    draw.text((leg_x + 212, p3["top"] + 5), "PCIe Path (ms)", fill=(20, 20, 20), font=font_label)

    if split_records:
        pts_ft = [(x_pixel(r["iter"]), int(p3["bottom"] - (r["fast_time_ms"] - y3_min) / (y3_max - y3_min) * (p3["bottom"] - p3["top"]))) for r in split_records]
        pts_pt = [(x_pixel(r["iter"]), int(p3["bottom"] - (r["pcie_time_ms"] - y3_min) / (y3_max - y3_min) * (p3["bottom"] - p3["top"]))) for r in split_records]
        draw.line(pts_ft, fill=(40, 160, 60), width=2)
        draw.line(pts_pt, fill=(230, 140, 20), width=2)

    draw.text((width // 2 - 60, height - 30), "Formal Iteration", fill=(60, 60, 60), font=font_label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    print(f"Regulation plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-log", type=Path, required=True)
    parser.add_argument("--tcp-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranges", type=str, default="20-40,55-65")
    args = parser.parse_args()

    ranges = []
    for item in args.ranges.split(","):
        s, e = item.split("-")
        ranges.append((int(s), int(e)))

    split_records = parse_algo_log(args.split_log, ranges)
    tcp_records = parse_algo_log(args.tcp_log, ranges)

    draw_comparison_plot(split_records, tcp_records, args.output, ranges)


if __name__ == "__main__":
    main()
