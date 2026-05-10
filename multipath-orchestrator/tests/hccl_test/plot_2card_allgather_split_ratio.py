#!/usr/bin/env python3
"""Parse and plot 2-card 256MB AllGather split_ratio feedback regulation with PCIe interference."""

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


def parse_split_ratio_log(log_path: Path) -> List[Dict]:
    """Parse iteration records from mpi_primitives log."""
    records = []
    current_iter = None
    fast_time = None
    pcie_time = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Match iter header
            m_iter = re.search(r"\[rank0\]\s+begin formal\s+(\d+)\s+iters", line)
            if m_iter:
                current_iter = int(m_iter.group(1))
                fast_time = None
                pcie_time = None
                continue

            # Match path times
            m_fast = re.search(r"\(max\)\s+fast_time=([0-9.eE+-]+)s", line)
            if m_fast:
                fast_time = float(m_fast.group(1)) * 1000.0  # to ms
                continue

            m_pcie = re.search(r"\(max\)\s+pcie_time=([0-9.eE+-]+)s", line)
            if m_pcie:
                pcie_time = float(m_pcie.group(1)) * 1000.0  # to ms
                continue

            # Match aggregate line
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

                is_contention = (20 <= current_iter < 40) or (50 <= current_iter < 60)

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


def draw_plot(records: List[Dict], output_path: Path, ranges: List[Tuple[int, int]]):
    """Draw a clean 3-panel figure: Bandwidth, Split Ratio (Alpha), Path Timings."""
    if not records:
        print("No records to plot.")
        return

    width, height = 1300, 950
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_title = get_font(20, bold=True)
    font_panel = get_font(15, bold=True)
    font_label = get_font(13, bold=True)
    font_tick = get_font(11)
    font_annot = get_font(12, bold=True)

    # Margins & Subplot bounding boxes
    left = 110
    right = width - 60
    plot_width = right - left

    # 3 subplots:
    # 1. Bandwidth: top=70, h=220
    # 2. Split Ratio: top=340, h=220
    # 3. Path Timing: top=610, h=220
    panels = [
        {"top": 80, "bottom": 310, "title": "AllGather Algorithm Bandwidth (algbw, GB/s)"},
        {"top": 360, "bottom": 590, "title": "Adaptive Split Ratio Alpha (PCIe Path Share)"},
        {"top": 640, "bottom": 870, "title": "Path Timings (Fast / PCIe Path, ms)"},
    ]

    # Global Title
    title_text = "2-Card 256MB AllGather: split_ratio Feedback Regulation (128MB PCIe Interference)"
    draw.text((left, 25), title_text, fill=(20, 20, 20), font=font_title)

    n_iters = len(records)
    x_min = 0
    x_max = max(r["iter"] for r in records)
    if x_max == x_min:
        x_max += 1

    def x_pixel(it: float) -> int:
        return int(left + (it - x_min) / (x_max - x_min) * plot_width)

    # Shaded interference zones on all panels
    for p in panels:
        p_top = p["top"]
        p_bottom = p["bottom"]
        for r_start, r_end in ranges:
            xs = x_pixel(r_start)
            xe = x_pixel(r_end)
            draw.rectangle([xs, p_top, xe, p_bottom], fill=(255, 235, 235))

    # Grid & Axes helper
    def draw_panel_box_and_grid(top, bottom, y_min, y_max, y_ticks, unit=""):
        # Box
        draw.rectangle([left, top, right, bottom], outline=(200, 200, 200), width=1)
        # X-grid
        for x_val in range(0, x_max + 1, 10):
            xp = x_pixel(x_val)
            draw.line([(xp, top), (xp, bottom)], fill=(240, 240, 240), width=1)
            draw.text((xp - 6, bottom + 5), str(x_val), fill=(90, 90, 90), font=font_tick)

        # Y-grid
        for y_val in y_ticks:
            yp = int(bottom - (y_val - y_min) / (y_max - y_min) * (bottom - top))
            draw.line([(left, yp), (right, yp)], fill=(230, 230, 230), width=1)
            lbl = f"{y_val:.2f}" if isinstance(y_val, float) else str(y_val)
            draw.text((left - 45, yp - 7), lbl, fill=(90, 90, 90), font=font_tick)

    # --- Panel 1: Bandwidth ---
    p1 = panels[0]
    draw.text((left, p1["top"] - 25), "Panel A: " + p1["title"], fill=(30, 30, 30), font=font_panel)
    bw_vals = [r["algbw"] for r in records]
    bw_min = 0.0
    bw_max = max(bw_vals) * 1.15 if bw_vals else 100.0
    bw_ticks = [round(bw_max * i / 5.0, 1) for i in range(6)]
    draw_panel_box_and_grid(p1["top"], p1["bottom"], bw_min, bw_max, bw_ticks)

    # Plot bandwidth curve
    pts_bw = []
    for r in records:
        xp = x_pixel(r["iter"])
        yp = int(p1["bottom"] - (r["algbw"] - bw_min) / (bw_max - bw_min) * (p1["bottom"] - p1["top"]))
        pts_bw.append((xp, yp))
    for i in range(len(pts_bw) - 1):
        draw.line([pts_bw[i], pts_bw[i + 1]], fill=(30, 120, 220), width=2)
    for xp, yp in pts_bw:
        draw.ellipse([xp - 2, yp - 2, xp + 2, yp + 2], fill=(20, 90, 180))

    # --- Panel 2: Split Ratio Alpha ---
    p2 = panels[1]
    draw.text((left, p2["top"] - 25), "Panel B: " + p2["title"], fill=(30, 30, 30), font=font_panel)
    a_vals = [r["split_pcie"] for r in records]
    a_min = 0.0
    a_max = max(max(a_vals) * 1.25, 0.35) if a_vals else 0.40
    a_ticks = [round(a_max * i / 5.0, 2) for i in range(6)]
    draw_panel_box_and_grid(p2["top"], p2["bottom"], a_min, a_max, a_ticks)

    pts_a = []
    for r in records:
        xp = x_pixel(r["iter"])
        yp = int(p2["bottom"] - (r["split_pcie"] - a_min) / (a_max - a_min) * (p2["bottom"] - p2["top"]))
        pts_a.append((xp, yp))
    for i in range(len(pts_a) - 1):
        draw.line([pts_a[i], pts_a[i + 1]], fill=(220, 100, 30), width=2)
    for xp, yp in pts_a:
        draw.ellipse([xp - 2, yp - 2, xp + 2, yp + 2], fill=(180, 70, 10))

    # --- Panel 3: Path Timings ---
    p3 = panels[2]
    draw.text((left, p3["top"] - 25), "Panel C: " + p3["title"], fill=(30, 30, 30), font=font_panel)
    t_vals = [r["fast_time_ms"] for r in records] + [r["pcie_time_ms"] for r in records]
    t_min = 0.0
    t_max = max(t_vals) * 1.15 if t_vals else 50.0
    t_ticks = [round(t_max * i / 5.0, 1) for i in range(6)]
    draw_panel_box_and_grid(p3["top"], p3["bottom"], t_min, t_max, t_ticks)

    pts_fast = []
    pts_pcie = []
    for r in records:
        xp = x_pixel(r["iter"])
        yp_f = int(p3["bottom"] - (r["fast_time_ms"] - t_min) / (t_max - t_min) * (p3["bottom"] - p3["top"]))
        yp_p = int(p3["bottom"] - (r["pcie_time_ms"] - t_min) / (t_max - t_min) * (p3["bottom"] - p3["top"]))
        pts_fast.append((xp, yp_f))
        pts_pcie.append((xp, yp_p))

    for i in range(len(pts_fast) - 1):
        draw.line([pts_fast[i], pts_fast[i + 1]], fill=(40, 160, 60), width=2)
        draw.line([pts_pcie[i], pts_pcie[i + 1]], fill=(200, 40, 40), width=2)
    for xp, yp in pts_fast:
        draw.ellipse([xp - 2, yp - 2, xp + 2, yp + 2], fill=(30, 130, 50))
    for xp, yp in pts_pcie:
        draw.ellipse([xp - 2, yp - 2, xp + 2, yp + 2], fill=(160, 30, 30))

    # Interference Banner Annotations
    for r_start, r_end in ranges:
        xs = x_pixel(r_start)
        xe = x_pixel(r_end)
        mid_x = (xs + xe) // 2
        banner_text = f"128MB PCIe Contention [{r_start}, {r_end})"
        draw.text((mid_x - 75, p1["top"] + 8), banner_text, fill=(180, 40, 40), font=font_annot)
        draw.text((mid_x - 75, p2["top"] + 8), banner_text, fill=(180, 40, 40), font=font_annot)
        draw.text((mid_x - 75, p3["top"] + 8), banner_text, fill=(180, 40, 40), font=font_annot)

    # Legend on Panel 3
    leg_x = right - 260
    leg_y = p3["top"] + 15
    draw.rectangle([leg_x - 10, leg_y - 5, right - 10, leg_y + 45], fill=(255, 255, 255), outline=(200, 200, 200))
    draw.line([(leg_x, leg_y + 8), (leg_x + 25, leg_y + 8)], fill=(40, 160, 60), width=3)
    draw.text((leg_x + 35, leg_y + 2), "Fast Path Time (ms)", fill=(30, 30, 30), font=font_tick)
    draw.line([(leg_x, leg_y + 28), (leg_x + 25, leg_y + 28)], fill=(200, 40, 40), width=3)
    draw.text((leg_x + 35, leg_y + 22), "PCIe Path Time (ms)", fill=(30, 30, 30), font=font_tick)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    print(f"Chart saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--ranges", default="20-40,50-60")
    args = parser.parse_args()

    # Parse ranges
    ranges = []
    for item in args.ranges.split(","):
        if "-" in item:
            s, e = item.strip().split("-")
            ranges.append((int(s), int(e)))

    records = parse_split_ratio_log(args.log)
    print(f"Parsed {len(records)} iteration records from {args.log}")

    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "iter", "algbw", "split_fast", "split_pcie",
                "fast_time_ms", "pcie_time_ms", "step_time_ms", "is_contention"
            ])
            writer.writeheader()
            writer.writerows(records)
        print(f"CSV saved to {args.csv_output}")

    draw_plot(records, args.output, ranges)

    # Analyze clean intervals
    clean_records = [r for r in records if not r["is_contention"] and r["iter"] >= 5]
    contention_records = [r for r in records if r["is_contention"]]

    if clean_records:
        clean_bw = [r["algbw"] for r in clean_records]
        clean_alpha = [r["split_pcie"] for r in clean_records]
        print(f"\n=== Clean Intervals (No Contention, iter>=5, N={len(clean_records)}) ===")
        print(f"  algbw: mean={sum(clean_bw)/len(clean_bw):.2f} GB/s, min={min(clean_bw):.2f}, max={max(clean_bw):.2f}")
        print(f"  alpha: mean={sum(clean_alpha)/len(clean_alpha):.4f}, min={min(clean_alpha):.4f}, max={max(clean_alpha):.4f}")

        # Top 5 bandwidth records in clean intervals and their alphas
        sorted_by_bw = sorted(clean_records, key=lambda x: x["algbw"], reverse=True)
        top5_alphas = [r["split_pcie"] for r in sorted_by_bw[:5]]
        print(f"  Top 5 bandwidths in clean state: {[round(r['algbw'], 2) for r in sorted_by_bw[:5]]} GB/s")
        print(f"  Corresponding alphas: {[round(a, 4) for a in top5_alphas]}")
        print(f"  Recommended Static Alpha Candidate: {sum(top5_alphas)/len(top5_alphas):.2f}")

    if contention_records:
        cont_bw = [r["algbw"] for r in contention_records]
        cont_alpha = [r["split_pcie"] for r in contention_records]
        print(f"\n=== Contention Intervals (N={len(contention_records)}) ===")
        print(f"  algbw: mean={sum(cont_bw)/len(cont_bw):.2f} GB/s, min={min(cont_bw):.2f}, max={max(cont_bw):.2f}")
        print(f"  alpha: mean={sum(cont_alpha)/len(cont_alpha):.4f}, min={min(cont_alpha):.4f}, max={max(cont_alpha):.4f}")


if __name__ == "__main__":
    main()
