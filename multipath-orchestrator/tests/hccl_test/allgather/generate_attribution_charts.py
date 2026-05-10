#!/usr/bin/env python3
"""Generate attribution and bottleneck visualization charts for PCIeCCL AllGather benchmark.

Reads summary.json and iterations.csv from msprof profiling runs and generates:
1. stage_breakdown_stacked.png: Stacked execution phase breakdown across iterations (normal vs interference)
2. slow_iterations_case_study.png: Comparative metrics for normal fast iterations vs outlier iterations
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_font(size=14, bold=False):
    """Load DejaVuSans font with fallback to PIL default font."""
    font_path = (
        '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf'
        if bold
        else '/usr/share/fonts/dejavu/DejaVuSans.ttf'
    )
    try:
        return ImageFont.truetype(font_path, size)
    except OSError:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def load_dataset(summary_path, iterations_path):
    """Load and parse summary.json and iterations.csv."""
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    with open(iterations_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        iterations = list(reader)

    return summary, iterations


def render_stage_breakdown_stacked(iterations, output_path):
    """Generate stacked execution stage breakdown across normal and interference iterations."""
    # Organize data by case and iteration
    cases_data = {
        'split_ratio_msprof': defaultdict(list),
        'tcp_msprof': defaultdict(list),
    }

    for row in iterations:
        case_name = 'split_ratio_msprof' if 'split_ratio_msprof' in row['case'] else 'tcp_msprof'
        iter_idx = int(row['iter'])
        cases_data[case_name][iter_idx].append(row)

    # Image canvas dimensions
    width, height = 1850, 1150
    im = Image.new('RGB', (width, height), '#FAFAFA')
    draw = ImageDraw.Draw(im)

    # Fonts
    title_font = get_font(22, bold=True)
    subtitle_font = get_font(14, bold=False)
    panel_title_font = get_font(16, bold=True)
    axis_font = get_font(12, bold=False)
    bold_axis_font = get_font(12, bold=True)
    legend_font = get_font(13, bold=False)
    callout_font = get_font(11, bold=True)

    # Title & Subtitle
    draw.text((60, 20), "PCIeCCL AllGather Execution Phase Breakdown & DMA Union Timeline", font=title_font, fill='#212121')
    draw.text((60, 52), "80-Iteration Timeline Breakdown (Host Acquire, Start Barrier, Loop) and Overlay DMA Busy Union under Background D2H Interference", font=subtitle_font, fill='#616161')

    # Color Palette
    color_acquire = '#1976D2'     # Blue: Host acquire
    color_barrier = '#FF7043'     # Orange: Start barrier
    color_loop = '#388E3C'        # Green: PCCL loop
    color_other = '#9E9E9E'       # Grey: Other Host overhead
    color_dma = '#D32F2F'         # Red: DMA busy union line
    color_inf_bg = '#FFF8E1'      # Pale Amber: Interference background
    color_grid = '#E0E0E0'        # Light gridlines

    # Draw Legend on its own dedicated row
    legend_items = [
        ("Host Acquire (ms)", color_acquire, 'box'),
        ("Start Barrier (ms)", color_barrier, 'box'),
        ("PCCL Loop (ms)", color_loop, 'box'),
        ("DMA Busy Union (ms)", color_dma, 'line'),
        ("Interference Interval [20,39], [55,64]", '#FFE082', 'box'),
    ]
    lx = 60
    ly = 85
    for label, col, style in legend_items:
        if style == 'box':
            draw.rectangle((lx, ly, lx + 20, ly + 14), fill=col, outline='#BDBDBD')
        else:
            draw.line((lx, ly + 7, lx + 20, ly + 7), fill=col, width=3)
            draw.ellipse((lx + 7, ly + 4, lx + 13, ly + 10), fill=col)
        draw.text((lx + 26, ly - 1), label, font=legend_font, fill='#333333')
        lx += len(label) * 8 + 48

    panels = [
        ('split_ratio_msprof', 'Split-Ratio Communication Algorithm (Adaptive Fast/PCIe Ratio)', 125, 590),
        ('tcp_msprof', 'TCP Communication Algorithm (Fixed Socket/PCIe Partitioning)', 635, 1100),
    ]

    for case_key, panel_title, p_top, p_bottom in panels:
        # Panel container box
        draw.rounded_rectangle((50, p_top, width - 50, p_bottom), radius=8, fill='#FFFFFF', outline='#E0E0E0', width=1)
        draw.text((70, p_top + 15), panel_title, font=panel_title_font, fill='#1A237E')

        # Sub-plot area for 80 iterations timeline
        t_left, t_right = 130, 1450
        t_top, t_bottom = p_top + 65, p_bottom - 50
        plot_height = t_bottom - t_top
        max_y = 50.0  # ms scale for timeline

        # Highlight interference iterations: [20, 39] and [55, 64]
        bar_width = (t_right - t_left) / 80.0
        for inf_start, inf_end in [(20, 39), (55, 64)]:
            x1 = t_left + inf_start * bar_width
            x2 = t_left + (inf_end + 1) * bar_width
            draw.rectangle((x1, t_top, x2, t_bottom), fill=color_inf_bg)
            draw.text((x1 + 8, t_top + 8), f"Interference [{inf_start},{inf_end}]", font=callout_font, fill='#E65100')

        # Draw Gridlines & Y-axis labels
        for y_val in [0, 10, 20, 30, 40, 50]:
            y_pos = t_bottom - (y_val / max_y) * plot_height
            draw.line((t_left, y_pos, t_right, y_pos), fill=color_grid, width=1)
            draw.text((t_left - 45, y_pos - 7), f"{y_val}ms", font=axis_font, fill='#757575')

        # Draw X-axis ticks & labels
        for x_tick in range(0, 81, 10):
            x_pos = t_left + min(x_tick, 79.5) * bar_width
            draw.line((x_pos, t_bottom, x_pos, t_bottom + 5), fill='#757575', width=1)
            draw.text((x_pos - 8, t_bottom + 8), str(x_tick), font=axis_font, fill='#616161')
        draw.text((t_left + (t_right - t_left) // 2 - 30, t_bottom + 25), "Iteration Index", font=bold_axis_font, fill='#424242')

        # Process each iteration
        dma_points = []
        for iter_idx in range(80):
            rows = cases_data[case_key][iter_idx]
            if not rows:
                continue

            # Compute rank averages in ms
            acq_ms = np.mean([float(r['pccl_acquire_us']) for r in rows]) / 1000.0
            barr_ms = np.mean([float(r['pccl_start_barrier_us']) for r in rows]) / 1000.0
            loop_ms = np.mean([float(r['pccl_loop_us']) for r in rows]) / 1000.0
            ag_ms = np.mean([float(r['ag_host_us']) for r in rows]) / 1000.0
            dma_ms = np.mean([float(r['dma_busy_us']) for r in rows]) / 1000.0

            # Stacked bars
            x_center = t_left + (iter_idx + 0.5) * bar_width
            bw = max(2, bar_width - 3)
            bx1 = x_center - bw / 2
            bx2 = x_center + bw / 2

            # Stack 1: Acquire
            y0 = t_bottom
            y1 = y0 - (acq_ms / max_y) * plot_height
            draw.rectangle((bx1, max(t_top, y1), bx2, y0), fill=color_acquire)

            # Stack 2: Start Barrier
            y2 = y1 - (barr_ms / max_y) * plot_height
            draw.rectangle((bx1, max(t_top, y2), bx2, max(t_top, y1)), fill=color_barrier)

            # Stack 3: Loop
            y3 = y2 - (loop_ms / max_y) * plot_height
            draw.rectangle((bx1, max(t_top, y3), bx2, max(t_top, y2)), fill=color_loop)

            # DMA Busy line point
            dma_y = t_bottom - (min(dma_ms, max_y) / max_y) * plot_height
            dma_points.append((x_center, dma_y))

            # Specific callouts for outlier iterations
            if case_key == 'split_ratio_msprof' and iter_idx == 23:
                # Annotation for iter 23 outlier
                callout_y = max(t_top + 10, y3 - 25)
                draw.line((x_center, y3, x_center + 25, callout_y), fill='#D32F2F', width=2)
                draw.rectangle((x_center + 20, callout_y - 20, x_center + 280, callout_y + 16), fill='#FFEBEE', outline='#D32F2F')
                draw.text((x_center + 26, callout_y - 16), "Iter 23 Outlier (Host Sync Spike)", font=callout_font, fill='#C62828')
                draw.text((x_center + 26, callout_y - 1), f"Acquire={acq_ms:.1f}ms, Barrier={barr_ms:.1f}ms, Loop={loop_ms:.1f}ms", font=axis_font, fill='#B71C1C')

            if case_key == 'split_ratio_msprof' and iter_idx == 57:
                callout_y = max(t_top + 30, y3 - 35)
                draw.line((x_center, y3, x_center + 25, callout_y), fill='#2E7D32', width=2)
                draw.rectangle((x_center + 20, callout_y - 20, x_center + 270, callout_y + 16), fill='#E8F5E9', outline='#2E7D32')
                draw.text((x_center + 26, callout_y - 16), "Iter 57 In-Loop Inflation", font=callout_font, fill='#1B5E20')
                draw.text((x_center + 26, callout_y - 1), f"Loop={loop_ms:.1f}ms, Acquire/Barrier<0.2ms", font=axis_font, fill='#2E7D32')

            if case_key == 'tcp_msprof' and iter_idx == 61:
                callout_y = max(t_top + 30, y3 - 35)
                draw.line((x_center, y3, x_center + 25, callout_y), fill='#2E7D32', width=2)
                draw.rectangle((x_center + 20, callout_y - 20, x_center + 270, callout_y + 16), fill='#E8F5E9', outline='#2E7D32')
                draw.text((x_center + 26, callout_y - 16), "Iter 61 In-Loop Inflation", font=callout_font, fill='#1B5E20')
                draw.text((x_center + 26, callout_y - 1), f"Loop={loop_ms:.1f}ms, Acquire/Barrier<0.2ms", font=axis_font, fill='#2E7D32')

        # Draw DMA busy union line
        if len(dma_points) > 1:
            draw.line(dma_points, fill=color_dma, width=2)
            for px, py in dma_points[::4]:
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color_dma)

        # Right Summary Sub-plot: Normal vs Interference Averages
        s_left, s_right = 1520, width - 80
        s_top, s_bottom = t_top, t_bottom

        draw.text((s_left - 10, p_top + 45), "Phase Averages", font=get_font(13, bold=True), fill='#37474F')

        # Compute phase averages
        normal_rows = [r for iters in cases_data[case_key].values() for r in iters if r['interference'] == '0']
        inf_rows = [r for iters in cases_data[case_key].values() for r in iters if r['interference'] == '1']

        phases = [
            ("Normal\n(50 iters)", normal_rows, s_left + 40),
            ("Interference\n(30 iters)", inf_rows, s_left + 140),
        ]

        # Draw summary gridlines
        for y_val in [0, 10, 20, 30]:
            y_pos = s_bottom - (y_val / 30.0) * plot_height
            draw.line((s_left, y_pos, s_right, y_pos), fill=color_grid, width=1)
            draw.text((s_left - 30, y_pos - 7), f"{y_val}", font=axis_font, fill='#757575')

        for phase_name, p_rows, bx in phases:
            acq = np.mean([float(r['pccl_acquire_us']) for r in p_rows]) / 1000.0
            barr = np.mean([float(r['pccl_start_barrier_us']) for r in p_rows]) / 1000.0
            loop = np.mean([float(r['pccl_loop_us']) for r in p_rows]) / 1000.0
            dma = np.mean([float(r['dma_busy_us']) for r in p_rows]) / 1000.0

            sb_w = 48
            sy0 = s_bottom
            sy1 = sy0 - (acq / 30.0) * plot_height
            sy2 = sy1 - (barr / 30.0) * plot_height
            sy3 = sy2 - (loop / 30.0) * plot_height

            draw.rectangle((bx - sb_w / 2, sy1, bx + sb_w / 2, sy0), fill=color_acquire)
            draw.rectangle((bx - sb_w / 2, sy2, bx + sb_w / 2, sy1), fill=color_barrier)
            draw.rectangle((bx - sb_w / 2, sy3, bx + sb_w / 2, sy2), fill=color_loop)

            # Label total on top
            total_stage = acq + barr + loop
            draw.text((bx - 18, sy3 - 18), f"{total_stage:.1f}ms", font=get_font(11, bold=True), fill='#212121')

            # DMA indicator bar adjacent
            dma_y = s_bottom - (dma / 30.0) * plot_height
            draw.line((bx + sb_w / 2 + 6, s_bottom, bx + sb_w / 2 + 6, dma_y), fill=color_dma, width=3)
            draw.text((bx + sb_w / 2 + 2, dma_y - 14), f"{dma:.1f}", font=get_font(10, bold=True), fill=color_dma)

            # Phase label below
            lines = phase_name.split('\n')
            for li, line in enumerate(lines):
                draw.text((bx - 26, s_bottom + 8 + li * 14), line, font=axis_font, fill='#424242')

    im.save(output_path)
    print(f"Generated stage breakdown stacked chart: {output_path}")


def render_slow_iterations_case_study(iterations, output_path):
    """Generate comparative case study chart across normal fast and outlier iterations."""
    width, height = 1850, 1000
    im = Image.new('RGB', (width, height), '#FAFAFA')
    draw = ImageDraw.Draw(im)

    # Fonts
    title_font = get_font(22, bold=True)
    subtitle_font = get_font(14, bold=False)
    group_title_font = get_font(14, bold=True)
    axis_font = get_font(12, bold=False)
    bold_axis_font = get_font(12, bold=True)
    legend_font = get_font(13, bold=False)
    callout_font = get_font(11, bold=True)

    # Title & Subtitle
    draw.text((60, 20), "PCIeCCL AllGather Bottleneck Breakdown & Jitter Attribution Case Study", font=title_font, fill='#212121')
    draw.text((60, 52), "msprof-only: Host Sync, Loop, Copy API and Hardware Execution; waiting causes remain hypotheses", font=subtitle_font, fill='#616161')

    # Color Palette for the 6 comparative metrics
    color_map = {
        'total': ('#37474F', 'Total Host Latency (ms)'),
        'acq': ('#1976D2', 'Host Acquire (ms)'),
        'barr': ('#FF7043', 'Start Barrier (ms)'),
        'loop': ('#388E3C', 'PCCL Loop (ms)'),
        'api': ('#AB47BC', 'Max Copy API (ms)'),
        'fg_dma': ('#E91E63', 'Foreground Max DMA Task (ms)'),
    }

    # Draw Legend on its own dedicated row
    lx = 60
    ly = 85
    for key, (col, label) in color_map.items():
        draw.rectangle((lx, ly, lx + 18, ly + 14), fill=col, outline='#BDBDBD')
        draw.text((lx + 24, ly - 1), label, font=legend_font, fill='#333333')
        lx += len(label) * 8 + 40

    # Extract case study data
    # 6 Case study groups:
    # 1. split_ratio Normal Mean
    # 2. split_ratio iter 23
    # 3. split_ratio iter 57
    # 4. tcp Normal Mean
    # 5. tcp iter 21
    # 6. tcp iter 61

    def calc_group_stats(case_name, iters=None, interference=None):
        selected = [r for r in iterations if case_name in r['case']]
        if iters is not None:
            selected = [r for r in selected if int(r['iter']) in iters]
        if interference is not None:
            selected = [r for r in selected if int(r['interference']) == interference]

        return {
            'acq': np.mean([float(r['pccl_acquire_us']) for r in selected]) / 1000.0,
            'barr': np.mean([float(r['pccl_start_barrier_us']) for r in selected]) / 1000.0,
            'loop': np.mean([float(r['pccl_loop_us']) for r in selected]) / 1000.0,
            'api': np.mean([float(r['foreground_copy_api_max_us']) for r in selected]) / 1000.0,
            'fg_dma': np.mean([float(r['foreground_connected_device_max_us']) for r in selected]) / 1000.0,
            'total': np.mean([float(r['ag_host_us']) for r in selected]) / 1000.0,
        }

    case_studies = [
        ("Split-Ratio\nNormal (Mean)", calc_group_stats('split_ratio_msprof', interference=0), '#E3F2FD', "Steady-State Baseline"),
        ("Split-Ratio Iter 23\n(Extreme Sync Outlier)", calc_group_stats('split_ratio_msprof', iters=[23]), '#FFEBEE', "Root Cause: Host Acquire + Barrier"),
        ("Split-Ratio Iter 57\n(In-Loop Inflation)", calc_group_stats('split_ratio_msprof', iters=[57]), '#E8F5E9', "Observed: Long Loop"),
        ("TCP Normal\n(Mean)", calc_group_stats('tcp_msprof', interference=0), '#E3F2FD', "Steady-State Baseline"),
        ("TCP Iter 21\n(In-Loop Inflation)", calc_group_stats('tcp_msprof', iters=[21]), '#FFF3E0', "Observed: Long Loop"),
        ("TCP Iter 61\n(In-Loop Inflation)", calc_group_stats('tcp_msprof', iters=[61]), '#E8F5E9', "Observed: Long Loop"),
    ]

    # Main plotting area
    p_left, p_right = 110, width - 80
    p_top, p_bottom = 125, 680
    plot_height = p_bottom - p_top
    max_y = 50.0  # ms scale

    # Container background
    draw.rounded_rectangle((p_left - 40, p_top - 15, p_right + 30, p_bottom + 110), radius=8, fill='#FFFFFF', outline='#E0E0E0', width=1)

    # Gridlines
    for y_val in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        y_pos = p_bottom - (y_val / max_y) * plot_height
        draw.line((p_left, y_pos, p_right, y_pos), fill='#EEEEEE', width=1)
        draw.text((p_left - 50, y_pos - 7), f"{y_val}ms", font=axis_font, fill='#757575')

    # Draw grouped bars
    group_width = (p_right - p_left) / len(case_studies)
    metrics = ['total', 'acq', 'barr', 'loop', 'api', 'fg_dma']
    bar_w = 26
    cluster_w = len(metrics) * bar_w

    for gi, (title, stats, bg_tint, verdict) in enumerate(case_studies):
        gx_center = p_left + (gi + 0.5) * group_width
        gx_start = gx_center - cluster_w / 2

        # Background tint for group card
        draw.rounded_rectangle((gx_center - group_width / 2 + 10, p_top, gx_center + group_width / 2 - 10, p_bottom), radius=6, fill=bg_tint, outline='#EEEEEE')

        # Verdict badge at top of group
        draw.text((gx_center - len(verdict) * 3 - 10, p_top + 10), verdict, font=get_font(10, bold=True), fill='#D32F2F' if 'Outlier' in verdict or 'Acquire' in verdict else '#2E7D32')

        # Draw bars for each metric
        for mi, metric in enumerate(metrics):
            val = stats[metric]
            col = color_map[metric][0]
            bx1 = gx_start + mi * bar_w + 3
            bx2 = bx1 + bar_w - 6
            by_top = p_bottom - (min(val, max_y) / max_y) * plot_height
            draw.rectangle((bx1, by_top, bx2, p_bottom), fill=col)

            # Value label on top of bar
            val_str = f"{val:.2f}" if val < 1.0 else f"{val:.1f}"
            lbl_font = get_font(9 if len(val_str) > 4 else 10, bold=True)
            draw.text((bx1 - 2, max(p_top + 30, by_top - 15)), val_str, font=lbl_font, fill='#212121')

        # Group label below axis
        lines = title.split('\n')
        for li, line in enumerate(lines):
            draw.text((gx_center - len(line) * 3.5, p_bottom + 12 + li * 16), line, font=group_title_font if li == 0 else axis_font, fill='#212121')

    # Bottom Diagnostic Summary Box
    box_top = p_bottom + 140
    box_bottom = height - 25
    draw.rounded_rectangle((60, box_top, width - 60, box_bottom), radius=8, fill='#FAFAFA', outline='#BDBDBD', width=1)
    draw.text((80, box_top + 14), "Measured Stages; Causal Attribution Requires Ablation", font=get_font(15, bold=True), fill='#B71C1C')

    conclusions = [
        (
            "1. Hardware Task Execution:",
            "Short DMA execution does not exclude pre-execution queueing. Cross-domain overlap requires clock validation.",
            '#2E7D32'
        ),
        (
            "2. Host Synchronization Drives Extreme Outliers:",
            "In split_ratio iter23 (47.0ms outer latency), Host Acquire (13.0ms avg, up to 28.3ms on rank2) and Start Barrier (12.2ms avg, up to 16.6ms on rank3) contribute 25.2ms (54.2% of total duration). The loop itself runs in only 14.0ms.",
            '#C62828'
        ),
        (
            "3. In-Loop Inflation:",
            "Long loops require API/task correlation and dependency measurements; polling, queueing and host scheduling are not yet separated.",
            '#E65100'
        ),
    ]

    c_y = box_top + 42
    for heading, text, h_col in conclusions:
        draw.text((80, c_y), heading, font=get_font(12, bold=True), fill=h_col)
        draw.text((610, c_y), text, font=get_font(12, bold=False), fill='#333333')
        c_y += 30

    im.save(output_path)
    print(f"Generated slow iterations case study chart: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--summary',
        type=Path,
        default=Path('tests/hccl_test/images/msprof_only_20261005/summary.json'),
        help='Path to summary.json',
    )
    parser.add_argument(
        '--iterations',
        type=Path,
        default=Path('tests/hccl_test/images/msprof_only_20261005/iterations.csv'),
        help='Path to iterations.csv',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('tests/hccl_test/images/msprof_only_20261005'),
        help='Directory to save generated charts',
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, iterations = load_dataset(args.summary, args.iterations)

    chart1_path = args.output_dir / 'stage_breakdown_stacked.png'
    chart2_path = args.output_dir / 'slow_iterations_case_study.png'

    render_stage_breakdown_stacked(iterations, chart1_path)
    render_slow_iterations_case_study(iterations, chart2_path)
    print("All attribution charts generated successfully!")


if __name__ == '__main__':
    main()
