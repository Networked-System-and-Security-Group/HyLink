#!/usr/bin/env python3
"""Validate and compare hold-band runs; account for next-round path feedback."""
import argparse
import csv
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from process_results import parse_log, summarize, envelope, interfered, RANGES


def analyze(path):
    points = parse_log(path)
    text = path.read_text()
    background = [dict(re.findall(r'(\w+)=([^\s]+)', line))
                  for line in text.splitlines() if line.startswith('pcie_background ')]
    expected = {(rank, direction, stage) for rank in range(2)
                for direction in ('d2h', 'h2d') for stage in range(3)}
    assert len(background) == 12
    assert {(int(r['rank']), r['direction'], int(r['stage'])) for r in background} == expected
    assert all(r['status'] == '0' and int(r['attempts']) == int(r['successes']) > 0 for r in background)
    feedback = []
    alpha = None
    for line in text.splitlines():
        match = re.search(r'split_pcie=([\d.]+)', line)
        if match:
            alpha = float(match[1])
        match = re.match(r'iter=(\d+) ', line)
        if match:
            feedback.append((int(match[1]), alpha))
    # Aggregation at launch i+1 describes completed iteration i. The final
    # iteration has no next-round aggregation; do not invent its actual alpha.
    ratios = [(current[0], nxt[1]) for current, nxt in zip(feedback, feedback[1:])]
    assert all(alpha is not None for _, alpha in ratios)
    jumps = sum(abs(b-a) > .05 for (i, a), (j, b) in zip(ratios, ratios[1:])
                if j == i+1 and any(start <= i < j < end for start, end in RANGES))
    report = summarize(points)
    report['interference_alpha_jumps_gt_0.05'] = jumps
    return points, ratios, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    root = parser.parse_args().run_dir
    cases = sorted(p for p in root.iterdir() if p.is_dir() and (p/'tcp.log').exists())
    datasets = {case.name: analyze(case/'tcp.log') for case in cases}
    summary = {name: data[2] for name, data in datasets.items()}
    (root/'summary.json').write_text(json.dumps(summary, indent=2))
    with (root/'summary.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['case', 'off_mean_GiBps', 'on_mean_GiBps', 'on_cv_percent', 'on_alpha_jumps_gt_0.05'])
        for name, stats in summary.items():
            writer.writerow([name, stats['no_interference']['mean_GiBps'],
                             stats['interference']['mean_GiBps'], stats['interference']['cv_percent'],
                             stats['interference_alpha_jumps_gt_0.05']])
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, constrained_layout=True)
    for ax, low in zip(axes, ('0.95', '0.90', '0.85', '0.80')):
        for name, (_, ratios, _) in datasets.items():
            if name.startswith(low):
                ax.plot([i for i, _ in ratios], [a for _, a in ratios], lw=.8, label=name)
        for a, b in RANGES:
            ax.axvspan(a, b, color='gray', alpha=.12)
        ax.set_ylabel('PCIe fraction'); ax.legend(); ax.grid(alpha=.2)
    axes[-1].set_xlabel('Iteration'); fig.savefig(root/'split_ratios.png', dpi=150); plt.close(fig)
    for bands in (False, True):
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, constrained_layout=True)
        for ax, low in zip(axes, ('0.90', '0.85', '0.80')):
            for name, (points, _, _) in datasets.items():
                if not (name.startswith(low) or name.startswith('0.95')):
                    continue
                if bands:
                    x, avg, std = zip(*envelope(points))
                    line, = ax.plot(x, avg, lw=.8, label=name)
                    ax.fill_between(x, [m-s for m,s in zip(avg,std)], [m+s for m,s in zip(avg,std)], color=line.get_color(), alpha=.15)
                else:
                    ax.plot([i for i, _, _ in points], [b for _, _, b in points], lw=.7, label=name)
                for a, b in RANGES:
                    ax.axvspan(a, b, color='gray', alpha=.025)
            ax.set_ylabel('AllGather GiB/s'); ax.legend(); ax.grid(alpha=.2)
        axes[0].set_title('2 ranks, 512MiB/rank, D2H+H2D 128MiB/direction; high=1.10\n' + ('11-iteration mean ± population std, phase-bounded' if bands else 'Raw measurements; no smoothing/filtering'))
        axes[-1].set_xlabel('Iteration')
        fig.savefig(root/('variance_envelope.png' if bands else 'bandwidth_raw.png'), dpi=150); plt.close(fig)
    print(json.dumps({k: {key: value for key, value in v.items() if key in
          ('no_interference', 'interference', 'interference_alpha_jumps_gt_0.05')} for k,v in summary.items()}, indent=2))


if __name__ == '__main__':
    main()
