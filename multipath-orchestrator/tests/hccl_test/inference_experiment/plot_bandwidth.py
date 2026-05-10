#!/usr/bin/env python3
"""Plot every original sample, with no smoothing or spike removal."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from process_results import RANGES, parse_log, algorithm_label, experiment_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--adaptive', choices=('tcp', 'split_ratio'), default='tcp')
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    for a, b in RANGES:
        ax.axvspan(a, b, color='grey', alpha=0.14)
    for algorithm, color in [('static', '#167ca6'), (args.adaptive, '#ec8a13')]:
        points = parse_log(args.run_dir / f'{algorithm}.log')
        ax.plot([p[0] for p in points], [p[2] for p in points], color=color,
                linewidth=0.9, label=algorithm_label(args.run_dir / f'{algorithm}.log', algorithm))
    ax.set(xlabel='Iteration', ylabel='Aggregate AllGather bandwidth (GiB/s)',
           title=f'{experiment_label(args.run_dir / "static.log")} — raw samples',
           xlim=(0, 1000), ylim=(0, None))
    ax.legend(loc='upper right')
    ax.grid(alpha=0.2)
    fig.savefig(args.run_dir / 'bandwidth_raw.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
