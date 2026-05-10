#!/usr/bin/env python3
"""Window mean ± one population standard deviation; never filter raw samples."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from process_results import RANGES, envelope, parse_log, algorithm_label, experiment_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--adaptive', choices=('tcp', 'split_ratio'), default='tcp')
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    for a, b in RANGES:
        ax.axvspan(a, b, color='grey', alpha=0.14)
    with (args.run_dir / 'variance_envelope.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['algorithm', 'iter', 'window_mean_GiBps', 'std_GiBps', 'variance_GiBps2'])
        for algorithm, color in [('static', '#167ca6'), (args.adaptive, '#ec8a13')]:
            stats = envelope(parse_log(args.run_dir / f'{algorithm}.log'))
            x, center, std = zip(*stats)
            writer.writerows((algorithm, i, avg, dev, dev * dev) for i, avg, dev in stats)
            ax.fill_between(x, [m-s for m, s in zip(center, std)],
                            [m+s for m, s in zip(center, std)], color=color, alpha=0.25)
            ax.plot(x, center, color=color, linewidth=1.2,
                    label=algorithm_label(args.run_dir / f'{algorithm}.log', algorithm))
    ax.set(xlabel='Iteration', ylabel='Aggregate AllGather bandwidth (GiB/s)',
           title=f'{experiment_label(args.run_dir / "static.log")}\n11-iteration mean ±1 std envelope (phase-bounded windows)',
           xlim=(0, 1000), ylim=(0, None))
    ax.legend(loc='upper right')
    ax.grid(alpha=0.2)
    fig.savefig(args.run_dir / 'bandwidth_variance_envelope.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
