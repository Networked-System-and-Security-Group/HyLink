#!/usr/bin/env python3
"""Compare static=.39 with continuous direction-specific background traffic."""
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from process_results import RANGES, parse_log, summarize, envelope

def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    root = parser.parse_args().run_dir
    data = {mode: parse_log(root / mode / 'static.log') for mode in ('d2h', 'h2d', 'both')}
    summary = {mode: summarize(points) for mode, points in data.items()}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    for bands in (False, True):
        fig, ax = plt.subplots(figsize=(14, 5))
        for mode, points in data.items():
            if bands:
                env = envelope(points)
                x, avg, std = zip(*env)
                line, = ax.plot(x, avg, label=mode.upper(), linewidth=1)
                ax.fill_between(x, [a-s for a,s in zip(avg,std)], [a+s for a,s in zip(avg,std)], color=line.get_color(), alpha=.18)
            else:
                ax.plot([p[0] for p in points], [p[2] for p in points], label=mode.upper(), linewidth=.7)
        for a, b in RANGES:
            ax.axvspan(a, b, color='gray', alpha=.13)
        ax.set(xlabel='Iteration', ylabel='Aggregate AllGather bandwidth (GiB/s)',
               title='2 ranks, 512MiB input/rank, static=0.39; 256MiB per background direction' + ('\n11-iteration mean ± population std (phase-clipped)' if bands else '\nRaw measurements (no smoothing)'))
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
        fig.savefig(root / ('variance_envelope.png' if bands else 'bandwidth_raw.png'), dpi=180)
        plt.close(fig)
    print(json.dumps({mode: {key: value for key, value in stats.items() if key in ('no_interference', 'interference')} for mode, stats in summary.items()}, indent=2))

if __name__ == '__main__':
    main()
