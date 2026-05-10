#!/usr/bin/env python3
"""Validate two complete MPI runs and summarize raw aggregate AllGather bandwidth."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
from statistics import mean, pstdev

RANGES = [(100, 200), (300, 600), (700, 900)]
STAGES = [(0, 100), (100, 200), (200, 300), (300, 600),
          (600, 700), (700, 900), (900, 1001)]


def interfered(iteration):
    return any(start <= iteration < end for start, end in RANGES)


def algorithm_label(path, algorithm):
    if algorithm == 'split_ratio':
        return 'Split ratio'
    with Path(path).open() as file:
        header = file.readline()
    key = 'static_alpha' if algorithm == 'static' else 'tcp_initial_alpha'
    match = re.search(r'\b' + key + r'=([\d.]+)', header)
    if not match:
        return 'Static' if algorithm == 'static' else 'TCP'
    return (f'Static (alpha={match[1]})' if algorithm == 'static'
            else f'TCP (initial alpha={match[1]})')


def interference_label(path):
    header = Path(path).read_text().splitlines()[0]
    match = re.search(r'\bpcie_direction=(d2h|h2d|both)', header)
    return {'d2h': 'D2H', 'h2d': 'H2D', 'both': 'D2H+H2D'}[match[1] if match else 'd2h']


def experiment_label(path):
    header = Path(path).read_text().splitlines()[0]
    fields = dict(re.findall(r'(\w+)=([^\s]+)', header))
    return (f'{fields.get("np", "2")} ranks: '
            f'{int(fields.get("input_bytes", 536870912)) // (1024**2)}MiB input/rank; '
            f'continuous {int(fields.get("interference_bytes", 268435456)) // (1024**2)}MiB '
            f'{interference_label(path)} interference/rank')


def parse_log(path, iters=1001):
    text = Path(path).read_text(errors='replace')
    header = text.splitlines()[0]
    rank_match = re.search(r'\bnp=(\d+)', header)
    ranks = int(rank_match[1]) if rank_match else 2
    size_match = re.search(r'\btotal_bytes=(\d+)', header)
    output_GiB = int(size_match[1]) / (1024 ** 3) if size_match else 1.0
    matches = re.findall(r'^iter=(\d+) event_ms=([\d.eE+-]+) bw_GBps=([\d.eE+-]+)', text, re.M)
    indices = [int(i) for i, _, _ in matches]
    if indices != list(range(iters)):
        raise ValueError(f'{path}: expected {iters} ordered unique iterations, found {len(indices)}')
    passed = re.findall(r'^rank (\d+): PASSED', text, re.M)
    if sorted(map(int, passed)) != list(range(ranks)):
        raise ValueError(f'{path}: missing {ranks}-rank correctness success: {passed}')
    points = []
    for iteration, elapsed, _ in matches:
        ms = float(elapsed)
        if not math.isfinite(ms) or ms <= 0:
            raise ValueError(f'{path}: invalid event time {elapsed}')
        # Each rank gathers two 512MiB inputs: 1GiB/event seconds. Native
        # benchmark's bw_GBps field counts only the 512MiB input and is ignored.
        points.append((int(iteration), ms, output_GiB * 1000.0 / ms))
    return points


def envelope(points, window=11):
    if window <= 0 or window % 2 == 0:
        raise ValueError('window must be a positive odd integer')
    values = {i: bw for i, _, bw in points}
    output = []
    for iteration, _, _ in points:
        start, end = next((a, b) for a, b in STAGES if a <= iteration < b)
        samples = [values[i] for i in range(max(start, iteration - window // 2),
                   min(end, iteration + window // 2 + 1)) if i in values]
        output.append((iteration, mean(samples), pstdev(samples)))
    return output


def summarize(points):
    result = {}
    selections = [('no_interference', [p for p in points if not interfered(p[0])]),
                  ('interference', [p for p in points if interfered(p[0])])]
    selections += [(f'{a}-{b}', [p for p in points if a <= p[0] < b]) for a, b in STAGES]
    for name, selected in selections:
        values = [p[2] for p in selected]
        avg, std = mean(values), pstdev(values)
        result[name] = dict(count=len(values), mean_GiBps=avg, std_GiBps=std,
                            variance_GiBps2=std ** 2, cv_percent=100 * std / avg,
                            min_GiBps=min(values), max_GiBps=max(values))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--adaptive', choices=('tcp', 'split_ratio'), default='tcp')
    args = parser.parse_args()
    summaries = {}
    with (args.run_dir / 'bandwidth.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['algorithm', 'iter', 'interference', 'event_ms', 'bandwidth_GiBps'])
        for algorithm in (args.adaptive, 'static'):
            points = parse_log(args.run_dir / f'{algorithm}.log')
            summaries[algorithm] = summarize(points)
            writer.writerows((algorithm, i, int(interfered(i)), ms, bw) for i, ms, bw in points)
    (args.run_dir / 'summary.json').write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
