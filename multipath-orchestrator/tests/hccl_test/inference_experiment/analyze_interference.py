#!/usr/bin/env python3
"""Audit successful background traffic and rank-max path timing without trace."""
import argparse
import json
from pathlib import Path
import re
from statistics import mean
from process_results import interfered, parse_log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('baseline_log', type=Path)
    args = parser.parse_args()
    parse_log(args.run_dir / 'static.log')
    text = (args.run_dir / 'static.log').read_text()
    background = [dict(re.findall(r'(\w+)=([^\s]+)', line))
                  for line in text.splitlines() if line.startswith('pcie_background ')]
    mode = re.search(r'pcie_direction=(d2h|h2d|both)', text.splitlines()[0])
    directions = ('d2h', 'h2d') if mode and mode[1] == 'both' else (mode[1] if mode else 'd2h',)
    for row in background:
        row.setdefault('direction', 'd2h')
    assert len(background) == 6 * len(directions)
    assert {(int(row['rank']), row['direction'], int(row['stage'])) for row in background} == {
        (rank, direction, stage) for rank in (0, 1) for direction in directions for stage in range(3)}
    for row in background:
        assert int(row['status']) == 0 and int(row['attempts']) == int(row['successes']) > 0
        # Host-call occupancy is NOT DMA hardware utilization.
        row['host_call_fraction'] = float(row['copy_seconds']) / float(row['span_seconds'])
    baseline = [dict(re.findall(r'(\w+)=([^\s]+)', line))
                for line in args.baseline_log.read_text().splitlines()
                if line.startswith('background_only ')]
    for row in baseline:
        row.setdefault('direction', 'd2h')
    assert len(baseline) == 2 * len(directions)
    assert {(int(row['rank']), row['direction']) for row in baseline} == {
        (rank, direction) for rank in (0, 1) for direction in directions}
    assert all(int(row['status']) == 0 and int(row['verified']) == 1 for row in baseline)
    rows = []
    fast = pcie = None
    for line in text.splitlines():
        match = re.search(r'\(max\) fast_time=([\d.]+)s', line)
        if match: fast = float(match[1]) * 1000
        match = re.search(r'\(max\) pcie_time=([\d.]+)s', line)
        if match: pcie = float(match[1]) * 1000
        match = re.match(r'iter=(\d+) event_ms=([\d.]+)', line)
        if match:
            assert fast is not None and pcie is not None
            rows.append((int(match[1]), fast, pcie, float(match[2])))
    paths = {}
    for name, enabled in [('no_interference', False), ('interference', True)]:
        selected = [row for row in rows if interfered(row[0]) == enabled]
        paths[name] = dict(count=len(selected), fast_ms=mean(row[1] for row in selected),
                          pcie_ms=mean(row[2] for row in selected),
                          event_ms=mean(row[3] for row in selected),
                          pcie_slower_count=sum(row[2] > row[1] for row in selected))
    report = dict(background_only=baseline, background_with_collective=background,
                  collective_paths=paths, caveat='stage counts include a final copy crossing the end boundary')
    (args.run_dir / 'interference_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
