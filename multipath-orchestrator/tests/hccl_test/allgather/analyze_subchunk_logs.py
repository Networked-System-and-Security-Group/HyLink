#!/usr/bin/env python3
"""Batch raw AG logs and correctly aligned prior-round path feedback."""
import argparse
import csv
import json
from pathlib import Path
import sys

from analyze_msprof import parse_benchmark_log, phase_statistics
from compare_msprof_blocks import read_case

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpi_primitives.plot_pcie_allgather import parse_path_metrics


def attach_path_timings(rows, timings):
    if not set(timings).issubset({row['iter'] for row in rows}):
        raise ValueError('Path feedback outside formal iteration identities')
    for row in rows:
        hccl, pcie = timings.get(row['iter'], (None, None))
        row.update(hccl_ms=hccl, pcie_ms=pcie)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports, all_rows = [], []
    for log in args.logs:
        text = log.read_text(errors='replace')
        report, rows = read_case(log)
        if 'Test PASSED.' not in text or any(f'rank {rank}: PASSED' not in text for rank in range(4)):
            raise ValueError('Missing four-rank correctness pass: ' + str(log))
        if {row['iter'] for row in rows} != set(range(80)) or len(rows) != 80:
            raise ValueError('Expected exactly 80 distinct formal iterations')
        metrics = parse_path_metrics(log, expected_size_bytes=268435456, expected_ranks=4)
        if set(metrics) != set(range(79)) or any(m.timing is None for m in metrics.values()):
            raise ValueError('Expected 79 aligned path feedback records: ' + str(log))
        attach_path_timings(rows, {iteration: (m.timing.hccl_ms, m.timing.pcie_ms)
                                 for iteration, m in metrics.items()})
        path_rows = [dict(iter=row['iter'], interference=row['interference'],
                          event_ms=row['pcie_ms'], bw_GBps=1) for row in rows if row['pcie_ms'] is not None]
        report['pcie_ms_phases'] = {phase: stats['event_ms'] for phase, stats in phase_statistics(path_rows).items()}
        fast_rows = [dict(iter=row['iter'], interference=row['interference'],
                          event_ms=row['hccl_ms'], bw_GBps=1) for row in rows if row['hccl_ms'] is not None]
        report['hccl_ms_phases'] = {phase: stats['event_ms'] for phase, stats in phase_statistics(fast_rows).items()}
        reports.append(report)
        all_rows.extend(rows)
    (args.output/'subchunk_log_summary.json').write_text(json.dumps(reports, indent=2))
    with (args.output/'subchunk_iterations.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps(dict(cases=len(reports), iterations=len(all_rows),
                         caveat='Final iter has no path feedback; msprof elapsed is not CPU time.')))


if __name__ == '__main__':
    main()
