#!/usr/bin/env python3
"""Audit existing CSV/CPU traces without loading raw msprof timelines.

FIFO matching is conditional evidence, NOT a connection_id-confirmed identity.
Only unique lane/stream assignments satisfying count, byte order and containment
are emitted. Clock alignment tolerance is explicit and negative residuals retained.
"""
import argparse
from collections import defaultdict
import csv
import itertools
import json
from pathlib import Path
import re
import statistics


def match_lane_tasks(host, device, tolerance=50):
    rows, warnings = [], []
    for operation in sorted({e['operation'] for e in host}):
        lanes, streams = defaultdict(list), defaultdict(list)
        for event in host:
            if event['operation'] == operation:
                lanes[event['lane']].append(event)
        for event in device:
            if event['operation'] == operation:
                streams[event['stream']].append(event)
        for group in list(lanes.values()) + list(streams.values()):
            group.sort(key=lambda e: e['start'])
        if len(lanes) != len(streams) or len(lanes) > 8:
            warnings.append(operation + ': lane/stream count mismatch')
            continue
        candidates = []
        lane_ids = sorted(lanes)
        for assignment in itertools.permutations(sorted(streams)):
            valid = True
            for lane, stream in zip(lane_ids, assignment):
                left, right = lanes[lane], streams[stream]
                if len(left) != len(right) or any(
                    a['bytes'] != b['bytes'] or b['start']+tolerance < a['start'] or
                    b['end'] > a['end']+tolerance for a, b in zip(left, right)):
                    valid = False
                    break
            if valid:
                candidates.append(assignment)
                if len(candidates) > 1:
                    break
        if len(candidates) != 1:
            warnings.append(operation + ': ambiguous or incompatible assignment')
            continue
        for lane, stream in zip(lane_ids, candidates[0]):
            for a, b in zip(lanes[lane], streams[stream]):
                rows.append(dict(lane=lane, stream=stream, bytes=a['bytes'], operation=operation,
                    observed_us=a['end']-a['start'],
                    submit_to_device_start_us=b['start']-a['start'],
                    device_execution_us=b['end']-b['start'],
                    device_end_to_observed_us=a['end']-b['end'],
                    host_start_us=a['start'], device_start_us=b['start'], device_end_us=b['end'],
                    host_end_us=a['end']))
    return rows, warnings


def align_host_tasks(host, offset):
    return [dict(e, start=e['start']+offset, end=e['end']+offset) for e in host]


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def baseline_stages(case):
    metadata = json.loads((case/'metadata.json').read_text())
    command = metadata['command']
    warmup = int(next(a.split('=')[1] for a in command if a.startswith('--warmup=')))
    iters = int(next(a.split('=')[1] for a in command if a.startswith('--iters=')))
    ranges = [tuple(map(int, part.split('-'))) for part in next(
        a.split('=')[1] for a in command if a.startswith('--interference-ranges=')).split(',') if part]
    text = (case/'run.log').read_text(errors='replace')
    pattern = (r'\[RANK(\d+)-TIMING seq=(\d+)\] total=([\d.]+) ms \| acq=([\d.]+) '
               r'sbarr=([\d.]+) loop=([\d.]+) dma_flush=([\d.]+) run_barr\+sig=([\d.]+) rel=([\d.]+)')
    parsed = list(re.finditer(pattern, text))
    # Ordinal joining is allowed only for this single-comm, single-op benchmark.
    for rank in range(4):
        seqs = [int(m[2]) for m in parsed if int(m[1]) == rank]
        if sorted(seqs) != list(range(1, warmup+iters+1)):
            raise ValueError(f'{case}: rank {rank} sequences not contiguous; do not guess iter mapping')
    rows = []
    for m in parsed:
        rank, seq = int(m[1]), int(m[2])
        iteration = seq-warmup-1
        if iteration < 0:
            continue
        row = dict(case=case.name, rank=rank, iter=iteration,
                   interference=int(any(a <= iteration <= b for a, b in ranges)))
        row.update(zip(('total_ms','acquire_ms','start_barrier_ms','loop_ms',
                        'flush_ms','finish_ms','release_ms'), map(float, m.group(3,4,5,6,7,8,9))))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tolerance-us', type=float, default=50)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    iterations = list(csv.DictReader((args.data/'iterations.csv').open()))
    hardware = defaultdict(list)
    for path in sorted(args.data.glob('*_device_tasks.csv')):
        for r in csv.DictReader(path.open()):
            if r['kind'] != 'foreground_dma':
                continue
            hardware[(r['source'], int(r['iter']))].append(dict(stream=int(r['stream']),
                start=float(r['start_us']), end=float(r['start_us'])+float(r['duration_us']),
                bytes=int(r['bytes']), operation=r['operation']))
    traces = {}
    anchors = {}
    matched, status = [], []
    for row in iterations:
        case = args.logs/Path(row['case']).name
        rank, iteration = int(row['rank']), int(row['iter'])
        key = (str(case), rank)
        if key not in traces:
            paths = list(case.glob(f'rank_{rank}_pid_*.json'))
            if len(paths) != 1:
                raise ValueError(f'{case}: expected one CPU trace for rank {rank}')
            data = json.loads(paths[0].read_text())
            epoch = data['wall_epoch_us']
            grouped = defaultdict(list)
            cpu_loops = {}
            for e in data['traceEvents']:
                if e.get('name') == f'Rank {rank} Graph Batch':
                    cpu_loops[(str(e['args']['comm_id']), str(e['args']['work_seq']))] = epoch+e['ts']
                if e.get('name') == 'Graph DMA' and e.get('ph') == 'X':
                    fields = e['args']; lane = fields['lane']
                    operation = 'device to host' if lane < 12 else 'host to device' if lane < 24 else 'device to device'
                    grouped[(str(fields['comm_id']), str(fields['work_seq']))].append(
                        dict(lane=lane, start=epoch+e['ts'], end=epoch+e['ts']+e['dur'],
                             bytes=fields['bytes'], operation=operation))
            traces[key] = grouped
            msprof_loops = {}
            for path in (case/'profile').rglob('msprof_tx_*.json'):
                for e in json.loads(path.read_text()):
                    if e.get('name', '').startswith(f'PCCL.loop|rank={rank}|'):
                        fields = dict(part.split('=',1) for part in e['name'].split('|')[1:])
                        marker_key = (fields['comm'], fields['seq'])
                        if marker_key in msprof_loops:
                            raise ValueError('duplicate clock anchor')
                        msprof_loops[marker_key] = float(e['ts'])
            anchors[key] = {k: msprof_loops[k]-t for k, t in cpu_loops.items() if k in msprof_loops}
        work_key = (row['pccl_comm'], row['pccl_seq'])
        if work_key not in anchors[key]:
            raise ValueError('missing local clock anchor')
        offset = anchors[key][work_key]
        host = align_host_tasks(traces[key][work_key], offset)
        device = hardware[(row['source'], iteration)]
        result, warnings = match_lane_tasks(host, device, args.tolerance_us)
        status.append(dict(case=case.name, rank=rank, iter=iteration,
            host_tasks=len(host), device_tasks=len(device), matched=len(result),
            clock_offset_us=offset, warnings=warnings))
        for r in result:
            matched.append(dict(case=case.name, rank=rank, iter=iteration, **r))
    write_csv(args.output/'conditional_task_matches.csv', matched)
    baseline = []
    for case in sorted(args.logs.glob('*_baseline')):
        baseline.extend(baseline_stages(case))
    write_csv(args.output/'baseline_stage_iterations.csv', baseline)
    phase_metrics = []
    for case in sorted({Path(r['case']).name for r in iterations}):
        for flag in ('0', '1'):
            selected = [r for r in iterations if Path(r['case']).name == case and r['interference'] == flag]
            phase_metrics.append(dict(case=case, interference=int(flag), records=len(selected),
                **{key+'_mean_us': statistics.mean(float(r[key]) for r in selected)
                   for key in ('h2h_queue_max_us','h2h_copy_max_us','foreground_dma_max_task_us')}))
    samples = []
    for case in sorted({r['case'] for r in matched}):
        for iteration in (21,23,57,61):
            selected = [r for r in matched if r['case'] == case and r['iter'] == iteration]
            if selected:
                worst = sorted(selected, key=lambda r:r['observed_us'], reverse=True)[:6]
                samples.append(dict(case=case, iter=iteration, matched_tasks=len(selected), worst=worst))
    summary = dict(tolerance_us=args.tolerance_us, matching_method='conditional_unique_FIFO_NOT_connection_id',
                   clock_alignment='per_work_PCCL_loop_host_anchor; includes marker-to-trace boundary uncertainty',
                   rank_iter_records=len(status), matched_tasks=len(matched),
                   complete_matches=sum(s['matched'] == s['host_tasks'] == s['device_tasks'] for s in status),
                   phase_metrics=phase_metrics, samples=samples, matching_status=status)
    (args.output/'audit_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({key:summary[key] for key in ('rank_iter_records','matched_tasks','complete_matches')}))


if __name__ == '__main__':
    main()
