#!/usr/bin/env python3
"""Summarize msprof-only, connection_id-linked foreground task CSVs.

Gaps mean no foreground task executing; NOT an idle PCIe bus (background may
occupy it). Host/device differences are not asserted to be queue latency.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def metrics(rows, background=()):
    intervals = sorted((float(r['device_start_us']),
                        float(r['device_start_us'])+float(r['device_execution_us'])) for r in rows)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    apis = {r['connection_id']:float(r['api_us']) for r in rows}
    gaps = [(a[1],b[0]) for a,b in zip(merged,merged[1:])]
    largest = max(gaps,key=lambda pair:pair[1]-pair[0], default=(0,0))
    covered = []
    for start,end in sorted(background):
        start,end = max(start,largest[0]),min(end,largest[1])
        if end <= start:
            continue
        if covered and start <= covered[-1][1]:
            covered[-1][1] = max(covered[-1][1],end)
        else:
            covered.append([start,end])
    return dict(tasks=len(rows), copy_api_count=len(apis), copy_api_sum_us=sum(apis.values()),
                copy_api_max_us=max(apis.values()),
                hardware_span_us=merged[-1][1]-merged[0][0],
                foreground_busy_union_us=sum(b-a for a,b in merged),
                foreground_gap_max_us=max((b[0]-a[1] for a,b in zip(merged,merged[1:])), default=0),
                foreground_max_gap_background_overlap_us=sum(b-a for a,b in covered),
                device_task_max_us=max(float(r['device_execution_us']) for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    args = parser.parse_args()
    report = json.loads((args.data/'summary.json').read_text())
    rows = []
    for source in report['files']:
        path = Path(source['source'])
        case = next(p.name for p in path.parents if (p/'run.log').exists())
        connected = args.data/(path.parent.parent.name+'_connected_tasks.csv')
        groups = defaultdict(list)
        # Full task durations are retained in the old device CSV even when
        # selected by AG overlap. Deduplicate across windows. Presence is
        # evidence; absence is NOT proof of no background activity.
        background = set()
        device_csv = args.data/(path.parent.parent.name+'_device_tasks.csv')
        for r in csv.DictReader(device_csv.open()):
            if r['kind'] == 'background_dma':
                start = float(r['start_us'])
                background.add((start,start+float(r['duration_us'])))
        if not connected.exists():
            raise ValueError('missing connected tasks: '+str(connected))
        for r in csv.DictReader(connected.open()):
            groups[(int(r['rank']),int(r['iter']))].append(r)
        for (rank, iteration), tasks in sorted(groups.items()):
            row = dict(case=case, rank=rank, iter=iteration,
                       interference=int(tasks[0]['interference']), loop_us=float(tasks[0]['loop_us']),
                       **metrics(tasks, background))
            row['host_to_device_gap_median_us'] = statistics.median(float(t['api_end_to_device_start_us']) for t in tasks)
            row['last_device_end_minus_loop_end_us'] = max(float(t['device_end_minus_loop_end_us']) for t in tasks)
            rows.append(row)
    with (args.data/'connected_iterations.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for case in sorted({r['case'] for r in rows}):
        for flag in (0,1):
            selected = [r for r in rows if r['case']==case and r['interference']==flag]
            summary.append(dict(case=case, interference=flag, records=len(selected),
                means={k:statistics.mean(r[k] for r in selected) for k in
                    ('loop_us','copy_api_sum_us','copy_api_max_us','hardware_span_us',
                     'foreground_busy_union_us','foreground_gap_max_us',
                     'foreground_max_gap_background_overlap_us','device_task_max_us')},
                slowest=sorted(selected,key=lambda r:r['loop_us'],reverse=True)[:4]))
    result = dict(records=len(rows), phases=summary,
                  caveat='same-device-domain foreground gaps, not bus idle or proven polling; cross-domain differences uncalibrated')
    (args.data/'connected_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(records=len(rows), phases=len(summary))))


if __name__ == '__main__':
    main()
