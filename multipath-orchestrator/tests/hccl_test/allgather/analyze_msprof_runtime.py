#!/usr/bin/env python3
"""Stream msprof executor APIs; all durations/timelines stay in Host domain.

API elapsed time includes descheduling, not just CPU execution. Outside-API
gaps are not CPU idle. No custom traces or Host/device clock subtraction.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
from decimal import Decimal
import json
from pathlib import Path
import statistics
from analyze_msprof import iter_events, marker, union_us

APIS = ('aclrtQueryEventStatus', 'aclrtRecordEvent', 'aclrtMemcpyAsync')


def runtime_metrics(start, end, events):
    intervals = sorted((max(start, t), min(end, t+d)) for _, t, d in events)
    merged = []
    for a, b in intervals:
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    edges = [[start, start]] + merged + [[end, end]]
    result = dict(api_count=len(events), api_sum_us=sum(d for _, _, d in events),
                  api_union_us=sum(b-a for a, b in merged),
                  outside_api_gap_max_us=max(b[0]-a[1] for a, b in zip(edges, edges[1:])))
    for name in APIS:
        durations = [d for n, _, d in events if n == name]
        result.update({name+'_count': len(durations), name+'_sum_us': sum(durations),
                       name+'_max_us': max(durations, default=0)})
    return result


def nested_metrics(outer, inner):
    outer = sorted(outer, key=lambda e: e[1])
    starts = [e[1] for e in outer]
    children = defaultdict(list)
    unmatched = 0
    for name, start, duration in inner:
        i = bisect_right(starts, start)-1
        if i >= 0 and start+duration <= outer[i][1]+outer[i][2]+1e-6:
            children[i].append((name, start, duration))
        else:
            unmatched += 1
    rows = []
    for i, (name, start, duration) in enumerate(outer):
        nested = children[i]
        covered = union_us((t, t+d) for _, t, d in nested)
        rows.append(dict(api=name, relative_start_us=start, acl_us=duration,
                         runtime_count=len(nested), runtime_union_us=covered,
                         runtime_max_us=max((d for _, _, d in nested), default=0),
                         runtime_names=','.join(sorted({n for n, _, _ in nested})),
                         acl_outside_runtime_us=max(0, duration-covered),
                         unmatched_runtime_count=unmatched))
    return rows


def collect_runtime(events, windows, origin=0, include_nested=False):
    origin = Decimal(str(origin))
    by_thread = defaultdict(list)
    for window in windows:
        by_thread[str(window['tid'])].append(window)
    index = {}
    for tid, group in by_thread.items():
        group.sort(key=lambda w: w['start'])
        if any(a['end'] > b['start'] for a, b in zip(group, group[1:])):
            raise ValueError('ambiguous overlapping executor loop markers')
        index[tid] = ([w['start'] for w in group], group)
    for event in events:
        name = event.get('name', '')
        inner = include_nested and name.startswith('Runtime@')
        if event.get('ph') != 'X' or not (name.startswith('AscendCL@') or inner):
            continue
        api = name.split('@')[-1]
        if (api not in APIS and not inner) or str(event.get('tid')) not in index:
            continue
        starts, group = index[str(event['tid'])]
        # Subtract epoch before float conversion: float epoch microseconds
        # quantize every short call to 0.25us, biasing large interval unions.
        t = float(Decimal(str(event['ts']))-origin)
        i = bisect_right(starts, t)-1
        if i >= 0 and t < group[i]['end']:
            group[i]['inner' if inner else 'events'].append((api, t, float(event.get('dur', 0))))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stages-only', action='store_true', help='read small msproftx files only')
    parser.add_argument('--nested', action='store_true', help='audit Runtime events nested inside ACL calls')
    parser.add_argument('--rank', type=int, help='analyze only one rank')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = defaultdict(list)
    for row in csv.DictReader((args.data/'iterations.csv').open()):
        if args.rank is not None and int(row['rank']) != args.rank:
            continue
        sources[row['source']].append(row)
    results, long_calls, stages, nested_rows = [], [], [], []
    for source, records in sources.items():
        path = Path(source)
        anchors = {}
        for event in iter_events(next(path.parent.glob('msprof_tx_*.json'))):
            name, fields = marker(event)
            if event.get('ph') == 'X' and name in {
                    'PCCL.batch', 'PCCL.acquire', 'PCCL.start_barrier', 'PCCL.loop',
                    'PCCL.flush', 'PCCL.finish_barrier_signal'}:
                key = (fields.get('rank'), fields.get('comm'), fields.get('seq'), name)
                if key in anchors:
                    raise ValueError('duplicate stage marker: '+str(key))
                anchors[key] = event
        origin = min(Decimal(str(e['ts'])) for k, e in anchors.items() if k[-1] == 'PCCL.loop')
        windows = []
        for row in records:
            key = (row['rank'], row['pccl_comm'], row['pccl_seq'])
            event = anchors.get(key+('PCCL.loop',))
            if event is None:
                raise ValueError('missing loop marker: '+str(key))
            identity = dict(case=row['case'], rank=int(row['rank']), iter=int(row['iter']),
                            interference=int(row['interference']), comm=row['pccl_comm'], seq=row['pccl_seq'])
            start = float(Decimal(str(event['ts']))-origin)
            windows.append(dict(start=start, end=start+float(event['dur']), tid=event['tid'],
                                identity=identity, events=[], inner=[]))
            for stage in ('batch', 'acquire', 'start_barrier', 'loop', 'flush', 'finish_barrier_signal'):
                anchor = anchors.get(key+('PCCL.'+stage,))
                if anchor is not None:
                    stages.append(dict(**identity, stage=stage, start_us=float(anchor['ts']),
                                       end_us=float(anchor['ts'])+float(anchor['dur']), duration_us=float(anchor['dur'])))
        if not args.stages_only:
            collect_runtime(iter_events(path), windows, origin, args.nested)
        for window in ([] if args.stages_only else windows):
            results.append(dict(**window['identity'], loop_start_us=float(origin)+window['start'],
                                loop_us=window['end']-window['start'],
                                **runtime_metrics(window['start'], window['end'], window['events'])))
            for api, start, duration in window['events']:
                if duration >= 100:
                    long_calls.append(dict(**window['identity'], api=api, start_us=float(origin)+start,
                                           duration_us=duration, end_us=float(origin)+start+duration))
            if args.nested:
                for nested in nested_metrics(window['events'], window['inner']):
                    nested_rows.append(dict(**window['identity'], **nested,
                                            start_us=float(origin)+nested['relative_start_us']))
        print(json.dumps(dict(source=source, windows=len(windows))), flush=True)
    write_csv(args.output/'runtime_iterations.csv', results)
    write_csv(args.output/'runtime_long_calls.csv', long_calls)
    write_csv(args.output/'host_stages.csv', stages)
    write_csv(args.output/'runtime_nested_calls.csv', nested_rows)
    phases = []
    for flag in (0, 1):
        selected = [r for r in results if r['interference'] == flag]
        if selected:
            numeric = ['loop_us', 'api_union_us', 'outside_api_gap_max_us'] + [n+s for n in APIS for s in ('_count', '_sum_us', '_max_us')]
            phases.append(dict(interference=flag, records=len(selected),
                               means={k: statistics.mean(r[k] for r in selected) for k in numeric},
                               slowest=sorted(selected, key=lambda r: r['loop_us'], reverse=True)[:12]))
    (args.output/'runtime_summary.json').write_text(json.dumps(dict(records=len(results),
        stage_records=len(stages), sources=list(sources), stages_only=args.stages_only,
        long_calls=len(long_calls), phases=phases, caveat=__doc__), indent=2))


if __name__ == '__main__':
    main()
