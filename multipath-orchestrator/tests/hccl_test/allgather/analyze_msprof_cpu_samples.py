#!/usr/bin/env python3
"""Batch sparse CPU/cswitch payloads from small msprof_tx JSON files only.

Metadata range duration is NOT the measured API duration. Samples are not
extrapolated into full-iteration critical-path costs or automatic root causes.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
from decimal import Decimal
import json
from pathlib import Path
import statistics
from analyze_msprof import iter_events, marker, parse_benchmark_log, phase_statistics


def extract_cpu_samples(events):
    events = list(events)
    loops, ags, samples = {}, [], []
    for event in events:
        if event.get('ph') != 'X':
            continue
        name, fields = marker(event)
        key = (str(event.get('pid')), fields.get('rank'), fields.get('comm'), fields.get('seq'))
        if name == 'PCCL.loop':
            if key in loops:
                raise ValueError('duplicate loop marker')
            loops[key] = (event, fields)
        elif name == 'AG':
            ags.append((event, fields))
        elif name == 'PCCL.api_cpu':
            samples.append((event, fields, key))
    rows, unmatched = [], 0
    for event, fields, key in samples:
        parent = loops.get(key)
        if parent is None:
            unmatched += 1
            continue
        loop, loop_fields = parent
        matches = [(ag, attrs) for ag, attrs in ags
                   if ag.get('pid') == event.get('pid') and attrs.get('rank') == fields.get('rank')
                   and float(ag['ts']) <= float(loop['ts']) < float(ag['ts'])+float(ag['dur'])]
        if len(matches) != 1:
            unmatched += 1
            continue
        ag, attrs = matches[0]
        if attrs.get('phase') != 'formal':
            continue
        wall, cpu = int(fields['wall_ns'])/1000, int(fields['cpu_ns'])/1000
        rows.append(dict(rank=int(fields['rank']), iter=int(attrs['iter']),
                         interference=int(attrs.get('interference', 0)), comm=fields['comm'], seq=fields['seq'],
                         api=fields['api'], node=fields.get('node', ''), subchunk=fields.get('sc', ''),
                         lane=fields.get('lane', ''), tid=event.get('tid'), sample_end_us=float(event['ts']),
                         loop_us=float(loop['dur']), wall_us=wall, cpu_us=cpu,
                         offcpu_estimate_us=max(0, wall-cpu),
                         voluntary=int(fields['nv']), involuntary=int(fields['ni']),
                         valid=int(fields['valid'])))
    return rows, unmatched


def link_acl_samples(rows, events):
    names = {'query': 'aclrtQueryEventStatus', 'record': 'aclrtRecordEvent', 'submit': 'aclrtMemcpyAsync'}
    wanted = {(str(r['tid']), names[r['api']]) for r in rows if r['api'] in names}
    groups = defaultdict(list)
    for event in events:
        name = event.get('name', '')
        key = (str(event.get('tid')), name.split('@')[-1])
        if event.get('ph') == 'X' and name.startswith('AscendCL@') and key in wanted:
            groups[key].append((Decimal(str(event['ts'])), float(event['dur'])))
    indexes = {}
    for key, group in groups.items():
        group.sort()
        indexes[key] = ([e[0] for e in group], group)
    for row in rows:
        row.update(acl_us='', measurement_minus_acl_us='', acl_end_to_marker_us='')
        key = (str(row['tid']), names.get(row['api']))
        if key not in indexes:
            continue
        starts, group = indexes[key]
        stamp = Decimal(str(row['sample_end_us']))
        i = bisect_right(starts, stamp)-1
        if i < 0:
            continue
        start, duration = group[i]
        gap = float(stamp-start)-duration
        # One foreground call is followed immediately by sample emission.
        # Do not silently associate far-away calls, or a still-running call.
        if gap < -1 or gap > max(100, row['wall_us']):
            continue
        row.update(acl_us=duration, measurement_minus_acl_us=row['wall_us']-duration,
                   acl_end_to_marker_us=gap)


def extract_host_stages(events):
    events = list(events)
    ags = [(e, marker(e)[1]) for e in events if e.get('ph') == 'X' and marker(e)[0] == 'AG']
    identities = {}
    for e in events:
        name, fields = marker(e)
        if e.get('ph') != 'X' or name != 'PCCL.loop':
            continue
        matches = [a for ag, a in ags if a.get('phase') == 'formal' and ag.get('pid') == e.get('pid')
                   and a.get('rank') == fields.get('rank') and float(ag['ts']) <= float(e['ts']) < float(ag['ts'])+float(ag['dur'])]
        if len(matches) == 1:
            key = (e.get('pid'), fields.get('rank'), fields.get('comm'), fields.get('seq'))
            if key in identities:
                raise ValueError('duplicate formal work identity')
            identities[key] = matches[0]
    rows = []
    stages = {'batch', 'acquire', 'start_barrier', 'loop', 'flush', 'finish_barrier_signal'}
    for e in events:
        name, fields = marker(e)
        if e.get('ph') != 'X' or name.removeprefix('PCCL.') not in stages:
            continue
        key = (e.get('pid'), fields.get('rank'), fields.get('comm'), fields.get('seq'))
        if key not in identities:
            continue
        attrs = identities[key]
        start, duration = float(e['ts']), float(e['dur'])
        rows.append(dict(rank=int(fields['rank']), iter=int(attrs['iter']), interference=int(attrs.get('interference', 0)),
                         comm=fields['comm'], seq=fields['seq'], stage=name[5:], start_us=start,
                         end_us=start+duration, duration_us=duration))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--link-acl', action='store_true', help='stream raw timelines to audit timer overhead')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, reports, host_stages, work_rows = [], [], [], []
    for root in args.inputs:
        logs = sorted(root.rglob('run.log'))
        for log in logs:
            metadata = json.loads(log.with_name('metadata.json').read_text())
            period = int(metadata['environment'].get('PCCL_MSPROF_CPU_SAMPLE_PERIOD', '0'))
            parsed = parse_benchmark_log(log)
            reports.append(dict(case=str(log.parent), sample_period=period,
                                benchmark_phases=phase_statistics(parsed[0])))
        # Intermediate host/timeline exports duplicate finalized exports.
        paths = sorted(p for p in root.rglob('msprof_tx_*.json')
                       if p.parent.name in ('mindstudio_profiler_output', 'mindspeed_output'))
        for path in paths:
            events = list(iter_events(path))
            current, unmatched = extract_cpu_samples(events)
            if args.link_acl and current:
                raw = path.with_name(path.name.replace('msprof_tx_', 'msprof_'))
                if not raw.exists():
                    raise ValueError('missing raw msprof timeline: '+str(raw))
                link_acl_samples(current, iter_events(raw))
            case = next(p for p in path.parents if (p/'run.log').exists())
            for row in current:
                row.update(case=str(case), source=str(path))
            for stage in extract_host_stages(events):
                stage['case'] = str(case)
                host_stages.append(stage)
                if stage['stage'] == 'loop':
                    work_rows.append(dict(source=str(path.with_name(path.name.replace('msprof_tx_', 'msprof_'))),
                                          rank=stage['rank'], iter=stage['iter'], interference=stage['interference'],
                                          pccl_comm=stage['comm'], pccl_seq=stage['seq'], case=str(case)))
            rows.extend(current)
            if unmatched:
                reports.append(dict(source=str(path), unmatched_samples=unmatched))
    if rows:
        with (args.output/'cpu_samples.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if host_stages:
        with (args.output/'host_stages.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(host_stages[0]))
            writer.writeheader()
            writer.writerows(host_stages)
    if work_rows:
        with (args.output/'iterations.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(work_rows[0]))
            writer.writeheader()
            writer.writerows(work_rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['case'], row['rank'], row['iter'])].append(row)
    iterations = []
    for (case, rank, iteration), values in sorted(groups.items()):
        valid = [r for r in values if r['valid']]
        long = [r for r in valid if r['wall_us'] >= 100]
        iterations.append(dict(case=case, rank=rank, iter=iteration, interference=values[0]['interference'],
                               loop_us=values[0]['loop_us'], samples=len(values), invalid_samples=len(values)-len(valid),
                               sampled_wall_us=sum(r['wall_us'] for r in valid),
                               sampled_cpu_us=sum(r['cpu_us'] for r in valid),
                               long_samples=len(long), long_offcpu_us=sum(r['offcpu_estimate_us'] for r in long),
                               long_voluntary=sum(r['voluntary'] for r in long),
                               long_involuntary=sum(r['involuntary'] for r in long)))
    if iterations:
        with (args.output/'cpu_sample_iterations.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(iterations[0]))
            writer.writeheader()
            writer.writerows(iterations)
    result = dict(samples=len(rows), rank_iterations=len(iterations), reports=reports,
                  long_samples=sorted((r for r in rows if r['valid'] and r['wall_us'] >= 100),
                                      key=lambda r: r['wall_us'], reverse=True)[:40], caveat=__doc__)
    (args.output/'cpu_sample_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(samples=len(rows), rank_iterations=len(iterations),
                         cases=sum('benchmark_phases' in r for r in reports))))


if __name__ == '__main__':
    main()
