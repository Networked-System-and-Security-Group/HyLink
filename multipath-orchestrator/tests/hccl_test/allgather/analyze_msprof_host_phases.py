#!/usr/bin/env python3
"""Batch low-frequency Host phases from msprof_tx exports, never custom traces.

CPU payloads use CLOCK_MONOTONIC / THREAD_CPUTIME on the same host. Barrier
publication is bounded by two clock samples; post-publication delays are
bounds, not precise store timestamps. Nonrunning time may include blocking.
"""
import argparse
from collections import defaultdict
import csv
from decimal import Decimal
import json
from pathlib import Path
import re
import statistics
from analyze_msprof import iter_events, marker


def extract_host_phases(events):
    events = [e for e in events if e.get('ph') == 'X']
    ags = [(e, marker(e)[1]) for e in events if marker(e)[0] == 'AG']
    identities, ranges = {}, {}
    for e in events:
        name, fields = marker(e)
        key = (e.get('pid'), fields.get('rank'), fields.get('comm'), fields.get('seq'))
        if name == 'PCCL.loop':
            start = Decimal(str(e['ts']))
            matches = [a for ag, a in ags if ag.get('pid') == e.get('pid')
                       and a.get('rank') == fields.get('rank')
                       and Decimal(str(ag['ts'])) <= start < Decimal(str(ag['ts']))+Decimal(str(ag['dur']))]
            if len(matches) != 1:
                raise ValueError('ambiguous or missing AG identity: '+str(key))
            if matches[0].get('phase') == 'formal':
                if key in identities:
                    raise ValueError('duplicate formal work: '+str(key))
                identities[key] = matches[0]
        elif name.startswith('PCCL.host.'):
            phase_key = key+(name.removeprefix('PCCL.host.'),)
            if phase_key in ranges:
                raise ValueError('duplicate Host phase: '+str(phase_key))
            ranges[phase_key] = e
    rows, points, unmatched = [], [], 0
    for e in events:
        name, fields = marker(e)
        if name != 'PCCL.host_cpu' and name not in {'PCCL.host.enqueued', 'PCCL.host.dequeued'}:
            continue
        if name == 'PCCL.host_cpu' and 'p' in fields:
            aliases = dict(p='phase', r='rank', g='comm', s='seq', b='begin_ns',
                           w='wall_ns', c='cpu_ns', v='vcs', i='ivcs', k='ok')
            fields = {aliases.get(k, k): v for k, v in fields.items()}
        key = (e.get('pid'), fields.get('rank'), fields.get('comm'), fields.get('seq'))
        if key not in identities:
            # Known warmups are intentionally excluded, other unmatched records
            # remain visible in the report (including absent completion tails).
            unmatched += 1
            continue
        attrs = identities[key]
        common = dict(pid=e.get('pid'), rank=int(fields['rank']), iter=int(attrs['iter']),
                      interference=int(attrs.get('interference', 0)), comm=fields['comm'],
                      seq=fields['seq'], tid=e.get('tid'))
        if name != 'PCCL.host_cpu':
            points.append(dict(common, phase=name.removeprefix('PCCL.host.'),
                               start_us=str(e['ts']), duration_us=float(e['dur'])))
            continue
        if fields.get('ok') != '1':
            continue
        wall, cpu = int(fields['wall_ns']), int(fields['cpu_ns'])
        if wall < 0 or cpu < 0:
            raise ValueError('negative valid clock measurement')
        phase = fields['phase']
        parent = ranges.get(key+(phase,))
        if parent is None:
            raise ValueError('missing phase range: '+str(key+(phase,)))
        begin = int(fields['begin_ns'])
        rows.append(dict(common, phase=phase, begin_ns=begin, end_ns=begin+wall,
                         wall_us=wall/1000, cpu_us=cpu/1000,
                         nonrunning_us=max(0, wall-cpu)/1000,
                         voluntary=int(fields['vcs']), involuntary=int(fields['ivcs']),
                         marker_start_us=str(parent['ts']), marker_duration_us=float(parent['dur'])))
    return rows, points, unmatched


def extract_host_ranges(events, cpu_rows=None):
    events = list(events)
    if cpu_rows is None:
        cpu_rows = extract_host_phases(events)[0]
    samples = {(r['pid'], r['rank'], r['comm'], r['seq'], r['phase']): r for r in cpu_rows}
    ags = [(e, marker(e)[1]) for e in events if e.get('ph') == 'X' and marker(e)[0] == 'AG']
    identities = {}
    for e in events:
        name, fields = marker(e)
        if e.get('ph') != 'X' or name != 'PCCL.loop':
            continue
        matches = [a for ag, a in ags if a.get('phase') == 'formal' and ag.get('pid') == e.get('pid')
                   and a.get('rank') == fields.get('rank')
                   and Decimal(str(ag['ts'])) <= Decimal(str(e['ts'])) < Decimal(str(ag['ts']))+Decimal(str(ag['dur']))]
        if len(matches) == 1:
            identities[(e.get('pid'), fields['rank'], fields['comm'], fields['seq'])] = matches[0]
    rows = []
    for e in events:
        name, fields = marker(e)
        if e.get('ph') != 'X' or not name.startswith('PCCL.host.'):
            continue
        phase = name.removeprefix('PCCL.host.')
        if phase in {'enqueued', 'dequeued'}:
            continue
        key = (e.get('pid'), fields['rank'], fields['comm'], fields['seq'])
        if key not in identities:
            continue
        attrs = identities[key]
        sample = samples.get((key[0], int(key[1]), key[2], key[3], phase), {})
        start, duration = Decimal(str(e['ts'])), Decimal(str(e['dur']))
        rows.append(dict(pid=key[0], rank=int(key[1]), comm=key[2], seq=key[3], iter=int(attrs['iter']),
                         interference=int(attrs.get('interference', 0)), phase=phase,
                         begin_ns=int(start*1000), end_ns=int((start+duration)*1000),
                         wall_us=float(duration), cpu_us=sample.get('cpu_us', ''),
                         sample_wall_us=sample.get('wall_us', ''),
                         voluntary=sample.get('voluntary', ''), involuntary=sample.get('involuntary', ''),
                         cpu_sample_present=int(bool(sample)), clock_domain='msprof_host'))
    return rows


def barrier_observations(rows):
    groups = defaultdict(list)
    for r in rows:
        match = re.fullmatch(r'barrier([0-7])_(arrive|wait|publish)', r['phase'])
        if match:
            groups[(r.get('case', ''), r['comm'], r['seq'], int(match[1]))].append((match[2], r))
    output = []
    for (case, comm, seq, slot), group in groups.items():
        arrivals = [r for phase, r in group if phase == 'arrive']
        publishers = [r for phase, r in group if phase == 'publish']
        if not arrivals or len(publishers) != 1:
            continue
        publish = publishers[0]
        last = max(arrivals, key=lambda r: r['end_ns'])
        spread = (last['end_ns']-min(r['end_ns'] for r in arrivals))/1000
        waits = {r['rank']: r for phase, r in group if phase == 'wait'}
        for arrival in arrivals:
            wait = waits.get(arrival['rank'])
            lower = max(0, wait['end_ns']-publish['end_ns'])/1000 if wait else 0
            upper = max(0, wait['end_ns']-publish['begin_ns'])/1000 if wait else 0
            output.append(dict(case=case, comm=comm, seq=seq, iter=arrival['iter'], slot=slot,
                               rank=arrival['rank'], observed_ranks=len(arrivals),
                               last_arrival_rank=publish['rank'], latest_arrival_sample_rank=last['rank'],
                               publisher_rank=publish['rank'],
                               arrival_spread_us=spread, resume_delay_lower_us=lower,
                               resume_delay_upper_us=upper,
                               wait_wall_us=wait.get('wall_us', '') if wait else 0,
                               wait_cpu_us=wait.get('cpu_us', '') if wait else 0,
                               involuntary=wait.get('involuntary', '') if wait else 0))
    return output


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, ranges, points, reports = [], [], [], []
    for root in args.inputs:
        paths = sorted(p for p in root.rglob('msprof_tx_*.json')
                       if p.parent.name in {'mindstudio_profiler_output', 'mindspeed_output'})
        for path in paths:
            events = list(iter_events(path))
            current, current_points, unmatched = extract_host_phases(events)
            current_ranges = extract_host_ranges(events, current)
            case = next(p for p in path.parents if (p/'run.log').exists())
            for r in current+current_points+current_ranges:
                r.update(case=str(case), source=str(path))
            rows.extend(current)
            ranges.extend(current_ranges)
            points.extend(current_points)
            reports.append(dict(source=str(path), phase_count=len(current), point_count=len(current_points),
                                range_count=len(current_ranges),
                                missing_cpu_metadata=len(current_ranges)-len(current),
                                unmatched_including_warmup=unmatched))
    barriers = barrier_observations(ranges)
    write_csv(args.output/'host_phase_cpu.csv', rows)
    write_csv(args.output/'host_points.csv', points)
    write_csv(args.output/'host_phase_ranges.csv', ranges)
    write_csv(args.output/'barrier_observations.csv', barriers)
    summary = {}
    for case in sorted({r['case'] for r in rows}):
        phases = {}
        for phase in sorted({r['phase'] for r in rows if r['case'] == case}):
            selected = [r for r in rows if r['case'] == case and r['phase'] == phase]
            phases[phase] = dict(count=len(selected), wall_median_us=statistics.median(r['wall_us'] for r in selected),
                                 wall_max_us=max(r['wall_us'] for r in selected),
                                 nonrunning_max_us=max(r['nonrunning_us'] for r in selected),
                                 involuntary_total=sum(r['involuntary'] for r in selected))
        summary[case] = phases
    (args.output/'host_phase_summary.json').write_text(json.dumps(dict(sources=reports, cases=summary), indent=2))
    print(json.dumps(dict(phase_rows=len(rows), points=len(points), barrier_rows=len(barriers))))


if __name__ == '__main__':
    main()
