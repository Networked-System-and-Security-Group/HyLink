#!/usr/bin/env python3
"""Join wait records by connection_id; never subtract Host/Device clocks.

Repeated records are not new wait submissions. Device interval unions are
profiling coverage, not proof of resource occupancy or contention.
"""
import argparse
from collections import defaultdict
import csv
from decimal import Decimal
import json
from pathlib import Path
from analyze_msprof import discover, iter_events, marker, union_us

KINDS = ('MEM_WAIT_VALUE', 'EVENT_WAIT', 'CAPTURE_WAIT')
APIS = ('aclrtValueWait', 'aclrtStreamWaitEvent')


def summarize_wait_records(apis, devices, ags):
    rows, links = {}, {}
    spans = defaultdict(list)
    connections = defaultdict(set)
    for start, end, fields, tid in ags:
        key = int(fields['rank']), int(fields['iter'])
        if key in rows:
            raise ValueError('duplicate rank/iteration')
        rows[key] = dict(rank=key[0], iter=key[1], interference=int(fields.get('interference', 0)),
                         host_value_wait_count=0, host_event_wait_count=0,
                         missing_device_connections=0)
        for kind in KINDS:
            rows[key].update({kind+'_records': 0, kind+'_connections': 0,
                              kind+'_sum_us': 0.0, kind+'_union_us': 0.0})
    for api in apis:
        t = Decimal(str(api['ts']))
        matches = [(int(f['rank']), int(f['iter'])) for a, b, f, tid in ags
                   if str(tid) == str(api['tid']) and Decimal(str(a)) <= t < Decimal(str(b))]
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError('ambiguous wait iteration')
        key = matches[0]
        cid = str(api['args']['connection_id'])
        if cid in links:
            raise ValueError('duplicate host wait connection')
        links[cid] = key
        field = 'host_value_wait_count' if api['name'].endswith('aclrtValueWait') else 'host_event_wait_count'
        rows[key][field] += 1
    seen = set()
    for event in devices:
        kind = event['name']
        cid = str(event['args'].get('connection_id'))
        if kind not in KINDS or cid not in links:
            continue
        key = links[cid]
        seen.add(cid)
        row = rows[key]
        row[kind+'_records'] += 1
        row[kind+'_sum_us'] += float(event['dur'])
        connections[key, kind].add(cid)
        t = Decimal(str(event['ts']))
        spans[key, kind].append((t, t+Decimal(str(event['dur']))))
    for cid, key in links.items():
        rows[key]['missing_device_connections'] += int(cid not in seen)
    for (key, kind), intervals in spans.items():
        origin = min(a for a, b in intervals)
        rows[key][kind+'_union_us'] = union_us((float(a-origin), float(b-origin)) for a, b in intervals)
        rows[key][kind+'_connections'] = len(connections[key, kind])
    return [rows[k] for k in sorted(rows)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    all_rows, reports = [], []
    for path in discover(args.inputs):
        ags = []
        for e in iter_events(next(path.parent.glob('msprof_tx_*.json'))):
            name, fields = marker(e)
            if e.get('ph') == 'X' and name == 'AG' and fields.get('phase') == 'formal':
                start = Decimal(str(e['ts']))
                ags.append((start, start+Decimal(str(e['dur'])), fields, e['tid']))
        hardware, apis, devices = set(), [], []
        for e in iter_events(path):
            name = e.get('name', '')
            if e.get('ph') == 'M' and name == 'process_name' and 'Ascend Hardware' in e.get('args', {}).get('name', ''):
                hardware.add(e['pid'])
            if e.get('ph') != 'X' or 'connection_id' not in e.get('args', {}):
                continue
            if name.startswith('AscendCL@') and name.split('@')[-1] in APIS or name in KINDS:
                small = {k: e.get(k) for k in ('name', 'ts', 'dur', 'pid', 'tid')}
                small['args'] = {'connection_id': e['args']['connection_id']}
                (devices if name in KINDS else apis).append(small)
        rows = summarize_wait_records(apis, [e for e in devices if e['pid'] in hardware], ags)
        if len(rows) != 80:
            raise ValueError(f'{path}: expected 80 formal iterations, got {len(rows)}')
        case = next(p for p in path.parents if (p/'run.log').exists())
        for row in rows:
            row.update(case=str(case), source=str(path))
        all_rows.extend(rows)
        report = dict(source=str(path), records=len(rows),
                      totals={k: sum(r[k] for r in rows) for k in rows[0] if k.endswith(('_count', '_records', '_connections', '_sum_us'))})
        reports.append(report)
        print(json.dumps(report), flush=True)
    if not all_rows:
        raise ValueError('no exported timelines')
    with (args.output/'wait_iterations.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    (args.output/'wait_summary.json').write_text(json.dumps(dict(records=len(all_rows), reports=reports, caveat=__doc__), indent=2))


if __name__ == '__main__':
    main()
