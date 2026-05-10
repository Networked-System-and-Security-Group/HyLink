#!/usr/bin/env python3
"""Batch msprof timelines without printing raw events. Times are microseconds.

Host ranges are not device execution times. DMA busy time is an interval union,
not a sum across streams; classification by size is specific to this experiment.
"""
import argparse
import bisect
import csv
import json
from pathlib import Path
import re
import statistics
from collections import defaultdict


def parse_benchmark_log(path):
    path = Path(path)
    text = path.read_text(errors='replace')
    metadata_path = path.parent/'metadata.json'
    ranges = []
    if metadata_path.exists():
        for arg in json.loads(metadata_path.read_text()).get('command', []):
            if arg.startswith('--interference-ranges='):
                ranges = [tuple(map(int, part.split('-'))) for part in arg.split('=', 1)[1].split(',') if part]
    ranks = [dict(case=path.parent.name, rank=int(rank), iter=int(iteration),
                  start_ms=float(start), end_ms=float(end), event_ms=float(event))
             for rank, iteration, start, end, event in re.findall(
                 r'^collective device=(\d+) iter=(\d+) start_ms=([\d.]+) end_ms=([\d.]+) event_ms=([\d.]+)', text, re.M)]
    rows = []
    for iteration, event, bw in re.findall(r'^iter=(\d+) event_ms=([\d.]+) bw_GBps=([\d.]+)', text, re.M):
        iteration = int(iteration)
        starts = [r['start_ms'] for r in ranks if r['iter'] == iteration]
        rows.append(dict(case=path.parent.name, iter=iteration, event_ms=float(event), bw_GBps=float(bw),
                         interference=int(any(a <= iteration <= b for a, b in ranges)),
                         start_skew_ms=round(max(starts)-min(starts), 9) if starts else ''))
    return rows, ranks


def phase_statistics(rows):
    result = {}
    for flag, label in [(0, 'normal'), (1, 'interference')]:
        selected = [r for r in rows if r['interference'] == flag]
        result[label] = {}
        for key in ('event_ms', 'bw_GBps'):
            values = sorted(r[key] for r in selected)
            if not values:
                continue
            mean = statistics.mean(values)
            result[label][key] = dict(count=len(values), mean=mean,
                cv_percent=100*statistics.pstdev(values)/mean if mean else 0,
                p95=values[min(len(values)-1, int(.95*len(values)))], maximum=max(values), minimum=min(values))
    return result


def discover(inputs):
    paths = set()
    for item in inputs:
        item = Path(item)
        candidates = list(item.rglob('msprof*.json')) if item.is_dir() else [item]
        exported = [p for p in candidates if p.parent.name in ('mindstudio_profiler_output', 'mindspeed_output')]
        if item.is_dir() and exported:
            candidates = exported
        paths.update(p for p in candidates if not p.name.startswith('msprof_tx_'))
    return sorted(paths)


def iter_events(path):
    """Incremental stdlib decoder for array and traceEvents-object exports."""
    decoder = json.JSONDecoder()
    with Path(path).open() as file:
        buffer = file.read(1024*1024)
        stripped = buffer.lstrip()
        if stripped.startswith('['):
            pos = len(buffer)-len(stripped)+1
        else:
            match = re.search(r'"traceEvents"\s*:\s*\[', buffer)
            if not match:
                raise ValueError(f'{path}: missing traceEvents array')
            pos = match.end()
        eof = False
        while True:
            while pos < len(buffer) and buffer[pos] in ' \t\r\n,':
                pos += 1
            if pos < len(buffer) and buffer[pos] == ']':
                return
            try:
                value, end = decoder.raw_decode(buffer, pos)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError(f'{path}: truncated or invalid JSON')
                more = file.read(1024*1024)
                buffer = buffer[pos:] + more
                pos = 0
                eof = not more
                continue
            yield value
            pos = end


def union_us(intervals):
    total = 0.0
    stop = None
    for start, end in sorted(intervals):
        if end <= start:
            continue
        total += end - max(start, stop if stop is not None else start) if stop is None or end > stop else 0
        stop = max(end, stop if stop is not None else end)
    return total


def marker(event):
    parts = event.get('name', '').split('|')
    return parts[0], dict(p.split('=', 1) for p in parts[1:] if '=' in p)


def overlap(event, start, end):
    a = max(float(event['ts']), start)
    b = min(float(event['ts']) + float(event.get('dur', 0)), end)
    return (a, b) if b > a else None


def correlate_msprof_tasks(events):
    """Join hardware to host APIs by connection_id, never by hardware time clipping.

    Cross-domain differences are exported as signed evidence, not proven queue
    latency. CANN uses distinct synthetic pids for msproftx and AscendCL layers;
    use actual OS thread IDs for API-to-loop, marker pid for loop-to-AG.
    """
    hardware = {}
    for e in events:
        if e.get('ph') == 'M' and e.get('name') == 'process_name':
            name = e.get('args', {}).get('name', '')
            if 'Ascend Hardware' in name:
                match = re.search(r'(?:device|Device)\s*[_:]?\s*(\d+)', name)
                hardware[e['pid']] = int(match[1]) if match else None
    apis, tasks = defaultdict(list), defaultdict(list)
    loops, ags = [], []
    for e in events:
        if e.get('ph') != 'X' or 'ts' not in e or 'dur' not in e:
            continue
        name, fields = marker(e)
        if name == 'PCCL.loop':
            loops.append((e, fields))
        elif name == 'AG' and fields.get('phase') == 'formal':
            ags.append((e, fields))
        cid = e.get('args', {}).get('connection_id')
        if cid is None:
            continue
        if name.split('@')[-1] == 'aclrtMemcpyAsync':
            apis[str(cid)].append(e)
        elif e.get('pid') in hardware and 'memcpy' in name.lower():
            tasks[str(cid)].append(e)
    indexes = []
    for ranges, thread_sensitive in ((loops, True), (ags, False)):
        grouped = defaultdict(list)
        for e, f in ranges:
            grouped[(None if thread_sensitive else e.get('pid'),
                     e.get('tid') if thread_sensitive else None)].append((e, f))
        index = {}
        for key, group in grouped.items():
            group.sort(key=lambda pair: float(pair[0]['ts']))
            index[key] = (group, [float(e['ts']) for e, f in group],
                          max(float(e['dur']) for e, f in group))
        indexes.append(index)
    rows, warnings = [], []
    for cid, candidates in apis.items():
        for api in candidates:
            start = float(api['ts'])
            groups = []
            for number, (index, tid) in enumerate(zip(indexes, (api.get('tid'), None))):
                marker_pid = groups[0][0][0].get('pid') if groups and len(groups[0]) == 1 else None
                group, starts, longest = index.get((None if number == 0 else marker_pid, tid), ([], [], 0))
                left, right = bisect.bisect_left(starts, start-longest), bisect.bisect_right(starts, start)
                groups.append([(e, f) for e, f in group[left:right]
                               if float(e['ts']) <= start < float(e['ts'])+float(e['dur'])])
            matches, ag_matches = groups
            if not matches:
                continue  # background thread or outside executor loop
            if not ag_matches:
                continue  # warmup
            if len(candidates) != 1 or len(matches) != 1 or len(ag_matches) != 1:
                warnings.append(f'connection {cid}: ambiguous API/work identity')
                continue
            loop, work = matches[0]
            ag, fields = ag_matches[0]
            # Some CANN exports name the sole hardware process just "Ascend
            # Hardware". Infer its card only for a single-device host export.
            only_card = len(hardware) == 1 and len({f['device'] for e, f in ags}) == 1
            selected = [t for t in tasks[cid] if hardware[t['pid']] == int(fields['device']) or
                        (hardware[t['pid']] is None and only_card)]
            if not selected:
                warnings.append(f'connection {cid}: missing hardware task')
                continue
            for task in selected:
                dev_start, duration = float(task['ts']), float(task['dur'])
                rows.append(dict(rank=int(fields['rank']), device=int(fields['device']),
                    iter=int(fields['iter']), interference=int(fields.get('interference', 0)),
                    comm=work.get('comm', ''), seq=work.get('seq', ''), connection_id=cid,
                    api_tid=api['tid'], stream=task['tid'], task_id=task['args'].get('Task Id', ''),
                    bytes=int(task['args'].get('size(B)', 0)), operation=task['args'].get('operation', ''),
                    api_start_us=start, api_us=float(api['dur']), device_start_us=dev_start,
                    device_execution_us=duration, loop_start_us=float(loop['ts']), loop_us=float(loop['dur']),
                    api_end_to_device_start_us=dev_start-start-float(api['dur']),
                    device_end_minus_loop_end_us=dev_start+duration-float(loop['ts'])-float(loop['dur'])))
    return rows, warnings


def analyze_file(path):
    path = Path(path)
    events = []
    event_count = 0
    for e in iter_events(path):
        event_count += 1
        name = e.get('name', '')
        if e.get('ph') == 'M' or name.startswith(('AG', 'PCCL.', 'AMP.', 'BG.')) or \
                any(word in name.lower() for word in ('memcopy', 'memcpy')):
            events.append(e)
    hardware = {}
    for e in events:
        if e.get('ph') == 'M' and e.get('name') == 'process_name':
            name = e.get('args', {}).get('name', '')
            if 'Ascend Hardware' in name:
                match = re.search(r'(?:device|Device)\s*[_:]?\s*(\d+)', name)
                hardware[e['pid']] = int(match.group(1)) if match else None
    ranges = []
    tasks = []
    copy_apis = []
    warnings = []
    for e in events:
        if e.get('ph') != 'X' or 'ts' not in e or 'dur' not in e:
            continue
        name, attrs = marker(e)
        if name.startswith(('AG', 'PCCL.', 'AMP.', 'BG.')):
            ranges.append((e, name, attrs))
        elif e.get('pid') in hardware and any(word in e.get('name', '').lower() for word in ('memcopy', 'memcpy')):
            tasks.append(e)
        elif e.get('name', '').split('@')[-1] == 'aclrtMemcpyAsync':
            copy_apis.append(e)
    bg_threads = {e.get('tid') for e, stage, fields in ranges if stage == 'BG.D2H'}
    bg_connections = {str(e['args']['connection_id']) for e in copy_apis
                      if e.get('tid') in bg_threads and 'connection_id' in e.get('args', {})}
    tasks.sort(key=lambda e: float(e['ts']))
    starts = [float(e['ts']) for e in tasks]
    max_duration = max((float(e['dur']) for e in tasks), default=0)
    iterations = []
    device_tasks = []
    for ag, name, attrs in ranges:
        if name != 'AG' or attrs.get('phase') != 'formal':
            continue
        rank, device, iteration = int(attrs['rank']), int(attrs['device']), int(attrs['iter'])
        start, end = float(ag['ts']), float(ag['ts']) + float(ag['dur'])
        row = dict(source=str(path), rank=rank, device=device, iter=iteration,
                   interference=int(attrs.get('interference', 0)), start_us=start, ag_host_us=end-start)
        for stage in ('acquire', 'start_barrier', 'loop', 'flush', 'finish_barrier_signal', 'batch'):
            row['pccl_' + stage + '_us'] = 0.0
        row.update(pccl_seq='', pccl_comm='', alpha='', fast_bytes='', pcie_bytes='')
        row.update(ag_submit_us=0.0, ag_wait_us=0.0, ag_prepare_us=0.0, ag_verify_us=0.0,
                   amp_fast_submit_us=0.0, amp_pcie_submit_us=0.0,
                   pccl_batch_start_delay_us='', pccl_batch_end_offset_us='')
        background_threads = {e.get('tid') for e, stage, fields in ranges
                              if stage == 'BG.D2H' and int(fields.get('rank', -1)) == rank}
        row.update(foreground_copy_api_us=0.0, foreground_copy_api_max_us=0.0,
                   background_copy_api_max_us=0.0)
        for api in copy_apis:
            clipped = overlap(api, start, end)
            if not clipped:
                continue
            if api.get('tid') in background_threads:
                row['background_copy_api_max_us'] = max(row['background_copy_api_max_us'], float(api['dur']))
            else:
                row['foreground_copy_api_us'] += clipped[1] - clipped[0]
                row['foreground_copy_api_max_us'] = max(row['foreground_copy_api_max_us'], float(api['dur']))
        matched_batches = []
        for event, stage, fields in ranges:
            if int(fields.get('rank', -1)) != rank or event.get('pid') != ag.get('pid'):
                continue
            if stage in ('AG.prepare', 'AG.verify') and int(fields.get('iter', -1)) == iteration:
                row['ag_'+stage[3:]+'_us'] += float(event['dur'])
            clipped = overlap(event, start, end)
            if not clipped:
                continue
            if stage.startswith('PCCL.'):
                key = 'pccl_' + stage[5:] + '_us'
                if key in row:
                    row[key] += clipped[1] - clipped[0]
                if stage == 'PCCL.batch':
                    matched_batches.append(fields)
                    row['pccl_batch_start_delay_us'] = float(event['ts'])-start
                    row['pccl_batch_end_offset_us'] = float(event['ts'])+float(event['dur'])-start
            elif stage == 'AMP.plan_submit':
                for key in ('alpha', 'fast_bytes', 'pcie_bytes'):
                    row[key] = fields.get(key, '')
            elif stage in ('AG.submit', 'AG.wait', 'AMP.fast_submit', 'AMP.pcie_submit'):
                key = stage.lower().replace('.', '_')+'_us'
                row[key] += clipped[1]-clipped[0]
        if len(matched_batches) == 1:
            row['pccl_seq'] = matched_batches[0].get('seq', '')
            row['pccl_comm'] = matched_batches[0].get('comm', '')
        elif len(matched_batches) != 1:
            warnings.append(f'rank {rank} iter {iteration}: {len(matched_batches)} overlapping PCCL batches; do not assume seq=iter+warmup')
        intervals = {'dma': [], 'foreground_dma': [], 'background_dma': []}
        counts = {'foreground_dma': 0, 'background_dma': 0}
        maxima = {'foreground_dma': 0.0, 'background_dma': 0.0}
        left, right = bisect.bisect_left(starts, start-max_duration), bisect.bisect_left(starts, end)
        for task in tasks[left:right]:
            task_device = hardware[task['pid']]
            if task_device is not None and task_device != device:
                continue
            if task_device is None and len(hardware) != 1:
                continue  # ambiguous card, never silently combine all devices
            clipped = overlap(task, start, end)
            if not clipped:
                continue
            args = task.get('args', {})
            size = args.get('size(B)', args.get('Size', args.get('size', 0)))
            try:
                size = int(size)
            except (TypeError, ValueError):
                size = 0
            intervals['dma'].append(clipped)
            # CANN 9.0 exports one 128MiB D2H as two 64MiB hardware tasks.
            background = str(args.get('connection_id', '')) in bg_connections or (
                size in (67108864,134217728) and args.get('operation') == 'device to host')
            kind = 'background_dma' if background else 'foreground_dma' if 0 < size <= 1048576 else None
            if kind:
                counts[kind] += 1
                intervals[kind].append(clipped)
                maxima[kind] = max(maxima[kind], float(task['dur']))
            device_tasks.append(dict(source=str(path), rank=rank, device=device, iter=iteration,
                kind=kind or 'unclassified', start_us=float(task['ts']), duration_us=float(task['dur']),
                overlap_us=clipped[1]-clipped[0], stream=task.get('tid'),
                task_id=args.get('Task Id', ''), connection_id=args.get('connection_id', ''),
                bytes=size, operation=args.get('operation', '')))
        for kind, spans in intervals.items():
            row[kind + '_busy_us'] = union_us(spans)
        for kind in counts:
            row[kind + '_count'] = counts[kind]
            row[kind + '_max_task_us'] = maxima[kind]
        iterations.append(row)
    if not iterations:
        warnings.append('no formal AG markers; enable CCL_MSPROF_MARKERS=1 and --msproftx=on')
    if not tasks:
        warnings.append('no hardware Memcopy tasks; device-side DMA metrics are unavailable')
    correlated, correlation_warnings = correlate_msprof_tasks(events)
    connected_by_work = defaultdict(list)
    for task in correlated:
        connected_by_work[(task['rank'], task['iter'])].append(task)
    for row in iterations:
        selected = connected_by_work[(row['rank'], row['iter'])]
        row['foreground_connected_count'] = len(selected)
        row['foreground_device_span_us'] = (max(t['device_start_us']+t['device_execution_us'] for t in selected)
                                            -min(t['device_start_us'] for t in selected)) if selected else ''
        row['foreground_connected_api_max_us'] = max((t['api_us'] for t in selected), default='')
        row['foreground_connected_device_max_us'] = max((t['device_execution_us'] for t in selected), default='')
        row['device_end_minus_loop_end_max_us'] = max((t['device_end_minus_loop_end_us'] for t in selected), default='')
    return dict(source=str(path), events=event_count, hardware_processes=hardware,
                dma_tasks=len(tasks), iterations=iterations, device_tasks=device_tasks,
                correlated_tasks=correlated, correlation_warnings=correlation_warnings,
                warnings=warnings)


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, output):
    from PIL import Image, ImageDraw
    width, height = 1400, 800
    im = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(im)
    colors = ['#1565c0', '#d32f2f', '#388e3c', '#7b1fa2']
    for panel, key in enumerate(('ag_host_us', 'pccl_loop_us')):
        top, bottom = 45 + panel*390, 355 + panel*390
        maximum = max((float(r[key]) for r in rows), default=1) or 1
        max_iter = max((r['iter'] for r in rows), default=1) or 1
        draw.text((70, top-25), key + ' (ms), raw / no smoothing', fill='black')
        for tick in range(6):
            y = bottom - tick*(bottom-top)/5
            draw.line((70, y, 1350, y), fill='#dddddd')
            draw.text((5, y), f'{maximum*tick/5000:.2f}', fill='black')
        for rank in sorted({r['rank'] for r in rows}):
            selected = sorted((r for r in rows if r['rank'] == rank), key=lambda r:r['iter'])
            points = [(70+r['iter']*1280/max_iter, bottom-float(r[key])*(bottom-top)/maximum) for r in selected]
            if len(points) >= 2:
                draw.line(points, fill=colors[rank%4], width=2)
            draw.text((1000+rank*85, top-25), f'rank {rank}', fill=colors[rank%4])
        draw.text((650, bottom+10), 'iteration', fill='black')
    im.save(output)


def plot_bandwidth(rows, output):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new('RGB', (1500, 850), 'white')
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    except OSError:
        font = ImageFont.load_default()
    algorithms = sorted({r['case'].rsplit('_', 1)[0] for r in rows})
    for panel, algo in enumerate(algorithms[:2]):
        top, bottom = 60+panel*395, 370+panel*395
        values = [r for r in rows if r['case'].rsplit('_', 1)[0] == algo]
        max_iter = max(r['iter'] for r in values) or 1
        maximum = max(r['bw_GBps'] for r in values)*1.05
        background = {r['iter'] for r in values if r['interference']}
        for iteration in background:
            x = 85+iteration*1350/max_iter
            draw.rectangle((x, top, min(1435, x+1350/max_iter), bottom), fill='#fff2cc')
        for tick in range(6):
            y = bottom-tick*(bottom-top)/5
            draw.line((85, y, 1435, y), fill='#dddddd')
            draw.text((10, y-12), f'{maximum*tick/5:.0f}', font=font, fill='black')
        for suffix, color in [('baseline', '#1565c0'), ('msprof', '#d32f2f')]:
            selected = sorted((r for r in values if r['case'].endswith('_'+suffix)), key=lambda r:r['iter'])
            points = [(85+r['iter']*1350/max_iter, bottom-r['bw_GBps']*(bottom-top)/maximum) for r in selected]
            if len(points) >= 2:
                draw.line(points, fill=color, width=3)
            draw.text((950+(['baseline','msprof'].index(suffix))*230, top-35), suffix, font=font, fill=color)
        draw.text((85, top-35), algo+' bandwidth (GB/s), raw', font=font, fill='black')
        for tick in range(0, max_iter+1, 10):
            draw.text((85+tick*1350/max_iter, bottom+5), str(tick), font=font, fill='black')
    im.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--expected-iters', type=int)
    parser.add_argument('--expected-ranks', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    rows = []
    for path in discover(args.inputs):
        report = analyze_file(path)
        current = report.pop('iterations')
        device_rows = report.pop('device_tasks')
        connected = report.pop('correlated_tasks')
        if connected:
            write_csv(args.output/(path.parent.parent.name+'_connected_tasks.csv'), connected)
        report['connected_tasks'] = len(connected)
        if device_rows:
            # Profile-directory names include the process identity; avoid rank filename collisions.
            identity = path.parent.parent.name
            write_csv(args.output/(identity+'_device_tasks.csv'), device_rows)
        rows.extend(current)
        reports.append(report)
    # Keep case boundaries: identical rank/iter from different runs are NOT duplicates.
    groups = {}
    for row in rows:
        source = Path(row['source'])
        case = next((p for p in source.parents if (p/'run.log').exists()), source.parent)
        row['case'] = str(case)
        groups.setdefault(str(case), []).append(row)
    summaries = []
    for case, values in groups.items():
        keys = [(r['rank'], r['iter']) for r in values]
        duplicates = len(keys) - len(set(keys))
        expected = args.expected_iters * args.expected_ranks if args.expected_iters else None
        warnings = []
        if duplicates:
            warnings.append(f'{duplicates} duplicate rank/iter records: possible overlapping export files')
        if expected is not None and len(set(keys)) != expected:
            warnings.append(f'expected {expected} rank/iter records, found {len(set(keys))}')
        summary = dict(case=case, records=len(values), warnings=warnings, phases={})
        for flag, label in [(0, 'normal'), (1, 'interference')]:
            phase = [r for r in values if r['interference'] == flag]
            summary['phases'][label] = {key: dict(mean=statistics.mean(float(r[key]) for r in phase),
                maximum=max(float(r[key]) for r in phase)) for key in
                ('ag_host_us', 'pccl_loop_us', 'foreground_dma_max_task_us', 'background_dma_max_task_us')
                } if phase else {}
        summary['slowest'] = sorted(values, key=lambda r:r['ag_host_us'], reverse=True)[:8]
        summaries.append(summary)
        filename = Path(case).name
        write_csv(args.output/(filename+'_iterations.csv'), values)
        if not duplicates:
            plot(values, args.output/(filename+'_timings.png'))
    write_csv(args.output/'iterations.csv', rows)
    logs = set()
    for item in args.inputs:
        if item.is_dir():
            logs.update(item.rglob('run.log'))
    log_rows, rank_rows, log_reports = [], [], []
    for path in sorted(logs):
        current, ranks = parse_benchmark_log(path)
        if current:
            log_rows.extend(current)
            rank_rows.extend(ranks)
            log_reports.append(dict(case=path.parent.name, phases=phase_statistics(current),
                max_start_skew_ms=max((r['start_skew_ms'] for r in current if r['start_skew_ms'] != ''), default=None)))
    write_csv(args.output/'bandwidth_iterations.csv', log_rows)
    write_csv(args.output/'benchmark_rank_iterations.csv', rank_rows)
    if log_rows:
        plot_bandwidth(log_rows, args.output/'baseline_vs_msprof_bandwidth.png')
    (args.output/'summary.json').write_text(json.dumps(dict(files=reports, cases=summaries,
                                                         benchmark_cases=log_reports), indent=2))
    print(json.dumps(dict(files=len(reports), records=len(rows), cases=len(groups),
        warnings=sum(len(r['warnings']) for r in reports)+sum(len(s['warnings']) for s in summaries),
        output=str(args.output))))
    if not rows or any(s['warnings'] for s in summaries):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
