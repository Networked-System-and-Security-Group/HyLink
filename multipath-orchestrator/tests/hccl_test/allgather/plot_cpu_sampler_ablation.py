#!/usr/bin/env python3
"""Plot raw bandwidth pairs and CPU ratios, not automatic causal verdicts."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
from PIL import Image, ImageDraw, ImageFont
from analyze_msprof import parse_benchmark_log


def classify_case(report, comparison):
    name = Path(report['case']).parent.name
    match = re.search(r'rep(\d+)', name)
    if not match:
        raise ValueError('Missing repetition identity: ' + name)
    if comparison in ('completion', 'slice', 'wait', 'host'):
        pattern = (r'completion_(old|fix)_rep\d+_\d+' if comparison == 'completion'
                   else r'host_timing_(off|on)_rep\d+_\d+' if comparison == 'host'
                   else r'completion_wait_(value|event)_rep\d+_\d+' if comparison == 'wait'
                   else r'subchunk_(1m|4m)_rep\d+_\d+')
        identity = re.fullmatch(pattern, name)
        if not identity:
            raise ValueError('Missing comparison binary identity: ' + name)
        mode = identity[1]
    else:
        mode = 'on' if report['sample_period'] else 'off'
    return int(match[1]), mode


def read_series(report, metric, data):
    log = Path(report['case']) / 'run.log'
    if metric == 'bw_GBps':
        return parse_benchmark_log(log)[0]
    with (data / 'subchunk_iterations.csv').open() as file:
        return [{'iter': int(row['iter']), metric: float(row[metric])}
                for row in csv.DictReader(file) if row['source'] == str(log) and row[metric]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--comparison', choices=('cpu', 'completion', 'slice', 'wait', 'host'), default='cpu')
    parser.add_argument('--metric', choices=('bw_GBps', 'pcie_ms', 'hccl_ms'), default='bw_GBps')
    args = parser.parse_args()
    summary = json.loads((args.data/'cpu_sample_summary.json').read_text())
    pairs = {}
    for report in summary['reports']:
        if 'benchmark_phases' not in report:
            continue
        case = Path(report['case'])
        rep, mode = classify_case(report, args.comparison)
        pair = pairs.setdefault(rep, {})
        if mode in pair:
            raise ValueError('Duplicate repetition/series identity')
        pair[mode] = read_series(report, args.metric, args.data)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    image = Image.new('RGB', (1500, 100+300*len(pairs)), 'white')
    draw = ImageDraw.Draw(image)
    metric_label = {'bw_GBps': 'bandwidth GB/s', 'pcie_ms': 'PCIe path ms', 'hccl_ms': 'HCCL path ms'}[args.metric]
    draw.text((60, 15), f'4-card AllGather 256MiB/rank, static alpha .125, BG 16MiB; raw {metric_label}, no smoothing', fill='black', font=font)
    completion = args.comparison == 'completion'
    slice_comparison = args.comparison == 'slice'
    wait_comparison = args.comparison == 'wait'
    host_comparison = args.comparison == 'host'
    legend = ('Completion publication: orange = fix; blue = old. CPU sampling OFF, other msprof options ON.'
              if completion else 'CPU sampling: orange = every 16 API calls; blue = off. Other msprof options stay ON.')
    if slice_comparison or wait_comparison:
        modes = {Path(r['case']).name for r in summary['reports'] if 'benchmark_phases' in r}
        profiling = 'OFF' if modes == {'static_baseline'} else 'ON' if modes == {'static_msprof'} else 'MIXED'
        legend = (f'Completion wait: orange = EventWait; blue = ValueWait. Subchunk 4MiB, CPU sampling OFF, profiler {profiling}.'
                  if wait_comparison else
                  f'Subchunk: orange = 4MiB; blue = 1MiB. Completion fix ON, CPU sampling OFF, profiler {profiling}.')
    if host_comparison:
        legend = 'Host phase timing: orange = ON; blue = OFF. Same binary, profiler ON, API CPU sampling OFF.'
    draw.text((60, 40), legend, fill='black', font=font)
    maximum = max(r[args.metric] for pair in pairs.values() for rows in pair.values() for r in rows)*1.05
    for panel, (rep, pair) in enumerate(sorted(pairs.items())):
        top, bottom = 100+panel*300, 325+panel*300
        draw.text((70, top-25), f'Repeat {rep}', fill='black', font=font)
        for iteration in list(range(20,40))+list(range(55,65)):
            x = 80+iteration*1320/79
            draw.rectangle((x, top, min(1400, x+1320/79), bottom), fill='#fff2cc')
        for tick in range(5):
            y = bottom-tick*(bottom-top)/4
            draw.line((80,y,1400,y),fill='#dddddd')
            draw.text((10,y-8),f'{maximum*tick/4:.0f}',fill='black',font=font)
        series = [('value', '#1565c0'), ('event', '#e67e22')] if wait_comparison else [('1m', '#1565c0'), ('4m', '#e67e22')] if slice_comparison else [
            ('old' if completion else 'off','#1565c0'), ('fix' if completion else 'on','#e67e22')]
        for mode, color in series:
            if mode not in pair:
                continue
            points = [(80+r['iter']*1320/79,bottom-r[args.metric]*(bottom-top)/maximum) for r in pair[mode]]
            draw.line(points,fill=color,width=2)
        for tick in range(0,80,10):
            draw.text((80+tick*1320/79,bottom+8),str(tick),fill='black',font=font)
    filename = ('host_timing_bandwidth_pairs.png' if host_comparison else
                'completion_wait_bandwidth_pairs.png' if wait_comparison else
                'subchunk_bandwidth_pairs.png' if slice_comparison else
                'completion_bandwidth_pairs.png' if completion else 'cpu_sampler_bandwidth_pairs.png')
    if args.metric != 'bw_GBps':
        filename = f'{args.comparison}_{args.metric}_pairs.png'
    image.save(args.data/filename)

    if completion or slice_comparison or wait_comparison or host_comparison:
        return
    rows = [r for r in csv.DictReader((args.data/'cpu_samples.csv').open())
            if int(r['valid']) and float(r['wall_us']) >= 100]
    if rows:
        image = Image.new('RGB',(1200,750),'white')
        draw = ImageDraw.Draw(image)
        draw.text((40,15),'Long sparse API samples: CPU/wall ratio versus wall elapsed (log scale)',fill='black',font=font)
        draw.text((40,40),'red: involuntary switch; green: voluntary only; gray: no observed switch. Not a root-cause verdict.',fill='black',font=font)
        xmax = max(3,math.log10(max(float(r['wall_us']) for r in rows)))
        for tick in range(6):
            y = 650-tick*110
            draw.line((100,y,1100,y),fill='#dddddd')
            draw.text((20,y-8),f'{tick*.2:.1f}',fill='black',font=font)
        for row in rows:
            wall,cpu = float(row['wall_us']),float(row['cpu_us'])
            x = 100+(math.log10(wall)-2)*1000/(xmax-2)
            y = 650-min(1.1,max(0,cpu/wall))*550
            color = '#d32f2f' if int(row['involuntary']) else '#228833' if int(row['voluntary']) else '#777777'
            draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
        for tick in range(6):
            value = 2+(xmax-2)*tick/5
            draw.text((100+tick*200,670),f'{10**value:.0f}us',fill='black',font=font)
        image.save(args.data/'cpu_long_api_ratios.png')


if __name__ == '__main__':
    main()
