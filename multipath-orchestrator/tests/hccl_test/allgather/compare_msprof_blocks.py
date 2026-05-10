#!/usr/bin/env python3
"""Compare block-size/profiler ablations; raw bandwidth, no smoothing."""
import argparse
import csv
import json
from pathlib import Path
import re
from analyze_msprof import parse_benchmark_log, phase_statistics


def read_case(log):
    log = Path(log)
    metadata = json.loads((log.parent/'metadata.json').read_text())
    size = int(next(a.split('=',1)[1] for a in metadata['command']
                    if a.startswith('--interference-size=')))
    profiled = int(metadata['environment']['CCL_MSPROF_MARKERS'])
    rows, ranks = parse_benchmark_log(log)
    info = dict(source=str(log), interference_bytes=size, profiled=profiled,
                records=len(rows), phases=phase_statistics(rows),
                background=[dict(rank=int(rank), copies=int(copies), active_ms=float(active),
                                 interval_GBps=float(bw)) for rank,copies,active,bw in re.findall(
                    r'interference_summary device=(\d+) copies=(\d+) copy_active_ms=([\d.]+) copy_interval_GBps=([\d.]+)',
                    log.read_text())])
    for row in rows:
        row.update(source=str(log), interference_bytes=size, profiled=profiled)
    return info, rows


def plot(rows, output):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new('RGB',(1500,850),'white'); draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    except OSError:
        font = ImageFont.load_default()
    for panel,size in enumerate(sorted({r['interference_bytes'] for r in rows},reverse=True)):
        top,bottom = 65+panel*395,370+panel*395
        selected = [r for r in rows if r['interference_bytes']==size]
        last = max(r['iter'] for r in selected) or 1
        maximum = max(r['bw_GBps'] for r in selected)*1.05
        for iteration in {r['iter'] for r in selected if r['interference']}:
            x = 85+iteration*1350/last
            draw.rectangle((x,top,min(1435,x+1350/last),bottom),fill='#fff2cc')
        draw.text((85,top-35),f'static alpha=.125, background {size//1048576}MiB: raw bandwidth (GB/s)',font=font,fill='black')
        for tick in range(6):
            y = bottom-tick*(bottom-top)/5
            draw.line((85,y,1435,y),fill='#dddddd')
            draw.text((5,y-12),f'{maximum*tick/5:.0f}',font=font,fill='black')
        for profiled,color,label in ((0,'#1565c0','no profiler'),(1,'#d32f2f','msprof')):
            values = sorted((r for r in selected if r['profiled']==profiled),key=lambda r:r['iter'])
            if len({r['iter'] for r in values}) != len(values):
                raise ValueError('multiple repetitions: plot separately, never mix identical iter indices')
            points = [(85+r['iter']*1350/last,bottom-r['bw_GBps']*(bottom-top)/maximum) for r in values]
            if len(points)>1: draw.line(points,fill=color,width=3)
            draw.text((1070+profiled*200,top-35),label,font=font,fill=color)
        for tick in range(0,last+1,10):
            draw.text((85+tick*1350/last,bottom+5),str(tick),font=font,fill='black')
    im.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    reports,rows = [],[]
    for log in args.logs:
        report,current = read_case(log); reports.append(report); rows.extend(current)
    (args.output/'block_profiler_comparison.json').write_text(json.dumps(reports,indent=2))
    with (args.output/'block_profiler_iterations.csv').open('w',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plot(rows,args.output/'block_profiler_bandwidth.png')
    print(json.dumps(dict(cases=len(reports),iterations=len(rows),output=str(args.output))))


if __name__ == '__main__':
    main()
