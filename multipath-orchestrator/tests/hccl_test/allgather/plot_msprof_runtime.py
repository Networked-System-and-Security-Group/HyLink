#!/usr/bin/env python3
"""Plot Host-only API coverage and stage timelines from batch CSV outputs."""
import argparse
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--iters', nargs='+', type=int, required=True)
    parser.add_argument('--case', default='')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--stages', action='store_true')
    modes.add_argument('--host-cpu', action='store_true')
    parser.add_argument('--phase', default='loop')
    args = parser.parse_args()
    filename = 'host_phase_cpu.csv' if args.host_cpu else 'host_stages.csv' if args.stages else 'runtime_iterations.csv'
    rows = [r for r in csv.DictReader((args.data/filename).open())
            if int(r['iter']) in args.iters and args.case in r['case']]
    if args.host_cpu:
        rows = [r for r in rows if r['phase'] == args.phase]
    cases = {r['case'] for r in rows}
    if len(cases) != 1:
        raise ValueError('select exactly one case')
    image = Image.new('RGB', (1300, 170+len(args.iters)*240), 'white')
    draw = ImageDraw.Draw(image)
    draw.font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    draw.text((30, 15), 'Host clock only; API elapsed includes waiting/descheduling; no causal claim', fill='black')
    colors = dict(acquire='#e8a047', start_barrier='#cc6688', loop='#589ac6',
                  flush='#555555', finish_barrier_signal='#7bba65')
    if args.host_cpu:
        draw.text((30, 40), 'Blue = thread CPU; gray = rest of measured wall (blocking/yield/preemption; NOT DMA busy)', fill='black')
    elif args.stages:
        draw.text((30, 40), 'acquire: orange | start barrier: pink | loop: blue | finish: green', fill='black')
    else:
        draw.text((30, 40), 'Coverage totals (not chronology): blue = API union; gray = rest of loop (NOT CPU idle)', fill='black')
    for panel, iteration in enumerate(args.iters):
        selected = [r for r in rows if int(r['iter']) == iteration]
        top = 90+panel*240
        origin = min(float(r['start_us']) for r in selected) if args.stages else 0
        limit = (max(float(r['wall_us']) for r in rows) if args.host_cpu else
                 max(float(r['end_us'])-origin for r in selected) if args.stages else
                 max(float(r['loop_us']) for r in rows))
        scale = 1050/limit
        draw.text((30, top), f'Iteration {iteration}', fill='black')
        for rank in range(4):
            y = top+35+rank*38
            draw.text((30, y+4), f'rank {rank}', fill='black')
            rank_rows = [r for r in selected if int(r['rank']) == rank]
            for row in rank_rows:
                if args.host_cpu:
                    wall, cpu = float(row['wall_us']), float(row['cpu_us'])
                    end = 120+wall*scale
                    draw.rectangle((120, y, end, y+22), fill='#dddddd')
                    draw.rectangle((120, y, 120+min(wall, max(0, cpu))*scale, y+22), fill='#589ac6')
                    draw.text((120, y+25), f"{args.phase}: CPU {cpu/1000:.2f} / wall {wall/1000:.2f} ms; v/i switches {row['voluntary']}/{row['involuntary']}", fill='black')
                elif args.stages:
                    if row['stage'] == 'batch':
                        continue
                    x0 = 120+(float(row['start_us'])-origin)*scale
                    x1 = 120+(float(row['end_us'])-origin)*scale
                    draw.rectangle((x0, y, max(x0+1, x1), y+22), fill=colors[row['stage']])
                else:
                    end = 120+float(row['loop_us'])*scale
                    union = float(row['api_union_us'])
                    draw.rectangle((120, y, end, y+22), fill='#dddddd')
                    draw.rectangle((120, y, 120+union*scale, y+22), fill='#589ac6')
                    draw.text((120, y+25), f"API {union/1000:.2f} / loop {float(row['loop_us'])/1000:.2f} ms; queries {row['aclrtQueryEventStatus_count']}", fill='black')
        for tick in range(6):
            x = 120+tick*210
            draw.text((x, top+202), f'{limit*tick/5/1000:.1f} ms', fill='black')
    output = args.data/('host_phase_cpu_coverage.png' if args.host_cpu else
                       'host_stage_timeline.png' if args.stages else 'runtime_api_coverage.png')
    image.save(output)
    print(output)


if __name__ == '__main__':
    main()
