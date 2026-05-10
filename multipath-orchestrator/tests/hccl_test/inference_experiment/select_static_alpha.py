#!/usr/bin/env python3
"""Select a well-sampled PCIe fraction from non-interfered TCP iterations."""
import json
import re
import sys
from pathlib import Path
from statistics import mean, median
from process_results import parse_log, interfered

points = {i: bw for i, _, bw in parse_log(sys.argv[1])}
groups = {}
alpha = None
for line in Path(sys.argv[1]).read_text().splitlines():
    match = re.search(r'split_pcie=([\d.]+)', line)
    if match:
        alpha = match[1]
    match = re.match(r'iter=(\d+) ', line)
    if match and not interfered(int(match[1])):
        if alpha is None:
            raise ValueError('Missing split ratio before iteration')
        groups.setdefault(alpha, []).append(points[int(match[1])])
stats = {alpha: dict(count=len(values), mean_GiBps=mean(values), median_GiBps=median(values))
         for alpha, values in groups.items()}
eligible = {alpha: row for alpha, row in stats.items() if row['count'] >= 10}
if not eligible:
    raise ValueError('No ratio has at least 10 non-interfered samples')
selected = max(eligible, key=lambda alpha: eligible[alpha]['mean_GiBps'])
Path(sys.argv[2]).write_text(json.dumps(dict(selected_alpha=selected, criterion='highest mean over >=10 non-interfered samples', candidates=stats), indent=2))
print(selected)
