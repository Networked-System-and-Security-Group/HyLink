#!/usr/bin/env python3
"""Summarize paired TCP/static sweep runs without selecting per-algorithm runs."""
import csv
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for pair in sorted((root / "pairs").iterdir()):
    if not pair.is_dir() or not (pair / "summary.json").exists():
        continue
    stats = json.loads((pair / "summary.json").read_text())
    tcp, static = stats["tcp"], stats["static"]
    base_tcp = tcp["no_interference"]["mean_GiBps"]
    base_static = static["no_interference"]["mean_GiBps"]
    interfered_tcp = tcp["interference"]
    interfered_static = static["interference"]
    base_gap = 100 * abs(base_tcp - base_static) / ((base_tcp + base_static) / 2)
    gain = 100 * (interfered_tcp["mean_GiBps"] / interfered_static["mean_GiBps"] - 1)
    stable = interfered_tcp["cv_percent"] < 5 and interfered_static["cv_percent"] < 5
    close = base_gap < 5
    improved = gain >= 5
    rows.append({
        "condition": pair.name,
        "tcp_no_interference_GiBps": base_tcp,
        "static_no_interference_GiBps": base_static,
        "baseline_gap_percent": base_gap,
        "tcp_interference_GiBps": interfered_tcp["mean_GiBps"],
        "static_interference_GiBps": interfered_static["mean_GiBps"],
        "tcp_interference_cv_percent": interfered_tcp["cv_percent"],
        "static_interference_cv_percent": interfered_static["cv_percent"],
        "tcp_gain_over_static_percent": gain,
        "screen_pass": stable and close and improved,
        "repeat_required": stable and close and improved,
    })

fields = list(rows[0]) if rows else ["condition"]
with (root / "paired_summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
(root / "paired_summary.json").write_text(json.dumps({
    "screen_rule": {
        "both_interfered_cv_percent_lt": 5,
        "baseline_relative_mean_gap_percent_lt": 5,
        "tcp_interference_gain_over_static_percent_gte": 5,
        "note": "One paired screen is not a reproducibility claim; every passing condition requires repeated adjacent pairs.",
    },
    "conditions": rows,
}, indent=2) + "\n")
print(f"summarized {len(rows)} paired conditions; passing={sum(row['screen_pass'] for row in rows)}")
