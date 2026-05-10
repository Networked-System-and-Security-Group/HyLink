#!/usr/bin/env python3
"""Choose the most frequently observed TCP split ratio from a no-PCIe run."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

log = Path(sys.argv[1])
out = Path(sys.argv[2])
current = None
ratios = []
formal = False
for line in log.read_text(errors="replace").splitlines():
    if line.startswith("[rank0] begin formal"):
        formal = True
    match = re.search(r"split_pcie=([0-9.]+)", line)
    if match:
        current = f"{float(match.group(1)):.4f}"
    match = re.match(r"iter=(\d+) ", line)
    if formal and match and current is not None and int(match.group(1)) < 1001:
        ratios.append(current)
if len(ratios) != 1001:
    raise SystemExit(f"expected 1001 formal TCP ratios, found {len(ratios)} in {log}")
counts = Counter(ratios)
max_count = max(counts.values())
modes = sorted(alpha for alpha, count in counts.items() if count == max_count)
# A tied mode is resolved by the median of the tied proportions, independent
# of throughput, to avoid choosing a ratio by outcome.
selected = modes[len(modes) // 2]
result = {
    "selected_alpha": selected,
    "criterion": "most_frequent_formal_TCP_ratio_no_interference; median tied mode; no throughput maximization",
    "formal_iterations": len(ratios),
    "counts": dict(sorted(counts.items(), key=lambda item: float(item[0]))),
    "modal_count": max_count,
    "modal_fraction_percent": 100 * max_count / len(ratios),
}
out.write_text(json.dumps(result, indent=2) + "\n")
print(selected)
