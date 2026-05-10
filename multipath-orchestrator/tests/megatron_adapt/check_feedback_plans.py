#!/usr/bin/env python3
"""Check that all ranks derive one AMP-CCL plan for each collective key."""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path


PLAN_RE = re.compile(
    r"\[Rank (?P<rank>\d+)\] Before (?P<op>AllGather|ReduceScatter),"
    r"(?: op=(?P<op_id>\d+) bytes=(?P<bytes>\d+) bucket_bytes=(?P<bucket>\d+)"
    r" dtype=(?P<dtype>-?\d+) nranks=(?P<nranks>\d+) cache_hit=(?P<hit>\d+))?"
    r"\s*alpha=(?P<alpha>[0-9.]+), plan\.use_pcie=(?P<use>\d+),"
    r" plan\.pcie_bytes=(?P<pcie>\d+), plan\.fast_bytes=(?P<fast>\d+),"
    r" domain_hash=(?P<domain>[0-9a-fA-F]+)"
)


def check(path: Path, require_independent_initial_alpha: bool = False) -> int:
    groups: dict[tuple[object, ...], list[tuple[int, tuple[object, ...]]]] = collections.defaultdict(list)
    first_alpha: dict[tuple[object, ...], dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    occurrences: dict[tuple[int, tuple[object, ...]], int] = collections.defaultdict(int)
    for match in PLAN_RE.finditer(path.read_text(errors="replace")):
        data = match.groupdict()
        total_bytes = int(data["bytes"] or (int(data["fast"]) + int(data["pcie"])))
        key = (
            data["op"],
            total_bytes,
            int(data["bucket"] or 0),
            int(data["dtype"] or -1),
            int(data["nranks"] or -1),
            data["domain"],
        )
        plan = (
            data["alpha"],
            int(data["use"]),
            int(data["fast"]),
            int(data["pcie"]),
        )
        rank = int(data["rank"])
        occurrence_key = (rank, key)
        occurrence = occurrences[occurrence_key]
        occurrences[occurrence_key] += 1
        groups[(key, occurrence)].append((rank, plan))
        first_alpha[key[1:]][data["op"]].append(float(data["alpha"]))

    if not groups:
        print(f"no AMP-CCL plan lines found in {path}", file=sys.stderr)
        return 2

    failed = False
    for key, entries in sorted(groups.items(), key=lambda item: str(item[0])):
        plans = collections.defaultdict(list)
        for rank, plan in entries:
            plans[plan].append(rank)
        if len(plans) > 1:
            failed = True
            print(f"plan mismatch key={key}")
            for plan, ranks in plans.items():
                print(f"  ranks={sorted(ranks)} plan={plan}")

    if require_independent_initial_alpha:
        for key, op_alphas in sorted(first_alpha.items(), key=lambda item: str(item[0])):
            for op, alphas in sorted(op_alphas.items()):
                if not alphas:
                    continue
                initial = alphas[0]
                if abs(initial - 0.01) > 1e-6:
                    failed = True
                    print(
                        f"initial alpha is not independent default key={key} op={op} "
                        f"alpha={initial:.6f}"
                    )

    if failed:
        return 1
    print(f"feedback plan consistency: PASS ({len(groups)} collective keys)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(
            f"usage: {sys.argv[0]} LOG [--require-independent-initial-alpha]",
            file=sys.stderr,
        )
        raise SystemExit(2)
    require_independent = len(sys.argv) == 3
    if require_independent and sys.argv[2] != "--require-independent-initial-alpha":
        print(f"unknown option: {sys.argv[2]}", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(check(Path(sys.argv[1]), require_independent))
