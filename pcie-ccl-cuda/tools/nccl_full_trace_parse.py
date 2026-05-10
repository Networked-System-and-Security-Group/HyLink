#!/usr/bin/env python3
"""Summarize JSONL emitted by tools/nccl_full_trace_intercept.c."""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path
from typing import Iterable


def fmt_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    number = float(value)
    for unit in units:
        if abs(number) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(number)} B"
            return f"{number:.2f} {unit}"
        number /= 1024.0
    return f"{value} B"


def read_rows(patterns: Iterable[str]) -> list[dict]:
    rows: list[dict] = []
    for pattern in patterns:
        for name in sorted(glob.glob(pattern)):
            path = Path(name)
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    row["_file"] = str(path)
                    rows.append(row)
    return rows


def print_counter(title: str, counter: collections.Counter) -> None:
    print(f"\n## {title}")
    for key, value in sorted(counter.items(), key=lambda item: str(item[0])):
        print(f"{value:6d}  {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="+", help="JSONL file(s) or glob pattern(s)")
    args = parser.parse_args()

    rows = read_rows(args.patterns)
    collectives = [row for row in rows if row.get("event") == "collective"]
    calls = [row for row in rows if row.get("event") == "call"]

    print(f"files: {len(set(row['_file'] for row in rows))}")
    print(f"rows: {len(rows)}")
    print(f"call events: {len(calls)}")
    print(f"collectives: {len(collectives)}")
    print_counter("Event counts", collections.Counter(row.get("event") for row in rows))
    print_counter("All API counts", collections.Counter(row.get("api") for row in rows))
    print_counter("Return codes by API", collections.Counter((row.get("api"), row.get("ret")) for row in rows))

    print_counter("Call API counts", collections.Counter(row.get("api") for row in calls))
    print_counter("Collective API counts", collections.Counter(row.get("api") for row in collectives))
    print_counter("Collective nranks", collections.Counter(row.get("nranks") for row in collectives))
    print_counter("Capture status", collections.Counter(row.get("capture") for row in collectives))
    print_counter("Group depth", collections.Counter(row.get("group_depth") for row in collectives))

    print("\n## Collective signatures")
    signatures = collections.Counter(
        (
            row.get("nranks"),
            row.get("api"),
            fmt_bytes(int(row.get("send_bytes", 0))),
            fmt_bytes(int(row.get("recv_bytes", 0))),
            row.get("datatype"),
            row.get("op"),
            "in-place" if row.get("inplace") else "out-of-place",
            f"group_depth={row.get('group_depth')}",
        )
        for row in collectives
    )
    for key, value in sorted(signatures.items(), key=lambda item: item[0]):
        print(f"{value:6d}  {key}")

    print("\n## Communicators")
    init_rows = [row for row in calls if row.get("api") in {"CommInitRank", "CommInitRankConfig"}]
    for key, value in sorted(collections.Counter(
        (
            row.get("api"),
            row.get("nranks"),
            row.get("rank"),
            row.get("out_rank"),
            row.get("out_nranks"),
            row.get("out_device"),
            row.get("ret"),
        )
        for row in init_rows
    ).items(), key=lambda item: item[0]):
        print(f"{value:6d}  {key}")

    print("\n## Per Process")
    for path in sorted(set(row["_file"] for row in rows)):
        local = [row for row in rows if row["_file"] == path]
        local_collectives = [row for row in local if row.get("event") == "collective"]
        devices = sorted(set(row.get("device") for row in local_collectives))
        print(
            f"{Path(path).name}: rows={len(local)} collectives={len(local_collectives)} "
            f"devices={devices} apis={collections.Counter(row.get('api') for row in local)}"
        )

    print("\n## Grouped Collectives")
    for row in collectives:
        if row.get("group_depth", 0) > 0:
            print(
                f"{Path(row['_file']).name}: group={row.get('group_id')} api={row.get('api')} "
                f"nranks={row.get('nranks')} send={fmt_bytes(int(row.get('send_bytes', 0)))} "
                f"recv={fmt_bytes(int(row.get('recv_bytes', 0)))} dtype={row.get('datatype')} "
                f"op={row.get('op')} {'in-place' if row.get('inplace') else 'out-of-place'} "
                f"stream={row.get('stream')}"
            )


if __name__ == "__main__":
    main()
