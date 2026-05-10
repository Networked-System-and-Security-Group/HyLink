#!/usr/bin/env python3
"""Process NCCL compare_multi_case logs into raw and summary CSV files."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

COLLECTIVES = ("allgather", "allreduce", "alltoall", "reducescatter")
CASE_RE = re.compile(
    r"---\s+mode=(?P<mode>\w+)\s+collective=(?P<collective>\w+)\s+cards=(?P<cards>\d+)\s+"
    r"size_bytes=(?P<size>\d+)\s+elems_arg=(?P<elems>\d+)\s+iters=(?P<iters>\d+)\s+---"
)
HEADER_RE = re.compile(
    r"===\s+mode=(?P<mode>\w+)\s+backend=NCCL\s+collective=(?P<collective>\w+)\s+"
    r"cards=(?P<cards>\d+)\s+timestamp=(?P<timestamp>[^\s=]+)\s+==="
)
BASELINE_AVG_RE = re.compile(r"Average bandwidth over\s+(?P<iters>\d+)\s+iters \(event\):\s+(?P<bw>[0-9.]+)\s+GB/s")
BASELINE_ITER_RE = re.compile(r"normal test on iter\s+\d+")
# Multi-process (mpi_primitives) result line, printed by rank 0 once per case:
#   result: PASSED timer=cuda_event_collective op=allgather nranks=4 count=... op_bytes=...
#     payload_bytes=... avg_time=... s ... algbw=X GB/s busbw=Y GB/s ...
MPI_RESULT_RE = re.compile(
    r"result:\s+(?P<status>PASSED|FAILED)\s+timer=\S+\s+op=(?P<op>\w+)\s+nranks=(?P<nranks>\d+)\s+"
    r"count=\d+\s+op_bytes=(?P<op_bytes>\d+)\s+payload_bytes=(?P<payload_bytes>\d+)\s+"
    r".*?algbw=(?P<algbw>[0-9.eE+-]+)\s+GB/s\s+busbw=(?P<busbw>[0-9.eE+-]+)\s+GB/s"
)
# Single-process baseline tests compute GB/s with 1024^3 (GiB/s), while AMPCCL
# aggregate lines and the MPI binary use decimal 1e9. Normalize baseline samples
# to decimal GB/s so the modes are directly comparable.
GIB_TO_GB = 1024.0 ** 3 / 1.0e9
TOTAL_BW_RE = re.compile(r"\[AMPCCL\]\[OFF\]\s+Total bw is\s+(?P<bw>[0-9.]+)\s+GB/s")
AGG_ALGBW_RE = re.compile(r"\[AMPCCL\]\[OFF\]\s+aggregate\s+op=.*?op_key_bytes=(?P<size>\d+).*?n=(?P<cards>\d+).*?algbw\(GB/s,decimal\)=(?P<algbw>[0-9.]+)")
AGG_RE = re.compile(r"\[AMPCCL\]\[OFF\]\s+aggregate\s+op=.*?op_key_bytes=(?P<size>\d+).*?n=(?P<cards>\d+)")
WARMUP_DONE_RE = re.compile(r"Warmup done")
FILE_RE = re.compile(r"(?P<cards>\d+)gpu_(?P<mode>baseline|adaptive|multipath)_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{4})\.log$")


def normalize_mode(mode: str) -> str:
    return "multipath" if mode == "adaptive" else mode


def infer_from_path(path: Path) -> dict[str, str | int]:
    info: dict[str, str | int] = {}
    if path.parent.name in COLLECTIVES:
        info["collective"] = path.parent.name
    m = FILE_RE.search(path.name)
    if m:
        info["cards"] = int(m.group("cards"))
        info["mode"] = normalize_mode(m.group("mode"))
        info["timestamp"] = m.group("timestamp")
    return info


def parse_log(path: Path) -> list[dict[str, object]]:
    inferred = infer_from_path(path)
    rows: list[dict[str, object]] = []
    current = {
        "collective": inferred.get("collective", ""),
        "mode": inferred.get("mode", ""),
        "cards": inferred.get("cards", ""),
        "timestamp": inferred.get("timestamp", ""),
        "size_bytes": "",
        "elems_arg": "",
        "iters": "",
    }
    per_size_total_index: defaultdict[tuple[object, object, object, object], int] = defaultdict(int)
    per_size_alg_index: defaultdict[tuple[object, object, object, object], int] = defaultdict(int)
    in_warmup: bool = True  # True until "Warmup done" or first "normal test on iter" seen

    with path.open("r", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            header = HEADER_RE.search(line)
            if header:
                current["collective"] = header.group("collective")
                current["mode"] = normalize_mode(header.group("mode"))
                current["cards"] = int(header.group("cards"))
                current["timestamp"] = header.group("timestamp")
                continue

            case = CASE_RE.search(line)
            if case:
                current["collective"] = case.group("collective")
                current["mode"] = normalize_mode(case.group("mode"))
                current["cards"] = int(case.group("cards"))
                current["size_bytes"] = int(case.group("size"))
                current["elems_arg"] = int(case.group("elems"))
                current["iters"] = int(case.group("iters"))
                in_warmup = True  # reset warmup state for each new case
                continue

            if WARMUP_DONE_RE.search(line):
                in_warmup = False
            elif BASELINE_ITER_RE.search(line):
                in_warmup = False

            agg = AGG_RE.search(line)
            if agg and not current.get("size_bytes"):
                current["size_bytes"] = int(agg.group("size"))
                current["cards"] = int(agg.group("cards"))

            avg = BASELINE_AVG_RE.search(line)
            if avg:
                rows.append({
                    **current,
                    "mode": normalize_mode(str(current.get("mode", "baseline"))),
                    "metric": "baseline_average_bandwidth",
                    "sample_index": 0,
                    "bandwidth_gbps": float(avg.group("bw")) * GIB_TO_GB,
                    "measure_iters": int(avg.group("iters")),
                    "is_warmup": False,
                    "log_file": str(path),
                    "line_no": line_no,
                })
                continue

            mpi_result = MPI_RESULT_RE.search(line)
            if mpi_result:
                # Convert payload-convention algbw to the project per-rank op_bytes
                # convention (op_bytes/payload ratio is 1 for allreduce, 1/n for
                # allgather/reducescatter/alltoall), matching AMPCCL aggregate lines
                # and the single-process baseline tests.
                op_bytes = float(mpi_result.group("op_bytes"))
                payload_bytes = float(mpi_result.group("payload_bytes"))
                ratio = op_bytes / payload_bytes if payload_bytes > 0 else 1.0
                algbw_project = float(mpi_result.group("algbw")) * ratio
                mode = str(current.get("mode", "")) or "baseline"
                if mode == "baseline":
                    metric = "baseline_average_bandwidth"
                else:
                    # Informational only: end-to-end multipath timing from the MPI
                    # binary. The standard multipath metrics still come from the
                    # AMPCCL aggregate lines parsed below.
                    metric = "multipath_result_bandwidth"
                rows.append({
                    **current,
                    "mode": normalize_mode(mode),
                    "metric": metric,
                    "sample_index": 0,
                    "bandwidth_gbps": algbw_project,
                    "measure_iters": current.get("iters", ""),
                    "is_warmup": False,
                    "log_file": str(path),
                    "line_no": line_no,
                })
                continue

            total = TOTAL_BW_RE.search(line)
            if total:
                key = (
                    current.get("collective", ""),
                    current.get("cards", ""),
                    current.get("mode", "multipath"),
                    current.get("size_bytes", ""),
                )
                sample_index = per_size_total_index[key]
                per_size_total_index[key] += 1
                rows.append({
                    **current,
                    "mode": "multipath",
                    "metric": "multipath_total_bandwidth",
                    "sample_index": sample_index,
                    "bandwidth_gbps": float(total.group("bw")),
                    "measure_iters": current.get("iters", ""),
                    "is_warmup": in_warmup,
                    "log_file": str(path),
                    "line_no": line_no,
                })

            # Also extract algbw from AMPCCL aggregate lines for multipath
            agg_alg = AGG_ALGBW_RE.search(line)
            if agg_alg:
                key = (
                    current.get("collective", ""),
                    current.get("cards", ""),
                    current.get("mode", "multipath"),
                    current.get("size_bytes", ""),
                )
                sample_index = per_size_alg_index[key]
                per_size_alg_index[key] += 1
                if not current.get("size_bytes"):
                    current["size_bytes"] = int(agg_alg.group("size"))
                    current["cards"] = int(agg_alg.group("cards"))
                rows.append({
                    **current,
                    "mode": "multipath",
                    "metric": "multipath_alg_bandwidth",
                    "sample_index": sample_index,
                    "bandwidth_gbps": float(agg_alg.group("algbw")),
                    "measure_iters": current.get("iters", ""),
                    "is_warmup": in_warmup,
                    "log_file": str(path),
                    "line_no": line_no,
                })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: defaultdict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("collective"),
            row.get("mode"),
            row.get("cards"),
            row.get("size_bytes"),
            row.get("metric"),
        )
        grouped[key].append(float(row["bandwidth_gbps"]))

    out = []
    for (collective, mode, cards, size_bytes, metric), values in sorted(grouped.items(), key=lambda x: tuple(str(v) for v in x[0])):
        out.append({
            "collective": collective,
            "mode": mode,
            "cards": cards,
            "size_bytes": size_bytes,
            "metric": metric,
            "samples": len(values),
            "bandwidth_mean_gbps": mean(values),
            "bandwidth_median_gbps": median(values),
            "bandwidth_min_gbps": min(values),
            "bandwidth_max_gbps": max(values),
        })
    return out


def collect_logs(logs_root: Path, collectives: tuple[str, ...]) -> list[Path]:
    logs = []
    for collective in collectives:
        cdir = logs_root / collective
        if cdir.is_dir():
            logs.extend(sorted(cdir.glob("*.log")))
    if not logs:
        logs = sorted(logs_root.rglob("*.log"))
    return logs


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, default=script_dir / "logs" / "compare")
    parser.add_argument("--csv-dir", type=Path, default=script_dir / "csv")
    parser.add_argument("--collectives", nargs="+", default=list(COLLECTIVES))
    parser.add_argument("--raw-name", default="raw_bandwidth.csv")
    parser.add_argument("--summary-name", default="summary_bandwidth.csv")
    args = parser.parse_args()

    logs = collect_logs(args.logs_root, tuple(args.collectives))
    rows: list[dict[str, object]] = []
    for log in logs:
        rows.extend(parse_log(log))

    raw_fields = [
        "collective", "mode", "cards", "size_bytes", "elems_arg", "iters", "metric",
        "sample_index", "bandwidth_gbps", "measure_iters", "is_warmup", "timestamp", "log_file", "line_no",
    ]
    summary_fields = [
        "collective", "mode", "cards", "size_bytes", "metric", "samples",
        "bandwidth_mean_gbps", "bandwidth_median_gbps", "bandwidth_min_gbps", "bandwidth_max_gbps",
    ]
    summary_rows = summarize(rows)

    write_csv(args.csv_dir / args.raw_name, rows, raw_fields)
    write_csv(args.csv_dir / args.summary_name, summary_rows, summary_fields)

    print(f"logs_scanned={len(logs)}")
    print(f"raw_rows={len(rows)} -> {args.csv_dir / args.raw_name}")
    print(f"summary_rows={len(summary_rows)} -> {args.csv_dir / args.summary_name}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
