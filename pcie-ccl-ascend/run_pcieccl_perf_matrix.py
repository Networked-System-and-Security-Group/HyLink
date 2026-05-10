#!/usr/bin/env python3
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BENCH_SP = ROOT / "build" / "bench_hccl_compat_sp"
PCCL_SO = ROOT / "build" / "libpccl.so"
HCCL_TEST_BIN = ROOT / "build" / "hccl_test" / "bin"

SIZES = [16 << 20, 32 << 20, 64 << 20, 128 << 20, 256 << 20, 512 << 20]
SIZE_LABELS = {
    16 << 20: "16M",
    32 << 20: "32M",
    64 << 20: "64M",
    128 << 20: "128M",
    256 << 20: "256M",
    512 << 20: "512M",
}
CARDS = {
    2: "0,1",
    4: "0,1,2,3",
    8: "0,1,2,3,4,5,6,7",
}

PRIMITIVES = {
    "AllGather": {
        "sp_section": "all_gather_sp",
        "mp_bin": "all_gather_test",
        "needs_op": False,
    },
    "ReduceScatter": {
        "sp_section": "reduce_scatter_sp",
        "mp_bin": "reduce_scatter_test",
        "needs_op": True,
    },
    "AllReduce": {
        "sp_section": "all_reduce_sp",
        "mp_bin": "all_reduce_test",
        "needs_op": True,
    },
}


def bus_factor(primitive, nranks):
    if primitive == "AllReduce":
        return 2.0 * (nranks - 1.0) / nranks
    return (nranks - 1.0) / nranks


def base_env():
    env = os.environ.copy()
    env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    return env


def run_command(name, cmd, env, log_dir):
    print(f"[RUN] {name}", flush=True)
    print("      " + " ".join(cmd), flush=True)
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - started
    log_path = log_dir / f"{name}.log"
    log_path.write_text(proc.stdout, encoding="utf-8")
    print(f"[DONE] {name} rc={proc.returncode} elapsed={elapsed:.1f}s log={log_path}", flush=True)
    if proc.returncode != 0:
        print(proc.stdout, flush=True)
        raise RuntimeError(f"{name} failed with rc={proc.returncode}")
    return proc.stdout


def parse_rows(text, has_busbw):
    rows = {}
    if has_busbw:
        pattern = re.compile(
            r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*(\w+)",
            re.MULTILINE,
        )
        for match in pattern.finditer(text):
            size = int(match.group(1))
            rows[size] = {
                "time_us": float(match.group(2)),
                "algbw": float(match.group(3)),
                "busbw": float(match.group(4)),
                "check": match.group(5),
            }
    else:
        pattern = re.compile(
            r"^\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*(\w+)",
            re.MULTILINE,
        )
        for match in pattern.finditer(text):
            size = int(match.group(1))
            rows[size] = {
                "time_us": float(match.group(2)),
                "algbw": float(match.group(3)),
                "check": match.group(4),
            }
    return rows


def parse_sp(text):
    parsed = {}
    for primitive, meta in PRIMITIVES.items():
        section = meta["sp_section"]
        marker = f"=== {section} ==="
        start = text.find(marker)
        if start < 0:
            raise RuntimeError(f"SP output missing section {section}")
        next_start = text.find("===", start + len(marker))
        block = text[start:] if next_start < 0 else text[start:next_start]
        parsed[primitive] = parse_rows(block, has_busbw=True)
    return parsed


def parse_mp(text, primitive, nranks):
    rows = parse_rows(text, has_busbw=False)
    factor = bus_factor(primitive, nranks)
    for row in rows.values():
        row["busbw"] = row["algbw"] * factor
    return rows


def check_rows(rows, context):
    missing = [SIZE_LABELS[s] for s in SIZES if s not in rows]
    if missing:
        raise RuntimeError(f"{context} missing sizes: {', '.join(missing)}")
    bad = [SIZE_LABELS[s] for s in SIZES if rows[s].get("check") != "success"]
    if bad:
        raise RuntimeError(f"{context} failed checks: {', '.join(bad)}")


def pct(num, den):
    return (num / den * 100.0) if den else 0.0


def fmt(v, ndigits=2):
    return f"{v:.{ndigits}f}"


def make_markdown(results, run_meta, raw_dir):
    lines = []
    lines.append("# PCIe CCL vs Native HCCL Performance Report")
    lines.append("")
    lines.append(f"- Date: {run_meta['date']}")
    lines.append(f"- Host: `{run_meta['host']}`")
    lines.append("- Device: Ascend 910B3, single node, cards 0-7")
    lines.append("- Data type: `fp32`; reduce op: `sum` for AllReduce/ReduceScatter")
    lines.append(f"- Message sizes: {', '.join(SIZE_LABELS[s] for s in SIZES)}")
    lines.append(f"- Iterations: `{run_meta['iters']}` measured, `{run_meta['warmup']}` warmup")
    lines.append("- SP suite: `bench_hccl_compat_sp`; MP suite: `hccl_test`")
    lines.append("- Baseline: native HCCL library; target: `LD_PRELOAD=build/libpccl.so` PCIe CCL")
    lines.append(f"- Raw logs: `{raw_dir}`")
    lines.append("")
    lines.append("Percent rules: `Time % = PCIe CCL time / HCCL time * 100` (lower is better); `AlgBW %` and `BusBW % = PCIe CCL / HCCL * 100` (higher is better).")
    lines.append("For MP, `hccl_test` prints execution time and algbw only; busbw below is computed with the standard collective bus factor: AllReduce `2*(N-1)/N`, AllGather/ReduceScatter `(N-1)/N`.")
    lines.append("")

    for model in ["SP", "MP"]:
        lines.append(f"## {model}")
        lines.append("")
        for primitive in ["AllGather", "ReduceScatter", "AllReduce"]:
            for nranks in [2, 4, 8]:
                lines.append(f"### {primitive}, {nranks} cards")
                lines.append("")
                lines.append("| Size | PCIe time (us) | PCIe algbw (GB/s) | PCIe busbw (GB/s) | HCCL time (us) | HCCL algbw (GB/s) | HCCL busbw (GB/s) | Time % | AlgBW % | BusBW % |")
                lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
                pcie_rows = results[model][nranks]["pcie"][primitive]
                hccl_rows = results[model][nranks]["hccl"][primitive]
                for size in SIZES:
                    pcie = pcie_rows[size]
                    hccl = hccl_rows[size]
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                SIZE_LABELS[size],
                                fmt(pcie["time_us"]),
                                fmt(pcie["algbw"], 5),
                                fmt(pcie["busbw"], 5),
                                fmt(hccl["time_us"]),
                                fmt(hccl["algbw"], 5),
                                fmt(hccl["busbw"], 5),
                                fmt(pct(pcie["time_us"], hccl["time_us"])),
                                fmt(pct(pcie["algbw"], hccl["algbw"])),
                                fmt(pct(pcie["busbw"], hccl["busbw"])),
                            ]
                        )
                        + " |"
                    )
                lines.append("")
    return "\n".join(lines)


def main():
    if not BENCH_SP.exists():
        raise RuntimeError(f"missing {BENCH_SP}")
    if not PCCL_SO.exists():
        raise RuntimeError(f"missing {PCCL_SO}")

    iters = int(os.environ.get("PCCL_PERF_ITERS", "10"))
    warmup = int(os.environ.get("PCCL_PERF_WARMUP", "2"))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "perf_results" / stamp
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    results = {"SP": {}, "MP": {}}

    for nranks, devices in CARDS.items():
        devices_arg = f"--devices={devices}"

        env_hccl_sp = base_env()
        env_hccl_sp["ASCEND_RT_VISIBLE_DEVICES"] = devices
        out = run_command(
            f"sp_hccl_{nranks}cards",
            [
                str(BENCH_SP),
                str(nranks),
                "16M",
                "512M",
                str(iters),
                str(warmup),
                devices_arg,
            ],
            env_hccl_sp,
            log_dir,
        )
        sp_hccl = parse_sp(out)
        for primitive, rows in sp_hccl.items():
            check_rows(rows, f"SP HCCL {primitive} {nranks} cards")

        env_pcie_sp = base_env()
        env_pcie_sp["LD_PRELOAD"] = str(PCCL_SO)
        env_pcie_sp["ASCEND_RT_VISIBLE_DEVICES"] = devices
        out = run_command(
            f"sp_pcie_{nranks}cards",
            [
                str(BENCH_SP),
                str(nranks),
                "16M",
                "512M",
                str(iters),
                str(warmup),
                "--init=root",
                "--rootinfo=zero",
                devices_arg,
            ],
            env_pcie_sp,
            log_dir,
        )
        sp_pcie = parse_sp(out)
        for primitive, rows in sp_pcie.items():
            check_rows(rows, f"SP PCIe {primitive} {nranks} cards")

        results["SP"][nranks] = {"hccl": sp_hccl, "pcie": sp_pcie}

    for nranks, devices in CARDS.items():
        results["MP"][nranks] = {"hccl": {}, "pcie": {}}
        for primitive, meta in PRIMITIVES.items():
            bin_path = HCCL_TEST_BIN / meta["mp_bin"]
            common_args = [
                str(bin_path),
                "-b",
                "16M",
                "-e",
                "512M",
                "-f",
                "2",
                "-d",
                "fp32",
                "-p",
                str(nranks),
                "-n",
                str(iters),
                "-w",
                str(warmup),
                "-c",
                "1",
            ]
            if meta["needs_op"]:
                common_args.extend(["-o", "sum"])

            env_hccl_mp = base_env()
            env_hccl_mp["ASCEND_RT_VISIBLE_DEVICES"] = devices
            env_hccl_mp["HCCL_TEST_USE_DEVS"] = devices
            out = run_command(
                f"mp_hccl_{primitive.lower()}_{nranks}cards",
                ["mpirun", "--allow-run-as-root", "-n", str(nranks), "env"] + common_args,
                env_hccl_mp,
                log_dir,
            )
            rows = parse_mp(out, primitive, nranks)
            check_rows(rows, f"MP HCCL {primitive} {nranks} cards")
            results["MP"][nranks]["hccl"][primitive] = rows

            env_pcie_mp = base_env()
            env_pcie_mp["LD_PRELOAD"] = str(PCCL_SO)
            env_pcie_mp["ASCEND_RT_VISIBLE_DEVICES"] = devices
            env_pcie_mp["HCCL_TEST_USE_DEVS"] = devices
            out = run_command(
                f"mp_pcie_{primitive.lower()}_{nranks}cards",
                ["mpirun", "--allow-run-as-root", "-n", str(nranks), "env"] + common_args,
                env_pcie_mp,
                log_dir,
            )
            rows = parse_mp(out, primitive, nranks)
            check_rows(rows, f"MP PCIe {primitive} {nranks} cards")
            results["MP"][nranks]["pcie"][primitive] = rows

    serializable = json.loads(json.dumps(results))
    (out_dir / "results.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    run_meta = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "iters": iters,
        "warmup": warmup,
    }
    md = make_markdown(results, run_meta, log_dir)
    md_path = out_dir / "pcieccl_perf_report.md"
    md_path.write_text(md + "\n", encoding="utf-8")
    latest = ROOT / "PCIe_CCL_perf_report.md"
    latest.write_text(md + "\n", encoding="utf-8")

    print(f"[REPORT] {md_path}", flush=True)
    print(f"[REPORT] {latest}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
