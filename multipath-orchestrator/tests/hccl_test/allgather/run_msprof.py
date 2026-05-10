#!/usr/bin/env python3
"""Paired, isolated four-rank AllGather profiler runs; keep raw data outside git."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def benchmark_options(iters, warmup, ranges, interference_size):
    if interference_size <= 0:
        raise ValueError('interference size must be positive')
    return ['--devices=4', '--send-count=67108864', '--iters='+str(iters),
            '--warmup='+str(warmup), '--interference-size='+str(interference_size),
            '--interference-ranges='+ranges]


def run_modes(profile_only, baseline_only):
    if profile_only and baseline_only:
        raise ValueError('profile-only and baseline-only are mutually exclusive')
    return [True] if profile_only else [False] if baseline_only else [False, True]


def h2h_shard_settings(size):
    if size < 0:
        raise ValueError('H2H shard size cannot be negative')
    return {'PCCL_H2H_SHARD_BYTES': str(size)} if size else {}


def completion_wait_settings(mode):
    if mode not in ('value', 'event'):
        raise ValueError('completion wait mode must be value or event')
    return {'PCCL_COMPLETION_WAIT_MODE': mode}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--app', type=Path, required=True)
    parser.add_argument('--pccl-library', type=Path, required=True)
    parser.add_argument('--amp-library', type=Path, required=True)
    parser.add_argument('--algorithms', default='split_ratio,tcp')
    parser.add_argument('--iters', type=int, default=80)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--ranges', default='20-39,55-64')
    parser.add_argument('--interference-size', type=int, default=134217728)
    parser.add_argument('--completion-wait-mode', choices=('value', 'event'), default='value')
    parser.add_argument('--h2h-shard-bytes', type=int, default=0,
                        help='pin H2H shard threshold across subchunk builds; 0 keeps library default')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--profile-only', action='store_true')
    modes.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args()
    shard_env = h2h_shard_settings(args.h2h_shard_bytes)
    completion_env = completion_wait_settings(args.completion_wait_mode)
    args.output.mkdir(parents=True, exist_ok=False)
    script_dir = Path(__file__).resolve().parent
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('AMPCCL_', 'PCCL_', 'CCL_')) or key == 'LD_PRELOAD':
            env.pop(key)
    env.update(NP='4', ACL_VISIBLE_DEVICES='0,1,2,3', ASCEND_RT_VISIBLE_DEVICES='0,1,2,3',
               AMPCCL_ENABLE='1', AMPCCL_ENABLE_PCIE='1', AMPCCL_SHM_PARAMS='1',
               AMPCCL_LOG_LEVEL='info', AMPCCL_LOG_COLOR='off', AMPCCL_STATIC_ALPHA='0.125',
               AMPCCL_SPLIT_RATIO_INITIAL_ALPHA='0.16', AMPCCL_SPLIT_RATIO_ETA='0.1',
               AMPCCL_SPLIT_RATIO_SMOOTH='0.4', AMPCCL_SPLIT_RATIO_MIN_ALPHA='0.05',
               AMPCCL_SPLIT_RATIO_WARMUP_STEPS='5', PCCL_FORCE_DMA_LANES='4',
               PCCL_H2H_WORKERS_PER_NUMA='8', PCCL_H2H_RANGE_ENABLE='0',
               PCCL_DISABLE_STAGING_LEASE='0', AMP_CCL_PATH=str(args.amp_library.resolve()),
               LD_PRELOAD=str(args.pccl_library.resolve()), CCL_PROFILE_APP=str(args.app.resolve()))
    env.update(shard_env)
    env.update(completion_env)
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (args.app, args.pccl_library, args.amp_library)}
    (args.output / 'binaries.json').write_text(json.dumps(hashes, indent=2))
    msprof = shutil.which('msprof')
    if not msprof:
        raise SystemExit('msprof not found')
    for algo in args.algorithms.split(','):
        if algo not in ('static', 'split_ratio', 'tcp'):
            raise SystemExit('unsupported algorithm: ' + algo)
        for profile in run_modes(args.profile_only, args.baseline_only):
            case = args.output / (algo + ('_msprof' if profile else '_baseline'))
            case.mkdir()
            case_env = dict(env, AMPCCL_ALGO=algo)
            command = ['bash', str(script_dir / 'run_mpi_with_pcie.sh'),
                       str(script_dir / 'msprof_rank.sh')] + benchmark_options(
                           args.iters, args.warmup, args.ranges, args.interference_size)
            if profile:
                command = [msprof, '--output=' + str(case / 'profile'), '--ascendcl=on',
                           '--runtime-api=on', '--task-time=l1', '--ai-core=off', '--aicpu=off',
                           '--msproftx=on'] + command
            (case / 'metadata.json').write_text(json.dumps(dict(command=command,
                environment={k: v for k, v in case_env.items() if k.startswith(
                    ('AMPCCL_', 'PCCL_', 'CCL_', 'LD_', 'ASCEND_', 'ACL_'))}), indent=2))
            print('START', case.name, flush=True)
            with (case / 'run.log').open('w') as log:
                result = subprocess.run(command, env=case_env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=900)
            log_text = (case / 'run.log').read_text(errors='replace')
            passed = len(re.findall(r'^rank [0-3]: PASSED', log_text, re.M))
            if result.returncode or passed != 4 or 'Test PASSED.' not in log_text:
                raise SystemExit(f'{case.name} failed: exit={result.returncode}, passed={passed}; see run.log')
            if profile and not list(case.rglob('msprof_*.json')):
                raise SystemExit(f'{case.name}: no exported msprof timeline')
            print('PASS', case.name, flush=True)


if __name__ == '__main__':
    main()
