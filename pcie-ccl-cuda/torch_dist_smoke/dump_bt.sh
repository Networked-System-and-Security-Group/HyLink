#!/usr/bin/env bash
# Attach gdb to each python worker (excluding the launcher) and dump all-thread backtraces.
set -uo pipefail
OUT=${1:-/tmp/bt}
mkdir -p "$OUT"
pids=$(pgrep -f "deadlock_repro.py|pretrain_gpt.py" | sort -u)
echo "worker pids: $pids"
for p in $pids; do
  # skip torchrun launcher (it has 'torch.distributed.run' in cmdline)
  if tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "distributed.run"; then
    echo "skip launcher $p"; continue
  fi
  echo "=== dumping $p ==="
  timeout 60 gdb -p "$p" -batch \
    -ex "set pagination off" \
    -ex "info threads" \
    -ex "thread apply all bt" \
    > "$OUT/bt_$p.txt" 2>&1
  echo "  -> $OUT/bt_$p.txt ($(wc -l < $OUT/bt_$p.txt) lines)"
done
echo "done"
