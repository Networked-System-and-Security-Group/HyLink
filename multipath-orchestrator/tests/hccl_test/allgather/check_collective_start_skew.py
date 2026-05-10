"""Check that input preparation is not included as cross-rank collective skew."""
import argparse
import re
from pathlib import Path


def check(path, devices, iters, limit_ms):
    rows = {}
    for device, iteration, start in re.findall(
        r"collective device=(\d+) iter=(\d+) start_ms=([\d.]+)", path.read_text()
    ):
        key = int(iteration)
        assert int(device) not in rows.setdefault(key, {}), "duplicate device/iteration"
        rows[key][int(device)] = float(start)
    assert set(rows) == set(range(iters)), "incomplete iteration timeline"
    skews = []
    for iteration, starts in sorted(rows.items()):
        assert set(starts) == set(range(devices)), "incomplete device timeline"
        skew = max(starts.values()) - min(starts.values())
        skews.append(skew)
        assert skew <= limit_ms, f"iter={iteration}: start skew {skew:.3f} ms exceeds {limit_ms} ms"
    print(f"PASS: {iters} iterations, max cross-device start skew={max(skews):.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--limit-ms", type=float, default=10.0)
    args = parser.parse_args()
    check(args.log, args.devices, args.iters, args.limit_ms)
