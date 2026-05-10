"""Shared utilities for PCCL trace analysis scripts."""
import json


def load_events(path):
    """Load Chrome JSON trace events from a file."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, {}
    return data.get("traceEvents", []), data


class TraceEvents:
    """Classified trace events from a PCCL trace file."""

    def __init__(self, d2h, h2d, h2h_sc, h2h_spin, h2h_outer, sched, all_events):
        self.d2h = d2h
        self.h2d = h2d
        self.h2h_sc = h2h_sc
        self.h2h_spin = h2h_spin
        self.h2h_outer = h2h_outer
        self.sched = sched
        self.all_events = all_events

    @classmethod
    def from_file(cls, path):
        events, _ = load_events(path)
        return cls.classify(events)

    @classmethod
    def classify(cls, events):
        d2h = []
        h2d = []
        h2h_sc = []
        h2h_spin = []
        h2h_outer = []
        sched = []

        for e in events:
            if e.get("ph") != "X":
                continue
            cat = e.get("cat", "")
            name = e.get("name", "")
            if cat == "gpu_dma" or cat == "d2h" or cat == "h2d":
                if "D2H" in name or cat == "d2h":
                    d2h.append(e)
                elif "H2D" in name or cat == "h2d":
                    h2d.append(e)
            elif cat == "h2h":
                if "SpinWait" in name:
                    h2h_spin.append(e)
                elif "sc=" in name:
                    h2h_sc.append(e)
                else:
                    h2h_outer.append(e)
            elif cat == "sched":
                sched.append(e)

        return cls(d2h, h2d, h2h_sc, h2h_spin, h2h_outer, sched, events)


def detect_iterations(sched_events, gap_us=2000):
    """Cluster scheduler events into iterations by time gap."""
    if not sched_events:
        return []
    sched_sorted = sorted(sched_events, key=lambda e: e["ts"])
    iterations = [[sched_sorted[0]]]
    for e in sched_sorted[1:]:
        if e["ts"] - iterations[-1][-1]["ts"] > gap_us:
            iterations.append([e])
        else:
            iterations[-1].append(e)
    return iterations


def iter_windows(iterations, all_events):
    """Return (start_us, end_us) for each iteration.

    The end of iteration i is the start of iteration i+1, except for the
    last iteration where the end is the latest event timestamp.
    """
    if not iterations:
        return []
    windows = []
    for i, it in enumerate(iterations):
        start = min(e["ts"] for e in it)
        if i + 1 < len(iterations):
            end = min(e["ts"] for e in iterations[i + 1])
        else:
            end = max(
                e["ts"] + e.get("dur", 0)
                for e in all_events
                if e.get("ph") == "X"
            )
        windows.append((start, end))
    return windows


def events_in_window(events, start, end):
    """Filter events whose ts falls within [start, end)."""
    return [e for e in events if start <= e["ts"] < end]


def span_us(events):
    """Time span from earliest start to latest end, in microseconds."""
    if not events:
        return 0
    s = min(e["ts"] for e in events)
    e_end = max(e["ts"] + e.get("dur", 0) for e in events)
    return e_end - s


def percentile(sorted_arr, p):
    """Return the p-th percentile (0-100) from a pre-sorted array."""
    if not sorted_arr:
        return 0
    idx = min(int(len(sorted_arr) * p / 100), len(sorted_arr) - 1)
    return sorted_arr[idx]


def percentiles_str(values, pcts=None, unit="us"):
    """Format percentile summary for a list of values (not necessarily sorted)."""
    if pcts is None:
        pcts = [0, 50, 90, 99, 100]
    s = sorted(values)
    parts = [f"P{p}={percentile(s, p)}{unit}" for p in pcts]
    return "  ".join(parts)


def format_size(nbytes):
    """Format byte count as human-readable string."""
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024 * 1024:
        return f"{nbytes // 1024} KB"
    elif nbytes < 1024 * 1024 * 1024:
        return f"{nbytes // (1024 * 1024)} MB"
    else:
        return f"{nbytes / (1024 * 1024 * 1024):.1f} GB"


def get_rank(event):
    """Parse rank number from an event name like 'Rank 3 ...'."""
    name = event.get("name", "")
    if "Rank " in name:
        try:
            rest = name.split("Rank ")[1]
            return int(rest.split()[0].strip(","))
        except (ValueError, IndexError):
            pass
    return None
