# Trace Analysis Scripts

Post-processing scripts for analyzing PCCL Perfetto/Chrome JSON trace files.

## Common Library

`trace_common.py` provides shared utilities used by all analysis scripts:
- Trace loading and event classification (`TraceEvents`)
- Iteration detection from scheduler events
- Percentile computation and formatting
- Time span and size formatting helpers

## Analysis Scripts

| Script | Description |
|---|---|
| `analyze_phases.py` | Per-iteration phase span analysis (D2H, H2H, H2D). Reports SpinWait breakdown and pipeline overlap between phases. |
| `analyze_trace.py` | General trace analysis: transfer size distributions, H2H copy/reduce bandwidth, SpinWait statistics, per-iteration timing, and per-rank timing. |
| `analyze_by_size.py` | Group iterations by data size and report per-size-group latency, phase breakdown, achieved algorithm bandwidth, and subchunk-level statistics. |
| `analyze_h2h.py` | H2H subchunk duration distribution. Detects preempted events (high wall-time vs low CPU-time) and reports per-CPU breakdown of stragglers. |
| `merge_msprof.py` | Merge Ascend msprof NPU hardware trace into a PCCL trace file. Aligns timestamps via `wall_epoch_us` and imports DMA, compute, and flow events. |

## Usage

```bash
python3 analyze_phases.py <trace.json>
python3 analyze_trace.py <trace.json>
python3 analyze_by_size.py <trace.json>
python3 analyze_h2h.py <trace.json> [label]
python3 merge_msprof.py <pccl_trace.json> <msprof.json> [-o merged.json]
```
