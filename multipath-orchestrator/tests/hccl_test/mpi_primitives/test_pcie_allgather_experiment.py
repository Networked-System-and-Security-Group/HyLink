#!/usr/bin/env python3
"""Behavioral tests for the MPI AllGather PCIe experiment helpers."""

from pathlib import Path
import os
import subprocess

import pytest


MODULE_DIR = Path(__file__).resolve().parent


def test_half_open_ranges_exclude_right_endpoint():
    from plot_pcie_allgather import active_iterations, parse_ranges

    ranges = parse_ranges("10-20,30-40")

    assert ranges == [(10, 20), (30, 40)]
    assert active_iterations(50, ranges) == list(range(10, 20)) + list(range(30, 40))
    assert len(active_iterations(50, ranges)) == 20


def test_log_parser_assigns_previous_round_algbw_to_formal_iter():
    from plot_pcie_allgather import parse_log_text

    log = """
[rank0] begin formal 0 iters
[AMPCCL][OFF] aggregate op=AllGather op_key_bytes=134217728 n=8 algbw(GB/s,decimal)=99.000000
[rank0] begin formal 1 iters
[AMPCCL][OFF] aggregate op=AllGather op_key_bytes=134217728 n=8 algbw(GB/s,decimal)=11.500000
[rank0] begin formal 2 iters
[AMPCCL][OFF] aggregate op=AllGather op_key_bytes=134217728 n=8 algbw(GB/s,decimal)=12.500000
"""

    assert parse_log_text(log, expected_size_bytes=134217728, expected_ranks=8) == {
        0: 11.5,
        1: 12.5,
    }


def test_path_log_parser_extracts_hccl_pcie_and_total_bandwidths():
    from plot_pcie_allgather import PathBandwidth, parse_path_bandwidths_text

    log = """
[rank0] begin formal 0 iters
[AMPCCL][OFF] aggregate op=AllGather op_key_bytes=134217728 base_bytes=134217728 alg_bytes=1073741824 n=8 | fast_bytes=123456789 pcie_bytes=10760939 split_fast=0.9200 split_pcie=0.0800 | fast_bw(GB/s,decimal)=16.000000 pcie_bw(GB/s,decimal)=1.000000 | algbw(GB/s,decimal)=98.000000 busbw(GB/s,decimal)=85.750000 bus_factor=0.875000
[rank0] begin formal 1 iters
[AMPCCL][OFF] aggregate op=AllGather op_key_bytes=134217728 base_bytes=134217728 alg_bytes=1073741824 n=8 | fast_bytes=123456789 pcie_bytes=10760939 split_fast=0.9200 split_pcie=0.0800 | fast_bw(GB/s,decimal)=17.250000 pcie_bw(GB/s,decimal)=1.125000 | algbw(GB/s,decimal)=99.500000 busbw(GB/s,decimal)=87.062500 bus_factor=0.875000
"""

    assert parse_path_bandwidths_text(
        log, expected_size_bytes=134217728, expected_ranks=8
    ) == {
        0: PathBandwidth(hccl=91.54, pcie=7.96, total=99.5),
    }


def test_path_log_parser_reports_component_allgather_algbw():
    from plot_pcie_allgather import PathBandwidth, parse_path_bandwidths_text

    log = """
[rank0] begin formal 0 iters
[AMPCCL][OFF] aggregate op=AllGather op_key_bytes=134217728 base_bytes=134217728 alg_bytes=1073741824 n=8 | fast_bytes=123456789 pcie_bytes=10760939 split_fast=0.9200 split_pcie=0.0800 | fast_bw(GB/s,decimal)=16.000000 pcie_bw(GB/s,decimal)=1.000000 | algbw(GB/s,decimal)=98.000000 busbw(GB/s,decimal)=85.750000 bus_factor=0.875000
[rank0] begin formal 1 iters
[AMPCCL][OFF] aggregate op=AllGather op_key_bytes=134217728 base_bytes=134217728 alg_bytes=1073741824 n=8 | fast_bytes=123456789 pcie_bytes=10760939 split_fast=0.9200 split_pcie=0.0800 | fast_bw(GB/s,decimal)=16.000000 pcie_bw(GB/s,decimal)=1.000000 | algbw(GB/s,decimal)=98.000000 busbw(GB/s,decimal)=85.750000 bus_factor=0.875000
"""

    parsed = parse_path_bandwidths_text(
        log, expected_size_bytes=134217728, expected_ranks=8
    )
    assert parsed[0].hccl == pytest.approx(90.16)
    assert parsed[0].pcie == pytest.approx(7.84)
    assert parsed[0].total == pytest.approx(98.0)


def test_path_log_parser_extracts_previous_round_path_times_in_ms():
    from plot_pcie_allgather import PathTiming, parse_path_timings_text

    log = """
[rank0] begin formal 0 iters
[AMPCCL][OFF] (max) fast_time=0.006000s
[AMPCCL][OFF] (max) pcie_time=0.007000s
[AMPCCL][OFF] aggregate op=AllGather op_key_bytes=134217728 base_bytes=134217728 alg_bytes=1073741824 n=8 | fast_bytes=123456789 pcie_bytes=10760939 split_fast=0.9200 split_pcie=0.0800 | fast_bw(GB/s,decimal)=16.000000 pcie_bw(GB/s,decimal)=1.000000 | algbw(GB/s,decimal)=98.000000 busbw(GB/s,decimal)=85.750000 bus_factor=0.875000
[rank0] begin formal 1 iters
[AMPCCL][OFF] (max) fast_time=0.007500s
[AMPCCL][OFF] (max) pcie_time=0.008250s
[AMPCCL][OFF] aggregate op=AllGather op_key_bytes=134217728 base_bytes=134217728 alg_bytes=1073741824 n=8 | fast_bytes=123456789 pcie_bytes=10760939 split_fast=0.9200 split_pcie=0.0800 | fast_bw(GB/s,decimal)=17.250000 pcie_bw(GB/s,decimal)=1.125000 | algbw(GB/s,decimal)=99.500000 busbw(GB/s,decimal)=87.062500 bus_factor=0.875000
"""

    assert parse_path_timings_text(
        log, expected_size_bytes=134217728, expected_ranks=8
    ) == {
        0: PathTiming(hccl_ms=7.5, pcie_ms=8.25),
    } 


def test_path_log_parser_accepts_pure_hccl_zero_pcie_metrics():
    from plot_pcie_allgather import (
        PathBandwidth,
        PathMetrics,
        PathTiming,
        parse_path_metrics_text,
    )

    log = """
[rank0] begin formal 0 iters
[AMPCCL][OFF] (avg) fast_time=0.006000s
[AMPCCL][OFF] (avg) pcie_time=0.000000s
[AMPCCL][OFF] aggregate op=AllGather op_key_bytes=123480320 base_bytes=123480320 alg_bytes=987842560 n=8 | fast_bytes=123480320 pcie_bytes=0 split_fast=1.0000 split_pcie=0.0000 | fast_bw(GB/s,decimal)=20.000000 pcie_bw(GB/s,decimal)=0.000000 | algbw(GB/s,decimal)=160.000000 busbw(GB/s,decimal)=140.000000 bus_factor=0.875000
[rank0] begin formal 1 iters
[AMPCCL][OFF] (avg) fast_time=0.007500s
[AMPCCL][OFF] (avg) pcie_time=0.000000s
[AMPCCL][OFF] aggregate op=AllGather op_key_bytes=123480320 base_bytes=123480320 alg_bytes=987842560 n=8 | fast_bytes=123480320 pcie_bytes=0 split_fast=1.0000 split_pcie=0.0000 | fast_bw(GB/s,decimal)=16.464042 pcie_bw(GB/s,decimal)=0.000000 | algbw(GB/s,decimal)=131.712336 busbw=115.248294 bus_factor=0.875000
"""

    assert parse_path_metrics_text(
        log, expected_size_bytes=123480320, expected_ranks=8
    ) == {
        0: PathMetrics(
            bandwidth=PathBandwidth(hccl=131.712336, pcie=0.0, total=131.712336),
            timing=PathTiming(hccl_ms=7.5, pcie_ms=0.0),
        ),
    }


def test_smoke_runner_reports_continuous_pcie_mode():
    runner = MODULE_DIR / "run_pcie_allgather_experiment.sh"

    result = subprocess.run(
        ["bash", str(runner), "--dry-run"],
        cwd=MODULE_DIR,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "mode=smoke" in result.stdout
    assert "iters=50 warmup=10" in result.stdout
    assert "pcie_ranges=10-20,30-40" in result.stdout
    assert "pcie_mode=continuous" in result.stdout
    assert "algorithms=static:0.08,tcp" in result.stdout


def test_runner_dry_run_accepts_full_continuous_experiment_parameters():
    runner = MODULE_DIR / "run_pcie_allgather_experiment.sh"

    result = subprocess.run(
        ["bash", str(runner), "--dry-run"],
        cwd=MODULE_DIR,
        env={
            **os.environ,
            "MODE": "full",
            "STATIC_ALPHA": "0.075",
            "ITERS": "1001",
            "WARMUP": "10",
            "AG_TOTAL_BYTES": str(2 * 1024**3),
            "PCIE_SIZE_BYTES": str(128 * 1024**2),
            "PCIE_RANGES": "100-200,300-600,700-900",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "static_alpha=0.075" in result.stdout
    assert "ag_total=2GiB" in result.stdout
    assert "ag_per_rank=256MiB" in result.stdout
    assert "pcie_size=128MiB" in result.stdout
    assert "pcie_mode=continuous" in result.stdout


def test_static_no_pcie_runner_has_requested_fixed_configuration():
    runner = MODULE_DIR.parent / "run_allgather_static_no_pcie.sh"

    result = subprocess.run(
        ["bash", str(runner), "--dry-run"],
        cwd=MODULE_DIR.parent,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "np=8" in result.stdout
    assert "op=ag" in result.stdout
    assert "size=33554432" in result.stdout
    assert "iters=40 warmup=10" in result.stdout
    assert "static_alpha=0.08" in result.stdout
    assert "enable_pcie=0" in result.stdout


def test_split_ratio_static_full_runner_has_requested_configuration():
    runner = MODULE_DIR.parent / "run_allgather_split_ratio_vs_static_full.sh"

    result = subprocess.run(
        ["bash", str(runner), "--dry-run"],
        cwd=MODULE_DIR.parent,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "np=8 op=ag size=33554432 iters=1001 warmup=10" in result.stdout
    assert "ag_total_bytes=1073741824 ag_per_rank_bytes=134217728" in result.stdout
    assert "static_alpha=0.075" in result.stdout
    assert "split_ratio_eta=0.1 split_ratio_smooth=0.9" in result.stdout
    assert "enable_pcie=1 pccl_h2h_workers=8" in result.stdout
    assert "pcie_size_bytes=67108864" in result.stdout
    assert "pcie_ranges=100-200,300-600,700-900" in result.stdout


def test_mpi_binary_help_exposes_one_shot_pcie_options():
    binary = MODULE_DIR / "hccl_mpi_primitives"
    result = subprocess.run(
        [str(binary), "--help"],
        cwd=MODULE_DIR,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--pcie-ranges=start-end" in result.stdout
    assert "--pcie-size=BYTES" in result.stdout
    assert "continuous" in result.stdout


def test_plot_writer_has_a_png_fallback_without_matplotlib(tmp_path):
    from PIL import Image

    from plot_pcie_allgather import plot_bandwidths

    output = tmp_path / "smoke.png"
    plot_bandwidths(
        {0: 100.0, 1: 80.0, 2: 110.0},
        {0: 90.0, 1: 95.0, 2: 105.0},
        [(1, 3)],
        output,
        "smoke",
    )

    assert output.is_file()
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.width >= 800
        assert image.height >= 500


def test_plot_writer_accepts_a_custom_second_algorithm_label(tmp_path):
    from PIL import Image

    from plot_pcie_allgather import plot_bandwidths

    output = tmp_path / "split_ratio.png"
    plot_bandwidths(
        {0: 100.0, 1: 80.0, 2: 110.0},
        {0: 90.0, 1: 95.0, 2: 105.0},
        [],
        output,
        "split-ratio smoke",
        second_label="Throughput(split_ratio)",
    )

    assert output.is_file()
    with Image.open(output) as image:
        assert image.format == "PNG"


def test_repair_isolated_spike_uses_three_points_on_each_side():
    from plot_pcie_allgather import repair_isolated_spikes

    raw = {iteration: 140.0 for iteration in range(9)}
    raw[4] = 170.0

    repaired, replaced = repair_isolated_spikes(raw, deviation_threshold=26.0)

    assert replaced == [4]
    assert repaired[4] == pytest.approx(140.0)
    assert repaired[3] == raw[3]
    assert repaired[5] == raw[5]


def test_smooth_value_ranges_interpolates_concentrated_noise_blocks():
    from plot_pcie_allgather import smooth_value_ranges

    raw = {iteration: 140.0 for iteration in range(20)}
    raw.update({6: 100.0, 7: 105.0, 8: 110.0, 9: 115.0})

    repaired, replaced = smooth_value_ranges(raw, [(6, 10)])

    assert replaced == [6, 7, 8, 9]
    assert all(repaired[iteration] == pytest.approx(140.0) for iteration in replaced)
    assert repaired[5] == raw[5]
    assert repaired[10] == raw[10]


def test_rolling_window_stats_returns_local_mean_and_population_stddev():
    from plot_pcie_allgather import rolling_window_stats

    means, stddevs = rolling_window_stats(
        {iteration: float(iteration + 1) for iteration in range(5)},
        window=3,
    )

    assert means == {
        0: pytest.approx(1.5),
        1: pytest.approx(2.0),
        2: pytest.approx(3.0),
        3: pytest.approx(4.0),
        4: pytest.approx(4.5),
    }
    assert stddevs[0] == pytest.approx(0.5)
    assert stddevs[2] == pytest.approx((2.0 / 3.0) ** 0.5)


def test_comparison_plot_accepts_window_variance_envelopes(tmp_path):
    from PIL import Image

    from plot_pcie_allgather import plot_bandwidths

    output = tmp_path / "window-envelope.png"
    plot_bandwidths(
        {iteration: 100.0 + iteration for iteration in range(9)},
        {iteration: 140.0 - iteration for iteration in range(9)},
        [],
        output,
        "window envelope smoke",
        window_envelope=True,
        envelope_window=3,
        envelope_sigma=1.0,
    )

    assert output.is_file()
    with Image.open(output) as image:
        assert image.format == "PNG"


def test_comparison_plot_accepts_paper_scatter_options(tmp_path):
    from PIL import Image

    from plot_pcie_allgather import plot_bandwidths

    output = tmp_path / "paper-scatter.png"
    plot_bandwidths(
        {iteration: 100.0 for iteration in range(9)},
        {iteration: 140.0 for iteration in range(9)},
        [],
        output,
        "paper smoke",
        second_label="Throughput(split_ratio)",
        y_min=60.0,
        points_only=True,
        second_spike_threshold=26.0,
        paper=True,
    )

    assert output.is_file()
    with Image.open(output) as image:
        assert image.format == "PNG"


def test_comparison_plot_accepts_explicit_bandwidth_axis_limits(tmp_path):
    from PIL import Image

    from plot_pcie_allgather import plot_bandwidths

    output = tmp_path / "paper-line.png"
    plot_bandwidths(
        {iteration: 100.0 for iteration in range(9)},
        {iteration: 140.0 for iteration in range(9)},
        [],
        output,
        "paper line smoke",
        second_label="Throughput(split_ratio)",
        y_min=0.0,
        y_max=181.0,
        paper=True,
    )

    assert output.is_file()
    with Image.open(output) as image:
        assert image.format == "PNG"


def test_comparison_plot_accepts_custom_line_width(tmp_path):
    from PIL import Image

    from plot_pcie_allgather import plot_bandwidths

    output = tmp_path / "thin-line.png"
    plot_bandwidths(
        {iteration: 100.0 for iteration in range(9)},
        {iteration: 140.0 for iteration in range(9)},
        [],
        output,
        "thin line smoke",
        line_width=0.8,
        paper=True,
    )

    assert output.is_file()
    with Image.open(output) as image:
        assert image.format == "PNG"


def test_comparison_plot_accepts_reference_figure_style(tmp_path):
    from PIL import Image

    from plot_pcie_allgather import plot_bandwidths

    output = tmp_path / "reference-style.png"
    plot_bandwidths(
        {iteration: 100.0 for iteration in range(9)},
        {iteration: 140.0 for iteration in range(9)},
        [],
        output,
        "",
        reference_style=True,
        paper=True,
    )

    assert output.is_file()
    with Image.open(output) as image:
        assert image.format == "PNG"


def test_reference_style_uses_large_axis_and_legend_fonts():
    from plot_pcie_allgather import (
        reference_style_font_sizes,
        reference_style_y_label_position,
        reference_style_x_label_size,
    )

    title_size, label_size, tick_size, legend_size = reference_style_font_sizes()

    assert title_size >= 17
    assert label_size == 31
    assert tick_size == 23
    assert legend_size >= 28
    assert reference_style_x_label_size() == 33
    assert reference_style_y_label_position() == (-0.1, 0.5)


def test_reference_style_y_label_is_vertically_centered(tmp_path):
    from PIL import Image

    from plot_pcie_allgather import plot_bandwidths

    output = tmp_path / "reference-style-centered-y-label.png"
    plot_bandwidths(
        {iteration: 100.0 for iteration in range(9)},
        {iteration: 140.0 for iteration in range(9)},
        [],
        output,
        "",
        reference_style=True,
        paper=True,
    )

    with Image.open(output).convert("RGB") as image:
        dark_rows = [
            y for y in range(135, 606)
            if any(max(image.getpixel((x, y))) < 80 for x in range(8, 38))
        ]

    assert dark_rows
    assert abs((min(dark_rows) + max(dark_rows)) / 2 - 370) <= 15


def test_comparison_plot_accepts_a_broken_y_axis(tmp_path):
    from PIL import Image

    from plot_pcie_allgather import plot_bandwidths

    output = tmp_path / "broken-y-axis.png"
    plot_bandwidths(
        {iteration: 100.0 for iteration in range(9)},
        {iteration: 140.0 for iteration in range(9)},
        [],
        output,
        "",
        y_min=54.0,
        y_max=181.0,
        axis_break=True,
        reference_style=True,
        paper=True,
    )

    assert output.is_file()
    with Image.open(output) as image:
        assert image.format == "PNG"


def test_path_plot_writer_draws_three_bandwidth_series(tmp_path):
    from PIL import Image

    from plot_pcie_allgather import PathBandwidth, plot_path_bandwidths

    output = tmp_path / "paths.png"
    plot_path_bandwidths(
        {
            0: PathBandwidth(hccl=17.0, pcie=1.0, total=100.0),
            1: PathBandwidth(hccl=18.0, pcie=1.5, total=110.0),
            2: PathBandwidth(hccl=16.0, pcie=0.8, total=95.0),
        },
        output,
        "path smoke",
    )

    assert output.is_file()
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.width >= 800
        assert image.height >= 500


def test_path_plot_writer_draws_optional_pure_hccl_reference(tmp_path):
    from PIL import Image

    from plot_pcie_allgather import PathBandwidth, plot_path_bandwidths

    output = tmp_path / "paths_with_reference.png"
    plot_path_bandwidths(
        {
            0: PathBandwidth(hccl=90.0, pcie=10.0, total=100.0),
            1: PathBandwidth(hccl=95.0, pcie=5.0, total=100.0),
            2: PathBandwidth(hccl=88.0, pcie=12.0, total=100.0),
        },
        output,
        "path smoke with reference",
        reference={0: 110.0, 1: 112.0, 2: 108.0},
    )

    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.width >= 800
        assert image.height >= 500


def test_path_time_plot_writer_draws_two_path_series(tmp_path):
    from PIL import Image

    from plot_pcie_allgather import PathTiming, plot_path_timings

    output = tmp_path / "path_times.png"
    plot_path_timings(
        {
            0: PathTiming(hccl_ms=7.0, pcie_ms=1.0),
            1: PathTiming(hccl_ms=8.0, pcie_ms=1.5),
            2: PathTiming(hccl_ms=6.0, pcie_ms=0.8),
        },
        output,
        "path time smoke",
    )

    assert output.is_file()
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.width >= 800
        assert image.height >= 500


def test_hccl_only_plot_writers_draw_hccl_series(tmp_path):
    from PIL import Image

    from plot_pcie_allgather import (
        PathBandwidth,
        PathTiming,
        plot_hccl_bandwidths,
        plot_hccl_timings,
    )

    metrics = {
        0: PathBandwidth(hccl=17.0, pcie=0.0, total=17.0),
        1: PathBandwidth(hccl=18.0, pcie=0.0, total=18.0),
        2: PathBandwidth(hccl=16.0, pcie=0.0, total=16.0),
    }
    timings = {
        0: PathTiming(hccl_ms=7.0, pcie_ms=0.0),
        1: PathTiming(hccl_ms=8.0, pcie_ms=0.0),
        2: PathTiming(hccl_ms=6.0, pcie_ms=0.0),
    }
    bandwidth_output = tmp_path / "hccl_only.png"
    time_output = tmp_path / "hccl_only_times.png"
    plot_hccl_bandwidths(metrics, bandwidth_output, "HCCL only")
    plot_hccl_timings(timings, time_output, "HCCL only time")

    for output in (bandwidth_output, time_output):
        assert output.is_file()
        with Image.open(output) as image:
            assert image.format == "PNG"
            assert image.width >= 800
            assert image.height >= 500


def test_plot_tick_spacing_scales_for_full_iteration_count():
    from plot_pcie_allgather import choose_tick_step

    assert choose_tick_step(list(range(49))) == 10
    assert choose_tick_step(list(range(1000))) == 100


def test_bandwidth_plot_uses_more_horizontal_reference_lines():
    import plot_pcie_allgather as plotter

    assert hasattr(plotter, "bandwidth_grid_tick_count")
    assert plotter.bandwidth_grid_tick_count() >= 10


def test_comparison_plot_accepts_static_and_both_spike_thresholds(tmp_path):
    from PIL import Image

    from plot_pcie_allgather import plot_bandwidths

    output = tmp_path / "spike_repair.png"
    static_data = {iteration: 140.0 for iteration in range(9)}
    static_data[4] = 110.0
    tcp_data = {iteration: 150.0 for iteration in range(9)}
    tcp_data[4] = 120.0

    rep_static, rep_tcp = plot_bandwidths(
        static_data,
        tcp_data,
        [],
        output,
        "spike repair test",
        static_spike_threshold=20.0,
        second_spike_threshold=20.0,
    )

    assert rep_static[4] == pytest.approx(140.0)
    assert rep_tcp[4] == pytest.approx(150.0)
    assert output.is_file()
    with Image.open(output) as image:
        assert image.format == "PNG"
