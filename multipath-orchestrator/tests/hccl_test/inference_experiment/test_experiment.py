import importlib.util
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def load_processing():
    spec = importlib.util.spec_from_file_location('experiment_processing', ROOT / 'process_results.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_passes_requested_fixed_configuration():
    result = subprocess.run(['bash', str(ROOT / 'run.sh'), '--dry-run'],
                            check=True, capture_output=True, text=True)
    for item in ('np=2', 'input_bytes=536870912', 'total_bytes=1073741824',
                 'interference_bytes=268435456', 'iters=1001', 'warmup=10',
                 'ranges=100-200,300-600,700-900', 'static_alpha=0.39',
                 'tcp_initial_alpha=0.075'):
        assert item in result.stdout


def test_parse_checks_completeness_and_uses_aggregate_gib_per_second(tmp_path):
    import pytest
    module = load_processing()
    log = tmp_path / 'test.log'
    log.write_text('iter=0 event_ms=10.000 bw_GBps=50.000\n'
                   'iter=1 event_ms=20.000 bw_GBps=25.000\n'
                   'rank 0: PASSED\nrank 1: PASSED\n')
    points = module.parse_log(log, iters=2)
    assert points == [(0, 10.0, 100.0), (1, 20.0, 50.0)]
    with pytest.raises(ValueError):
        module.parse_log(log, iters=3)


def test_envelope_keeps_spikes_and_does_not_cross_phase_boundaries():
    module = load_processing()
    points = [(98, 1, 100), (99, 1, 0), (100, 1, 20), (101, 1, 20)]
    stats = module.envelope(points, window=3)
    assert stats[1] == (99, 50.0, 50.0)
    assert stats[2] == (100, 20.0, 0.0)


def test_plot_labels_follow_each_log_configuration(tmp_path):
    module = load_processing()
    static = tmp_path / 'static.log'
    tcp = tmp_path / 'tcp.log'
    static.write_text('algorithm=static static_alpha=0.39 tcp_initial_alpha=0.075\n')
    tcp.write_text('algorithm=tcp static_alpha=0.075 tcp_initial_alpha=0.075\n')
    assert module.algorithm_label(static, 'static') == 'Static (alpha=0.39)'
    assert module.algorithm_label(tcp, 'tcp') == 'TCP (initial alpha=0.075)'


def test_both_plot_scripts_generate_images_from_complete_logs(tmp_path):
    import sys
    from PIL import Image
    for algorithm in ('tcp', 'static'):
        (tmp_path / f'{algorithm}.log').write_text(
            ''.join(f'iter={i} event_ms=20.000 bw_GBps=25.000\n' for i in range(1001))
            + 'rank 0: PASSED\nrank 1: PASSED\n')
    for script, output in [('plot_bandwidth.py', 'bandwidth_raw.png'),
                           ('plot_variance_envelope.py', 'bandwidth_variance_envelope.png')]:
        subprocess.run([sys.executable, str(ROOT / script), str(tmp_path)], check=True)
        with Image.open(tmp_path / output) as image:
            assert image.width >= 1000
            assert image.height >= 400
    import csv
    with (tmp_path / 'variance_envelope.csv').open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2002
    assert float(rows[0]['window_mean_GiBps']) == 50.0
    assert float(rows[0]['variance_GiBps2']) == 0.0
