import importlib.util
from pathlib import Path


def test_overlapping_streams_are_not_summed_and_idle_gaps_are_preserved():
    path = Path(__file__).with_name('summarize_msprof_connections.py')
    assert path.exists(), 'missing msprof-only connection summary'
    spec = importlib.util.spec_from_file_location('connections_summary', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [dict(connection_id='1', api_us=5, device_start_us=100, device_execution_us=20),
            dict(connection_id='2', api_us=7, device_start_us=110, device_execution_us=20),
            dict(connection_id='3', api_us=9, device_start_us=150, device_execution_us=10)]
    result = module.metrics(rows)
    assert result['hardware_span_us'] == 60
    assert result['foreground_busy_union_us'] == 40
    assert result['foreground_gap_max_us'] == 20
    assert result['copy_api_sum_us'] == 21
    covered = module.metrics(rows, [(135,155)])
    assert covered['foreground_max_gap_background_overlap_us'] == 15
