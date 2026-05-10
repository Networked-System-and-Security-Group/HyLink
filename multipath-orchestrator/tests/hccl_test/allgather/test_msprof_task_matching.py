import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).with_name('audit_msprof_evidence.py')
    assert path.exists(), 'missing task-level evidence audit'
    spec = importlib.util.spec_from_file_location('audit_msprof_evidence', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_unique_fifo_match_separates_queue_service_and_notice():
    a = module()
    host = [dict(lane=2, start=100, end=1000, bytes=1024, operation='device to host'),
            dict(lane=2, start=200, end=1200, bytes=2048, operation='device to host')]
    device = [dict(stream=19, start=800, end=900, bytes=1024, operation='device to host'),
              dict(stream=19, start=1000, end=1100, bytes=2048, operation='device to host')]
    rows, warnings = a.match_lane_tasks(host, device, tolerance=0)
    assert not warnings and len(rows) == 2
    assert rows[0]['submit_to_device_start_us'] == 700
    assert rows[0]['device_execution_us'] == 100
    assert rows[0]['device_end_to_observed_us'] == 100


def test_ambiguous_stream_mapping_is_not_presented_as_causal_match():
    a = module()
    host = [dict(lane=i, start=100, end=1000, bytes=1024, operation='host to device') for i in (12, 13)]
    device = [dict(stream=i, start=200, end=300, bytes=1024, operation='host to device') for i in (20, 21)]
    rows, warnings = a.match_lane_tasks(host, device, tolerance=0)
    assert rows == [] and warnings


def test_local_clock_anchor_shifts_both_ends_without_changing_latency():
    a = module()
    host = [dict(lane=2, start=100, end=1000, bytes=1024, operation='device to host')]
    aligned = a.align_host_tasks(host, -65)
    assert aligned[0]['start'] == 35 and aligned[0]['end'] == 935
    assert host[0]['start'] == 100
