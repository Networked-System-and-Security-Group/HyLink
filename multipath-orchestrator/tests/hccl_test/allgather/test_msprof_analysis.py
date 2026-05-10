import importlib.util
import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).with_name('analyze_msprof.py')


def load_module():
    assert SCRIPT.exists(), 'missing batch msprof analyzer'
    spec = importlib.util.spec_from_file_location('analyze_msprof', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event(name, ts, dur, pid=9, tid=1, **args):
    return dict(name=name, ts=ts, dur=dur, pid=pid, tid=tid, ph='X', args=args)


def test_clips_concurrent_dma_and_excludes_other_devices(tmp_path):
    module = load_module()
    events = [
        dict(name='process_name', ph='M', pid=90, args={'name': 'Ascend Hardware device 0'}),
        dict(name='process_name', ph='M', pid=91, args={'name': 'Ascend Hardware device 1'}),
        event('AG|rank=0|device=0|iter=2|phase=formal|interference=1', 100, 100),
        event('PCCL.loop|rank=0|comm=7|seq=8', 110, 80),
        event('MemcopyAsync', 90, 30, pid=90, tid=3, operation='device to host', **{'size(B)': 1048576}),
        event('MemcopyAsync', 110, 30, pid=90, tid=4, operation='host to device', **{'size(B)': 1048576}),
        event('MemcopyAsync', 180, 40, pid=90, tid=5, operation='device to host', **{'size(B)': 134217728}),
        event('MemcopyAsync', 100, 100, pid=91, tid=3, operation='device to host', **{'size(B)': 1048576}),
        event('AscendCL@aclrtMemcpyAsync', 120, 6, pid=92, tid=1),
        event('AG.submit|rank=0|device=0|iter=2', 100, 25),
        event('AMP.pcie_submit|rank=0|comm=123', 105, 15),
    ]
    path = tmp_path / 'msprof_1.json'
    path.write_text(json.dumps(events))
    result = module.analyze_file(path)
    assert len(result['iterations']) == 1
    row = result['iterations'][0]
    assert row['rank'] == 0 and row['iter'] == 2 and row['pccl_loop_us'] == 80
    assert row['dma_busy_us'] == 60  # union [100,140] + [180,200]
    assert row['foreground_dma_count'] == 2 and row['background_dma_count'] == 1
    assert row['foreground_dma_busy_us'] == 40
    assert row['foreground_copy_api_us'] == 6
    assert len(result['device_tasks']) == 3
    assert row['ag_submit_us'] == 25 and row['amp_pcie_submit_us'] == 15

    # CANN splits one 128MiB background copy into two 64MiB hardware tasks.
    events[6]['args']['size(B)'] = 67108864
    path.write_text(json.dumps(events))
    assert module.analyze_file(path)['iterations'][0]['background_dma_count'] == 1


def test_dict_format_missing_markers_and_duplicate_inputs(tmp_path):
    module = load_module()
    path = tmp_path / 'msprof_2.json'
    path.write_text(json.dumps({'traceEvents': [event('aclrtMemcpyAsync', 10, 8)]}))
    assert module.analyze_file(path)['iterations'] == []
    assert module.analyze_file(path)['warnings']
    assert module.discover([tmp_path, path]) == [path]


def test_streams_large_strings_and_ignores_tx_duplicate(tmp_path):
    module = load_module()
    path = tmp_path / 'msprof_3.json'
    events = [event('unrelated' + 'x' * 1200000, 1, 2), event('AG|rank=0|device=0|iter=0|phase=formal', 4, 8)]
    path.write_text(json.dumps({'traceEvents': events, 'other': 'unused'}))
    assert list(module.iter_events(path)) == events
    (tmp_path/'msprof_tx_3.json').write_text('[]')
    assert module.discover([tmp_path]) == [path]


def test_log_uses_max_rank_event_and_common_epoch_skew(tmp_path):
    module = load_module()
    case = tmp_path/'tcp_baseline'
    case.mkdir()
    (case/'metadata.json').write_text(json.dumps({'command':['app', '--interference-ranges=1-1']}))
    path = case/'run.log'
    path.write_text('iter=0 event_ms=10.000000 bw_GBps=107.374182\n'
                    'iter=1 event_ms=20.000000 bw_GBps=53.687091\n'
                    'collective device=0 iter=0 start_ms=100.000000 end_ms=111.000000 event_ms=10.000000\n'
                    'collective device=1 iter=0 start_ms=104.350000 end_ms=111.000000 event_ms=6.000000\n')
    rows, ranks = module.parse_benchmark_log(path)
    assert rows[0]['start_skew_ms'] == 4.35
    assert rows[0]['event_ms'] == 10 and rows[0]['interference'] == 0
    assert rows[1]['interference'] == 1
    assert len(ranks) == 2


def test_msprof_connection_join_keeps_tasks_outside_host_window(tmp_path):
    module = load_module()
    events = [dict(name='process_name', ph='M', pid=90, args={'name':'Ascend Hardware'}),
              event('AG|rank=0|device=0|iter=2|phase=formal', 100, 100),
              event('PCCL.loop|rank=0|comm=7|seq=8', 110, 80, tid=4),
              event('AscendCL@aclrtMemcpyAsync', 120, 6, pid=92, tid=4, connection_id=77),
              event('MEMCPY_ASYNC', 1000, 30, pid=90, tid=5,
                    connection_id=77, operation='device to host', **{'size(B)':1048576})]
    rows, warnings = module.correlate_msprof_tasks(events)
    assert not warnings and len(rows) == 1
    row = rows[0]
    assert row['iter'] == 2 and row['comm'] == '7' and row['seq'] == '8'
    assert row['api_us'] == 6 and row['device_execution_us'] == 30
    assert row['api_end_to_device_start_us'] == 874
    assert row['device_end_minus_loop_end_us'] == 840
    path = tmp_path/'msprof.json'
    path.write_text(json.dumps(events))
    aggregate = module.analyze_file(path)['iterations'][0]
    assert aggregate['foreground_connected_count'] == 1
    assert aggregate['foreground_device_span_us'] == 30


def test_msprof_connection_join_rejects_duplicate_api_identity():
    module = load_module()
    events = [dict(name='process_name', ph='M', pid=90, args={'name':'Ascend Hardware device 0'}),
              event('AG|rank=0|device=0|iter=2|phase=formal',100,100),
              event('PCCL.loop|rank=0|comm=7|seq=8',110,80,tid=4),
              event('AscendCL@aclrtMemcpyAsync',120,6,tid=4,connection_id=77),
              event('AscendCL@aclrtMemcpyAsync',130,6,tid=4,connection_id=77),
              event('MEMCPY_ASYNC',150,10,pid=90,connection_id=77,
                    operation='device to host', **{'size(B)':1048576})]
    rows, warnings = module.correlate_msprof_tasks(events)
    assert rows == [] and warnings


def test_msprof_rank_does_not_enable_inherited_legacy_trace():
    env = dict(os.environ, PMI_RANK='0', CCL_PROFILE_APP='/usr/bin/env', CCL_TRACE_DIR='/unused',
               PCCL_TRACE_FILE='/unused/legacy.json')
    result = subprocess.run(['bash', str(SCRIPT.with_name('msprof_rank.sh'))],
                            env=env, capture_output=True, text=True, check=True)
    assert 'PCCL_TRACE_FILE=' not in result.stdout
    assert 'CCL_TRACE_DIR=' not in result.stdout


def test_background_classification_uses_connection_not_fixed_size(tmp_path):
    module = load_module()
    events = [dict(name='process_name',ph='M',pid=90,args={'name':'Ascend Hardware device 0'}),
              event('AG|rank=0|device=0|iter=0|phase=formal|interference=1',100,100),
              event('BG.D2H|rank=0',110,50,tid=7),
              event('AscendCL@aclrtMemcpyAsync',120,6,pid=92,tid=7,connection_id=99),
              event('MEMCPY_ASYNC',130,20,pid=90,connection_id=99,
                    operation='device to host', **{'size(B)':16777216})]
    path = tmp_path/'msprof.json'; path.write_text(json.dumps(events))
    result = module.analyze_file(path)
    assert result['iterations'][0]['background_dma_count'] == 1
    assert result['device_tasks'][0]['kind'] == 'background_dma'
