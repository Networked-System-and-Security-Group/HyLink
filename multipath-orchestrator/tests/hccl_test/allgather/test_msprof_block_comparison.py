import importlib.util
import json
from pathlib import Path


def test_comparison_keeps_background_size_and_capture_mode(tmp_path):
    path = Path(__file__).with_name('compare_msprof_blocks.py')
    assert path.exists(), 'missing profiler/block ablation comparison'
    spec = importlib.util.spec_from_file_location('block_comparison', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    (tmp_path/'metadata.json').write_text(json.dumps(dict(
        command=['app','--interference-size=16777216','--interference-ranges=1-1'],
        environment={'CCL_MSPROF_MARKERS':'0'})))
    log = tmp_path/'run.log'
    log.write_text('iter=0 event_ms=10 bw_GBps=100\niter=1 event_ms=20 bw_GBps=50\n')
    info, rows = module.read_case(log)
    assert info['interference_bytes'] == 16777216 and info['profiled'] == 0
    assert info['phases']['interference']['event_ms']['mean'] == 20
    assert rows[1]['interference'] == 1 and rows[0]['profiled'] == 0
