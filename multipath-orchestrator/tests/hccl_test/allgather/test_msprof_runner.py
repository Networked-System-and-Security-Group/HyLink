import importlib.util
from pathlib import Path


def test_background_size_is_independent_of_collective_input():
    path = Path(__file__).with_name('run_msprof.py')
    spec = importlib.util.spec_from_file_location('msprof_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    command = module.benchmark_options(80,5,'20-39,55-64',16777216)
    assert '--interference-size=16777216' in command
    assert '--send-count=67108864' in command
    assert '--iters=80' in command


def test_unprofiled_ablation_does_not_launch_profiler():
    path = Path(__file__).with_name('run_msprof.py')
    spec = importlib.util.spec_from_file_location('msprof_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.run_modes(False, True) == [False]
    assert module.run_modes(True, False) == [True]
    assert module.run_modes(False, False) == [False,True]


def test_h2h_shard_size_can_be_pinned_across_slice_builds():
    import pytest
    from run_msprof import h2h_shard_settings
    assert h2h_shard_settings(131072) == {'PCCL_H2H_SHARD_BYTES': '131072'}
    assert h2h_shard_settings(0) == {}
    with pytest.raises(ValueError):
        h2h_shard_settings(-1)


def test_completion_notification_mode_is_explicit_and_rejects_typos():
    import pytest
    from run_msprof import completion_wait_settings
    assert completion_wait_settings('value') == {'PCCL_COMPLETION_WAIT_MODE': 'value'}
    assert completion_wait_settings('event') == {'PCCL_COMPLETION_WAIT_MODE': 'event'}
    with pytest.raises(ValueError):
        completion_wait_settings('evnet')


def test_removed_in_code_sampling_options_are_not_advertised():
    source = Path(__file__).with_name('run_msprof.py').read_text()
    for removed in ('--host-timing', '--cpu-sample-period', 'CCL_MSPROF_MARKERS',
                    'PCCL_MSPROF_CPU_SAMPLE_PERIOD', 'PCCL_MSPROF_HOST_TIMING',
                    'PCCL_DIAGNOSTIC_DMA_EVENTS', 'PCCL_DEBUG_TIMING'):
        assert removed not in source
