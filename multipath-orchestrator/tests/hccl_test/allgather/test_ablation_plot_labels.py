from plot_cpu_sampler_ablation import classify_case


def test_completion_labels_are_binary_identity_not_sampling_flags():
    report = {'case': '/logs/completion_fix_rep2_20261005/static_msprof', 'sample_period': 0}
    assert classify_case(report, 'completion') == (2, 'fix')
    report['case'] = '/logs/completion_old_rep1_20261005/static_msprof'
    assert classify_case(report, 'completion') == (1, 'old')


def test_cpu_labels_still_use_sampling_flag():
    report = {'case': '/logs/cpu_probe16_on_rep3_20261005/static_msprof', 'sample_period': 16}
    assert classify_case(report, 'cpu') == (3, 'on')


def test_host_labels_do_not_mislabel_enabled_low_frequency_timing_as_off():
    report = {'case': '/logs/host_timing_on_rep2_20261005/static_msprof', 'sample_period': 0}
    assert classify_case(report, 'host') == (2, 'on')
    report['case'] = '/logs/host_timing_off_rep1_20261005/static_msprof'
    assert classify_case(report, 'host') == (1, 'off')


def test_wait_labels_use_notification_mode_not_cpu_sampling():
    report = {'case': '/logs/completion_wait_event_rep2_20261005/static_baseline', 'sample_period': 0}
    assert classify_case(report, 'wait') == (2, 'event')
    report['case'] = '/logs/completion_wait_value_rep1_20261005/static_baseline'
    assert classify_case(report, 'wait') == (1, 'value')


def test_slice_labels_use_build_identity_for_baseline_runs():
    report = {'case': '/logs/subchunk_4m_rep2_20261005/static_baseline', 'sample_period': 0}
    assert classify_case(report, 'slice') == (2, '4m')
    report['case'] = '/logs/subchunk_1m_rep1_20261005/static_baseline'
    assert classify_case(report, 'slice') == (1, '1m')


def test_path_plot_keeps_case_identity_and_omits_missing_final_feedback(tmp_path):
    from plot_cpu_sampler_ablation import read_series
    (tmp_path / 'subchunk_iterations.csv').write_text(
        'source,iter,pcie_ms\n/logs/subchunk_1m_rep1_20261005/static_baseline/run.log,0,7.5\n'
        '/logs/subchunk_1m_rep1_20261005/static_baseline/run.log,1,\n'
        '/logs/subchunk_4m_rep1_20261005/static_baseline/run.log,0,4.0\n')
    report = {'case': '/logs/subchunk_1m_rep1_20261005/static_baseline'}
    assert read_series(report, 'pcie_ms', tmp_path) == [{'iter': 0, 'pcie_ms': 7.5}]
