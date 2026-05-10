from analyze_msprof_runtime import runtime_metrics, collect_runtime
import analyze_msprof_runtime as analysis


def test_union_clips_nested_calls_and_exposes_outside_gaps():
    result = runtime_metrics(0, 100, [
        ('aclrtMemcpyAsync', 10, 40), ('aclrtRecordEvent', 20, 10),
        ('aclrtQueryEventStatus', 90, 30)])
    assert result['api_sum_us'] == 80
    assert result['api_union_us'] == 50
    assert result['outside_api_gap_max_us'] == 40
    assert result['aclrtQueryEventStatus_max_us'] == 30


def test_only_executor_ascendcl_layer_matches_not_background_or_nested_runtime():
    window = dict(start=0, end=100, tid=7, events=[])
    events = [dict(ph='X', name=name, ts=10, dur=20, tid=tid, pid=999)
              for name, tid in [('AscendCL@aclrtMemcpyAsync', 7),
                                ('Runtime@aclrtMemcpyAsync', 7),
                                ('AscendCL@aclrtMemcpyAsync', 8)]]
    collect_runtime(events, [window])
    assert window['events'] == [('aclrtMemcpyAsync', 10., 20.)]


def test_epoch_timestamps_do_not_round_microsecond_api_durations():
    window = dict(start=0, end=100, tid=7, events=[])
    collect_runtime([dict(ph='X', name='AscendCL@aclrtMemcpyAsync',
                          ts='1791171654498020.810', dur=1.13, tid=7)],
                    [window], origin='1791171654498020.710')
    result = runtime_metrics(0, 100, window['events'])
    assert abs(result['api_union_us']-1.13) < 1e-9


def test_nested_runtime_is_union_not_sum_and_cross_boundary_is_unmatched():
    result = analysis.nested_metrics(
        [('aclrtRecordEvent', 0, 100), ('aclrtQueryEventStatus', 100, 20)],
        [('EventRecord', 10, 50), ('nested', 20, 10), ('cross', 95, 10)])
    assert result[0]['runtime_union_us'] == 50
    assert result[0]['acl_outside_runtime_us'] == 50
    assert result[0]['runtime_count'] == 2
    assert result[1]['runtime_count'] == 0
    assert result[0]['unmatched_runtime_count'] == 1


def test_runtime_collection_keeps_layers_separate_on_executor_thread():
    window = dict(start=0, end=100, tid=7, events=[], inner=[])
    collect_runtime([dict(ph='X', name='Runtime@EventRecord', ts=10, dur=5, tid=7),
                     dict(ph='X', name='Runtime@EventRecord', ts=10, dur=5, tid=8)],
                    [window], include_nested=True)
    assert window['events'] == []
    assert window['inner'] == [('EventRecord', 10., 5.)]
