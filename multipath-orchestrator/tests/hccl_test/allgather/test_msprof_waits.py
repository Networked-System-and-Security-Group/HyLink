from analyze_msprof_waits import summarize_wait_records


def test_wait_identity_uses_connection_not_host_device_clock_overlap():
    ags = [(100, 120, {'rank': '1', 'iter': '52', 'interference': '0'}, 7)]
    apis = [dict(ts=105, tid=7, name='AscendCL@aclrtValueWait', args={'connection_id': 3}),
            dict(ts=95, tid=7, name='AscendCL@aclrtValueWait', args={'connection_id': 4})]
    devices = [dict(ts='999.0', dur=1, tid=17, name='MEM_WAIT_VALUE', args={'connection_id': 3}),
               dict(ts='999.5', dur=1, tid=17, name='MEM_WAIT_VALUE', args={'connection_id': 3}),
               dict(ts=1000, dur=100, tid=17, name='MEM_WAIT_VALUE', args={'connection_id': 4})]
    rows = summarize_wait_records(apis, devices, ags)
    assert len(rows) == 1
    row = rows[0]
    assert (row['rank'], row['iter'], row['host_value_wait_count']) == (1, 52, 1)
    assert row['MEM_WAIT_VALUE_records'] == 2
    assert row['MEM_WAIT_VALUE_connections'] == 1
    assert row['MEM_WAIT_VALUE_sum_us'] == 2
    assert row['MEM_WAIT_VALUE_union_us'] == 1.5


def test_event_wait_is_counted_separately_from_value_wait():
    ags = [(100, 120, {'rank': '0', 'iter': '1', 'interference': '1'}, 7)]
    apis = [dict(ts=105, tid=7, name='AscendCL@aclrtStreamWaitEvent', args={'connection_id': 8})]
    devices = [dict(ts=110, dur=20, tid=17, name='EVENT_WAIT', args={'connection_id': 8})]
    row = summarize_wait_records(apis, devices, ags)[0]
    assert row['host_event_wait_count'] == 1
    assert row['host_value_wait_count'] == 0
    assert row['EVENT_WAIT_records'] == 1
    assert row['EVENT_WAIT_sum_us'] == 20
    assert row['MEM_WAIT_VALUE_records'] == 0
