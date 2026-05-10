from analyze_msprof_cpu_samples import extract_cpu_samples
import analyze_msprof_cpu_samples as analysis


def event(name, start, duration):
    return dict(name=name, ph='X', ts=str(start), dur=duration, pid=100, tid=7)


def test_sample_identity_uses_comm_seq_and_cpu_payload_not_marker_duration():
    events = [event('AG|rank=1|iter=9|phase=formal|interference=1', 0, 1000),
              event('PCCL.loop|rank=1|comm=15|seq=99', 10, 900),
              event('PCCL.api_cpu|rank=1|comm=15|seq=99|api=query|node=2|sc=3|lane=4|wall_ns=800000|cpu_ns=10000|nv=0|ni=1|valid=1', 850, 1),
              event('PCCL.api_cpu|rank=1|comm=16|seq=99|api=query|wall_ns=900000|cpu_ns=1|nv=0|ni=1|valid=1', 851, 1)]
    rows, unmatched = extract_cpu_samples(events)
    assert unmatched == 1
    assert len(rows) == 1
    assert rows[0]['iter'] == 9 and rows[0]['wall_us'] == 800
    assert rows[0]['cpu_us'] == 10 and rows[0]['offcpu_estimate_us'] == 790
    assert rows[0]['involuntary'] == 1


def test_missing_loop_and_invalid_clock_are_not_fabricated():
    rows, unmatched = extract_cpu_samples([event('PCCL.api_cpu|rank=1|comm=15|seq=99|api=query|wall_ns=0|cpu_ns=0|nv=0|ni=0|valid=0', 1, 1)])
    assert rows == [] and unmatched == 1


def test_sampling_overhead_is_checked_against_actual_cann_api():
    row = dict(tid=7, api='query', sample_end_us=100, wall_us=9)
    apis = [dict(ph='X', tid=7, ts='90', dur=8, name='AscendCL@aclrtQueryEventStatus'),
            dict(ph='X', tid=8, ts='95', dur=4, name='AscendCL@aclrtQueryEventStatus')]
    analysis.link_acl_samples([row], apis)
    assert row['acl_us'] == 8
    assert row['measurement_minus_acl_us'] == 1
    assert row['acl_end_to_marker_us'] == 2


def test_stage_duration_is_not_clipped_to_ag_and_does_not_require_cpu_samples():
    rows = analysis.extract_host_stages([
        event('AG|rank=1|iter=9|phase=formal|interference=1', 0, 100),
        event('PCCL.loop|rank=1|comm=15|seq=99', 10, 80),
        event('PCCL.batch|rank=1|comm=15|seq=99', 5, 110)])
    batch = next(r for r in rows if r['stage'] == 'batch')
    assert batch['iter'] == 9 and batch['duration_us'] == 110
