from analyze_msprof_host_phases import extract_host_phases, barrier_observations


def event(name, ts, dur=1, tid=7):
    return dict(name=name, ph='X', ts=str(ts), dur=dur, pid=100, tid=tid)


def test_release_outside_ag_retains_identity_and_cpu_uses_payload():
    events = [event('AG|rank=1|iter=9|phase=formal|interference=1', 100, 100),
              event('PCCL.loop|rank=1|comm=17|seq=99', 110, 80),
              event('PCCL.host.release|rank=1|comm=17|seq=99', 220, 800),
              event('PCCL.host_cpu|phase=release|rank=1|comm=17|seq=99|begin_ns=1000000|wall_ns=800000|cpu_ns=10000|vcs=0|ivcs=2|ok=1', 1030),
              event('PCCL.host_cpu|phase=release|rank=1|comm=18|seq=99|begin_ns=1000000|wall_ns=800000|cpu_ns=10000|vcs=0|ivcs=2|ok=1', 1030)]
    rows, points, unmatched = extract_host_phases(events)
    assert unmatched == 1 and points == [] and len(rows) == 1
    assert rows[0]['iter'] == 9 and rows[0]['phase'] == 'release'
    assert rows[0]['wall_us'] == 800 and rows[0]['cpu_us'] == 10
    assert rows[0]['nonrunning_us'] == 790 and rows[0]['involuntary'] == 2
    assert rows[0]['marker_duration_us'] == 800


def test_barrier_separates_arrival_skew_from_post_publication_resume_delay():
    rows = [dict(rank=0, comm='17', seq='99', iter=9, phase='barrier3_arrive', begin_ns=1000, end_ns=2000),
            dict(rank=1, comm='17', seq='99', iter=9, phase='barrier3_arrive', begin_ns=10000, end_ns=11000),
            dict(rank=1, comm='17', seq='99', iter=9, phase='barrier3_publish', begin_ns=12000, end_ns=13000),
            dict(rank=0, comm='17', seq='99', iter=9, phase='barrier3_wait', begin_ns=3000, end_ns=25000)]
    out = barrier_observations(rows)
    assert len(out) == 2
    r0 = next(r for r in out if r['rank'] == 0)
    assert r0['arrival_spread_us'] == 9
    assert r0['resume_delay_lower_us'] == 12 and r0['resume_delay_upper_us'] == 13
    assert r0['last_arrival_rank'] == 1


def test_invalid_samples_and_warmup_are_not_used_as_barrier_evidence():
    events = [event('AG|rank=1|iter=0|phase=warmup', 0, 100),
              event('PCCL.loop|rank=1|comm=17|seq=1', 10, 80),
              event('PCCL.host_cpu|phase=prepare|rank=1|comm=17|seq=1|begin_ns=1|wall_ns=2|cpu_ns=1|vcs=0|ivcs=0|ok=1', 20),
              event('AG|rank=1|iter=1|phase=formal', 100, 100),
              event('PCCL.loop|rank=1|comm=17|seq=2', 110, 80),
              event('PCCL.host_cpu|phase=prepare|rank=1|comm=17|seq=2|begin_ns=1|wall_ns=2|cpu_ns=1|vcs=0|ivcs=0|ok=0', 120)]
    rows, _, _ = extract_host_phases(events)
    assert rows == []


def test_last_arrival_is_publisher_not_thread_with_late_clock_sample():
    rows = [dict(rank=0, comm='17', seq='99', iter=9, phase='barrier3_arrive', begin_ns=1000, end_ns=25000),
            dict(rank=1, comm='17', seq='99', iter=9, phase='barrier3_arrive', begin_ns=10000, end_ns=11000),
            dict(rank=1, comm='17', seq='99', iter=9, phase='barrier3_publish', begin_ns=12000, end_ns=13000)]
    out = barrier_observations(rows)
    assert out[0]['last_arrival_rank'] == 1
    assert out[0]['latest_arrival_sample_rank'] == 0


def test_host_ranges_keep_publication_bounds_when_cpu_metadata_is_missing():
    from analyze_msprof_host_phases import extract_host_ranges
    events = [event('AG|rank=1|iter=9|phase=formal|interference=1', 100, 100),
              event('PCCL.loop|rank=1|comm=17|seq=99', 110, 80),
              event('PCCL.host.barrier3_publish|rank=1|comm=17|seq=99', '120.25', 2)]
    rows = extract_host_ranges(events)
    assert len(rows) == 1 and rows[0]['iter'] == 9
    assert rows[0]['begin_ns'] == 120250 and rows[0]['end_ns'] == 122250
    assert rows[0]['cpu_us'] == '' and rows[0]['cpu_sample_present'] == 0


def test_compact_cpu_payload_preserves_full_identity_and_measurements():
    events = [event('AG|rank=1|iter=9|phase=formal|interference=1', 100, 100),
              event('PCCL.loop|rank=1|comm=17|seq=99', 110, 80),
              event('PCCL.host.prepare|rank=1|comm=17|seq=99', 105, 5),
              event('PCCL.host_cpu|p=prepare|r=1|g=17|s=99|b=1000000|w=10000|c=1000|v=0|i=1|k=1', 111)]
    rows, _, unmatched = extract_host_phases(events)
    assert unmatched == 0 and len(rows) == 1
    assert rows[0]['seq'] == '99' and rows[0]['comm'] == '17'
    assert rows[0]['wall_us'] == 10 and rows[0]['cpu_us'] == 1
