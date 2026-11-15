from telos.training.benchmark_timing import BenchmarkTimer


def test_completed_work_and_warmup_use_monotonic_clock():
    now = [0.0]
    def sync():
        now[0] += 0.25  # Simulated work still queued when host call returns.
    timer = BenchmarkTimer(sync, warmup=2, clock=lambda: now[0])
    for _ in range(4):
        timer.begin()
        now[0] += 0.1
        timer.end()
    assert timer.steps == 2
    assert all(abs(value - 350) < 1e-8 for value in timer.latencies)
    assert abs(timer.elapsed - 1.2) < 1e-8


def test_slowest_rank_controls_stop_and_latency():
    now = [0.0]
    timer = BenchmarkTimer(lambda: None, maximum=lambda value: value * 2,
                           warmup=1, clock=lambda: now[0])
    timer.begin(); now[0] = 1; timer.end()
    timer.begin(); now[0] = 2; timer.end()
    assert timer.steps == 1
    assert timer.elapsed == 2
    assert timer.latencies == [2000]


def test_short_run_has_no_measured_samples():
    timer = BenchmarkTimer(lambda: None)
    timer.begin(); timer.end()
    assert timer.steps == 0
    assert timer.elapsed == 0
    assert timer.latencies == []
