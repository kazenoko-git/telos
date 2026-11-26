"""Host-wall timing for native SSD-backed generation."""

from time import perf_counter_ns


class SSDGenerationProfile:
    """Collect time spent at existing SSD and MLX synchronization points."""

    def __init__(self):
        self.steps = []

    def begin_step(self, phase, cache):
        self._phase = phase
        self._bytes_read = cache.bytes_read
        self._hits = cache.hits
        self._misses = cache.misses
        self._sync_read_ns = self._async_wait_ns = self._mlx_eval_ns = 0
        self._started_ns = perf_counter_ns()

    def read(self, function, *args, asynchronous):
        started = perf_counter_ns()
        result = function(*args)
        elapsed = perf_counter_ns() - started
        if asynchronous:
            self._async_wait_ns += elapsed
        else:
            self._sync_read_ns += elapsed
        return result

    def evaluate(self, function, *args):
        started = perf_counter_ns()
        result = function(*args)
        self._mlx_eval_ns += perf_counter_ns() - started
        return result

    def end_step(self, cache):
        wall_ns = perf_counter_ns() - self._started_ns
        ssd_wait_ns = self._sync_read_ns + self._async_wait_ns
        self.steps.append({
            "phase": self._phase,
            "wall_seconds": wall_ns / 1e9,
            "ssd_wait_seconds": ssd_wait_ns / 1e9,
            "sync_read_seconds": self._sync_read_ns / 1e9,
            "async_wait_seconds": self._async_wait_ns / 1e9,
            "mlx_eval_wall_seconds": self._mlx_eval_ns / 1e9,
            "other_seconds": (wall_ns - ssd_wait_ns - self._mlx_eval_ns) / 1e9,
            "expert_bytes_read": cache.bytes_read - self._bytes_read,
            "expert_hits": cache.hits - self._hits,
            "expert_misses": cache.misses - self._misses,
        })

    def report(self):
        phases = {}
        for phase in ("prefill", "decode"):
            steps = [step for step in self.steps if step["phase"] == phase]
            phases[phase] = {"steps": len(steps)}
            if steps:
                phases[phase].update({key: sum(step[key] for step in steps) for key in steps[0] if key != "phase"})
        return {"phases": phases, "steps": self.steps}
