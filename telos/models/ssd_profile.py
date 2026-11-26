"""Host-wall timing for native SSD-backed generation."""

from time import perf_counter_ns


class SSDGenerationProfile:
    """Collect time spent at existing SSD and MLX synchronization points."""

    _SECTIONS = ("backbone_router", "experts", "tail", "sampling")
    _ATTENTION_SECTIONS = ("attention", "routing", "residual_mix", "experts", "tail", "sampling")
    _ATTENTION_COMPONENTS = ("attention_projections", "attention_convolutions", "attention_setup", "attention_sdpa")

    def __init__(self, *, split_decode=False, split_attention=False, attention_stride=1, attention_components=False):
        if type(attention_stride) is not int or attention_stride < 1:
            raise ValueError("Attention profiling stride must be a positive integer")
        self.steps = []
        self._split_decode = split_decode or split_attention
        self._split_attention = split_attention
        self._attention_stride = attention_stride
        self._sections = self._ATTENTION_SECTIONS if split_attention else self._SECTIONS
        self.attention_components = attention_components
        if attention_components:
            self._sections = (*self._sections, *self._ATTENTION_COMPONENTS)
        self.split_step = False
        self.split_attention_step = False

    def begin_step(self, phase, cache):
        self._phase = phase
        self.split_step = self._split_decode and phase == "decode"
        self.split_attention_step = self.split_step and self._split_attention
        self._section_ns = dict.fromkeys(self._sections, 0) if self.split_step else None
        self._attention_visits = 0
        self.route_materializations = 0
        self.batched_expert_visits = 0
        self.compiled_layer_visits = 0
        self._bytes_read = cache.bytes_read
        self._hits = cache.hits
        self._misses = cache.misses
        self._sync_read_ns = self._async_wait_ns = self._mlx_eval_ns = 0
        self._started_ns = perf_counter_ns()

    def begin_visit(self, visit):
        if self._split_attention and self._phase == "decode":
            self.split_attention_step = (visit + len(self.steps)) % self._attention_stride == 0
            self.split_step = self.split_attention_step
            self._attention_visits += self.split_attention_step

    def read(self, function, *args, asynchronous):
        started = perf_counter_ns()
        result = function(*args)
        elapsed = perf_counter_ns() - started
        if asynchronous:
            self._async_wait_ns += elapsed
        else:
            self._sync_read_ns += elapsed
        return result

    def evaluate(self, function, *args, section=None):
        started = perf_counter_ns()
        result = function(*args)
        elapsed = perf_counter_ns() - started
        self._mlx_eval_ns += elapsed
        if self.split_step and section is not None:
            self._section_ns[section] += elapsed
        return result

    def end_step(self, cache):
        wall_ns = perf_counter_ns() - self._started_ns
        ssd_wait_ns = self._sync_read_ns + self._async_wait_ns
        step = {
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
            "route_materializations": self.route_materializations,
            "batched_expert_visits": self.batched_expert_visits,
            "compiled_layer_visits": self.compiled_layer_visits,
        }
        if self._split_decode:
            step.update({f"{name}_eval_seconds": ns / 1e9
                         for name, ns in (self._section_ns or dict.fromkeys(self._sections, 0)).items()})
        if self._split_attention:
            step["profiled_attention_visits"] = self._attention_visits
        self.steps.append(step)

    def report(self):
        phases = {}
        for phase in ("prefill", "decode"):
            steps = [step for step in self.steps if step["phase"] == phase]
            phases[phase] = {"steps": len(steps)}
            if steps:
                phases[phase].update({key: sum(step[key] for step in steps) for key in steps[0] if key != "phase"})
        return {"phases": phases, "steps": self.steps}
