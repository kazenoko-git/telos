"""Completed-update timing. Synchronization is restricted to benchmark mode."""
import time


class BenchmarkTimer:
    """Exclude warmup; include host input, device work and rank synchronization.

    Fencing each measured update intentionally measures synchronized update
    latency. Aggregate elapsed also includes reporting between updates. This
    is distinct from an unconstrained asynchronous throughput experiment.
    """
    def __init__(self, synchronize, maximum=lambda value: value, warmup=5, clock=time.perf_counter):
        self.synchronize, self.maximum = synchronize, maximum
        self.warmup, self.clock = warmup, clock
        self.updates = self.steps = 0
        self.latencies = []
        self.started = self.update_started = None
        self.elapsed = 0.0

    def begin(self):
        self.synchronize()
        self.update_started = self.clock()

    def end(self):
        self.synchronize()
        now = self.clock()
        self.updates += 1
        if self.updates <= self.warmup:
            # All workers complete warmup before starting the aggregate clock.
            self.maximum(0.0)
            self.started = self.clock()
            return
        self.steps += 1
        self.latencies.append(self.maximum((now - self.update_started) * 1000.0))
        self.elapsed = self.maximum(self.clock() - self.started)


def pytorch_benchmark_timer(trainer):
    import torch
    if trainer.is_tpu:
        import torch_xla.core.xla_model as xm
        def synchronize():
            xm.mark_step()
            xm.wait_device_ops()
        def maximum(value):
            return xm.mesh_reduce("telos_benchmark_max", value, max)
    elif getattr(trainer, "is_ddp", False):
        def synchronize():
            torch.cuda.synchronize(trainer.device)
        def maximum(value):
            elapsed = torch.tensor(value, dtype=torch.float64, device=trainer.device)
            torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
            return elapsed.item()
    else:
        def synchronize():
            if trainer.device.type == "cuda":
                for index in getattr(trainer.model, "device_ids", [trainer.device]):
                    torch.cuda.synchronize(index)
        def maximum(value):
            return value
    return BenchmarkTimer(synchronize, maximum)
