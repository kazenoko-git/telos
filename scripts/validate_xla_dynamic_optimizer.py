"""Check AdamW parity and LR-driven compilation on a real XLA device.

Run with --device cpu for numerical checks only. CPU execution cannot validate
XLA graph stability. Results include compilation counts rather than inferring
recompilation from scheduler values or mocked graph hashes.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from telos.training.lr_schedule import WarmupCosineLR
from telos.training.xla_optimizer import XLAAdamW


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'xla'], default='xla')
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.steps < 8:
        parser.error('At least eight steps are required to check steady-state compilation')
    xm = metrics = None
    if args.device == 'xla':
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as metrics
        device = xm.xla_device()
    else:
        device = torch.device('cpu')
    torch.manual_seed(73)
    initial = torch.randn(64, 64)
    gradients = [torch.randn_like(initial) for _ in range(args.steps)]
    reference = torch.nn.Parameter(initial.clone())
    native = torch.optim.AdamW([reference], lr=.003, betas=(.9, .95))
    schedule = WarmupCosineLR(native, args.steps, args.steps + 10, update_cadence=25)
    parameter = torch.nn.Parameter(initial.to(device))
    candidate = XLAAdamW([parameter], lr=.003, betas=(.9, .95))
    actual_schedule = WarmupCosineLR(candidate, args.steps, args.steps + 10, update_cadence=25)
    counts = []
    if metrics:
        metrics.clear_all()
    for grad in gradients:
        reference.grad, parameter.grad = grad.clone(), grad.to(device)
        native.step(); schedule.step()
        candidate.step(); actual_schedule.step()
        assert schedule.get_last_lr() == actual_schedule.get_last_lr()
        if xm:
            xm.mark_step()
            xm.wait_device_ops()
            data = metrics.metric_data('CompileTime')
            counts.append(data[0] if data else 0)
    actual = parameter.detach().cpu()
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
    stable = counts[-1] == counts[3] if counts else None
    report = dict(device=str(device), steps=args.steps,
                  max_parameter_difference=(actual - reference).abs().max().item(),
                  compile_counts=counts, stable_after_four_updates=stable,
                  learning_rates_equal=True, torch_version=torch.__version__)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if stable is False:
        raise SystemExit('XLA compilation count increased after warmup')


if __name__ == '__main__':
    main()
