"""AdamW with changing update scalars supplied as device data.

Native AdamW versions may extract tensor learning rates back to Python. Keep
its checkpoint layout and host scheduler, but transfer decay/bias-correction
controls as one vector, avoiding changing Python constants in the XLA graph.
"""
import math
import torch


class XLAAdamW(torch.optim.AdamW):
    """Single-tensor AdamW for real parameters; no host reads of device state.

    The update has the same mathematics as native AdamW. Moving scalar
    multiplication out of addcdiv can change floating-point rounding.
    When `fsdp_sharded=True` and `spmd_mesh` is provided, lazily initialized
    `exp_avg` / `exp_avg_sq` tensors are explicitly annotated with the same
    GSPMD FSDP sharding spec as their parameter.
    """

    def __init__(self, params, *args, spmd_mesh=None, fsdp_sharded: bool = False, **kwargs):
        super().__init__(params, *args, **kwargs)
        self._spmd_mesh = spmd_mesh
        self._fsdp_sharded = bool(fsdp_sharded)

    def _maybe_shard_state(self, param: torch.Tensor, tensor: torch.Tensor) -> None:
        if not self._fsdp_sharded or self._spmd_mesh is None:
            return
        from .xla_utils import compute_fsdp_param_spec
        spec = compute_fsdp_param_spec(param, int(self._spmd_mesh.size()))
        if spec is not None:
            import torch_xla.distributed.spmd as xs
            xs.mark_sharding(tensor, self._spmd_mesh, spec)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if any(group.get(key, False) for key in ('capturable', 'differentiable', 'fused', 'foreach')):
                raise ValueError('XLAAdamW requires the single-tensor non-capturable path')
            lr = group['lr']
            if isinstance(lr, torch.Tensor):
                if lr.device.type != 'cpu':
                    raise ValueError('XLAAdamW scheduler LR must stay on the host')
                lr = float(lr)
            beta1, beta2 = group['betas']
            # Parameters can have different step counts when a head is unused.
            controls_by_step = {}
            for param in group['params']:
                if param.grad is None:
                    continue
                if param.is_complex() or param.grad.is_sparse:
                    raise ValueError('XLAAdamW requires real, dense gradients')
                grad = -param.grad if group.get('maximize', False) else param.grad
                state = self.state[param]
                if not state:
                    state['step'] = torch.tensor(0., device='cpu')
                    state['exp_avg'] = torch.zeros_like(param)
                    state['exp_avg_sq'] = torch.zeros_like(param)
                    self._maybe_shard_state(param, state['exp_avg'])
                    self._maybe_shard_state(param, state['exp_avg_sq'])
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(param)
                        self._maybe_shard_state(param, state['max_exp_avg_sq'])
                # AdamW keeps non-capturable step counters on the host, including
                # when restoring its standard state dict.
                if state['step'].device.type != 'cpu':
                    raise ValueError('XLAAdamW step counters must be restored on the host')
                state['step'].add_(1)
                step = int(state['step'])
                key = (step, param.device, param.dtype)
                if key not in controls_by_step:
                    host = torch.tensor([
                        1 - lr * group['weight_decay'],
                        lr / (1 - beta1 ** step),
                        math.sqrt(1 - beta2 ** step),
                    ], dtype=param.dtype, device='cpu')
                    controls_by_step[key] = host.to(param.device)
                decay, step_size, correction = controls_by_step[key].unbind()
                if group['weight_decay'] != 0:
                    param.mul_(decay)
                mean, variance = state['exp_avg'], state['exp_avg_sq']
                mean.lerp_(grad, 1 - beta1)
                variance.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if group['amsgrad']:
                    maximum = state['max_exp_avg_sq']
                    torch.maximum(maximum, variance, out=maximum)
                    variance = maximum
                denom = variance.sqrt().div_(correction).add_(group['eps'])
                param.add_((mean / denom) * -step_size)
        return loss
