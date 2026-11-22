"""Opt-in Cautious AdamW optimizer (ICLR 2026, arXiv:2411.16085).

Masks coordinate updates where momentum and the current gradient disagree in
sign, rescaling active coordinates to preserve step scale without retuning.
"""

import math
import torch


class CautiousAdamW(torch.optim.AdamW):
    """AdamW with cautious sign-alignment masking on parameter updates."""

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            amsgrad = group["amsgrad"]
            maximize = group.get("maximize", False)

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.is_complex() or param.grad.is_sparse:
                    raise ValueError("CautiousAdamW requires real, dense gradients")
                grad = -param.grad if maximize else param.grad
                state = self.state[param]
                if not state:
                    state["step"] = torch.tensor(0.0, device="cpu")
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(param)

                state["step"].add_(1)
                step = int(state["step"])

                if wd != 0.0:
                    param.mul_(1.0 - lr * wd)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.lerp_(grad, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if amsgrad:
                    max_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_sq, exp_avg_sq, out=max_sq)
                    denom_sq = max_sq
                else:
                    denom_sq = exp_avg_sq

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1
                denom = (denom_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                update = exp_avg / denom
                mask = (update * grad > 0).to(update.dtype)
                mask_scale = mask.mean().clamp_min(1e-3)
                param.add_(update * (mask / mask_scale), alpha=-step_size)

        return loss
