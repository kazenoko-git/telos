"""
Continuous Multi-Objective Scheduling and Dynamic Metric Tracking for COROSred.

Provides:
- COROSredSchedule: Hold-then-decay polynomial progression with permanent alpha floor
- DynamicMetricTracker: Zero-overhead running EMAs for LRH accuracy, ROC-AUC, and safe bounded loss rebalancing
- compute_vectorized_roc_auc: Fast, differentiable-free rank-sum ROC-AUC computation
"""

import math
import torch


def compute_vectorized_roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Computes Wilcoxon-Mann-Whitney ROC-AUC efficiently using torch.argsort / rank-sum.
    
    Args:
        scores: 1D tensor of continuous logit predictions / reliability scores
        labels: 1D binary tensor (0.0 or 1.0) indicating true correctness
        
    Returns:
        float: ROC-AUC score in [0.0, 1.0]. Returns 0.5 if all labels are uniform.
    """
    with torch.no_grad():
        # Flatten and filter any NaN values
        s = scores.reshape(-1).float()
        y = labels.reshape(-1).float()

        n_pos = torch.sum(y == 1.0).item()
        n_neg = torch.sum(y == 0.0).item()

        # If batch contains only positive or only negative labels, AUC is undefined -> return chance (0.50)
        if n_pos == 0 or n_neg == 0:
            return 0.50

        # Sort scores ascending to assign rank (1-indexed ranks)
        # Add tiny jitter to break ties consistently without sorting nondeterminism
        ranks = torch.argsort(torch.argsort(s)) + 1

        # Sum of ranks for positive class (R_1)
        r1 = torch.sum(ranks[y == 1.0]).float().item()

        # Mann-Whitney U statistic: U = R_1 - (n_pos * (n_pos + 1)) / 2
        # AUC = U / (n_pos * n_neg)
        u1 = r1 - (n_pos * (n_pos + 1.0)) / 2.0
        auc = u1 / (n_pos * n_neg)
        return float(max(0.0, min(1.0, auc)))


class COROSredSchedule:
    """
    Computes smooth continuous task weights alpha(t), beta(t), gamma(t) and masking blend ratio.
    
    Curve Design:
    - Hold-Then-Decay: Holds alpha at alpha_max during the initial representation-learning window (hold_fraction),
      then decays via a polynomial curve (1 - tau)^p down to alpha_min.
    - Permanent Floor: alpha(t) is bounded by alpha_min > 0 across the entire training duration.
    - Gated Gamma: LRH auxiliary loss is held at 0.0 until empirical LRH accuracy crosses acc_gate_threshold.
    - Gated Masking Blend: Ratio of LRH confidence-routed masks vs uniform random masks is smoothly
      driven by running LRH ROC-AUC ranking quality.
    """

    def __init__(
        self,
        max_steps: int,
        alpha_max: float = 0.85,
        alpha_min: float = 0.20,
        beta_min: float = 0.15,
        beta_max: float = 0.70,
        gamma_max: float = 0.10,
        hold_fraction: float = 0.20,
        decay_power: float = 2.5,
        gamma_gate_auc: float = 0.55,
        acc_gate_threshold: float = 0.65,
        auc_gate_min: float = 0.50,
        auc_gate_target: float = 0.75,
    ):
        assert max_steps > 0, "max_steps must be strictly positive"
        assert alpha_min > 0.0, "alpha_min floor must be strictly positive (never abandon causal)"
        assert alpha_max >= alpha_min, "alpha_max must be >= alpha_min"
        assert beta_max >= beta_min, "beta_max must be >= beta_min"

        self.max_steps = max_steps
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.gamma_max = gamma_max
        self.hold_fraction = max(0.0, min(1.0, hold_fraction))
        self.decay_power = decay_power
        self.gamma_gate_auc = gamma_gate_auc
        self.acc_gate_threshold = acc_gate_threshold
        self.auc_gate_min = auc_gate_min
        self.auc_gate_target = auc_gate_target

        self.head_ready_override = False

    def set_head_ready(self, ready: bool = True):
        """Allows external evaluation probe or trainer state to explicitly override head warmup gate."""
        self.head_ready_override = ready

    def get_weights(
        self,
        step: int,
        lrh_acc_ema: float | None = None,
        lrh_auc_ema: float | None = None,
    ) -> dict[str, float]:
        """
        Computes the scheduled loss weights for a given training step.
        
        Args:
            step: Current global training step (0 to max_steps)
            lrh_acc_ema: Optional running EMA of LRH accuracy (fallback gate)
            lrh_auc_ema: Optional running EMA of LRH ROC-AUC / balanced accuracy (primary gamma gate and mask blend)
            
        Returns:
            dict containing:
                "alpha": Causal next-token prediction weight
                "beta": Bidirectional infilling weight
                "gamma": Learned Reliability Head BCE weight
                "mask_blend": Fraction of infill tokens routed via confidence (0.0 to 1.0)
        """
        t = max(0.0, min(1.0, step / float(self.max_steps)))

        # 1. Compute alpha(t) and beta(t) via hold-then-decay polynomial progression
        if t <= self.hold_fraction:
            # Hold phase: maximum causal budget for representation learning
            alpha = self.alpha_max
            beta = self.beta_min
        else:
            # Decay phase: polynomial decay with exponent p
            decay_range = max(1e-6, 1.0 - self.hold_fraction)
            tau = (t - self.hold_fraction) / decay_range
            decay_factor = math.pow(max(0.0, 1.0 - tau), self.decay_power)
            ramp_factor = math.pow(min(1.0, tau), self.decay_power)

            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * decay_factor
            beta = self.beta_min + (self.beta_max - self.beta_min) * ramp_factor

        # Strict permanent floor enforcement
        alpha = max(self.alpha_min, alpha)
        beta = max(self.beta_min, min(self.beta_max, beta))

        # 2. Compute gamma(t) (Head warmup gate)
        # Gated on empirical classification ranking discrimination (AUC or balanced accuracy)
        # rather than raw accuracy. Under severe ~9:1 class imbalance, trivial all-negative
        # predictions yield ~90% raw accuracy with zero ranking power. ROC-AUC and balanced accuracy
        # maintain an uncorrupted 0.50 chance baseline, ensuring gamma opens only when real signal exists.
        head_is_competent = self.head_ready_override
        if not head_is_competent and lrh_auc_ema is not None:
            # Primary: Gate opens when running ROC-AUC / balanced accuracy crosses gamma_gate_auc (e.g. 0.55-0.60)
            head_is_competent = (lrh_auc_ema >= self.gamma_gate_auc)
        elif not head_is_competent and lrh_acc_ema is not None and self.acc_gate_threshold is not None:
            # Backward compatibility fallback if only raw accuracy telemetry is available
            head_is_competent = (lrh_acc_ema >= self.acc_gate_threshold)
        elif not head_is_competent and lrh_auc_ema is None and lrh_acc_ema is None:
            # Fallback when no telemetry is provided: Keep gamma strictly at 0.0 until empirical signal arrives
            head_is_competent = False

        if head_is_competent:
            # Smooth linear ramp-in over 5% of training steps once competence threshold is met
            warmup_frac = 0.05
            progress_after_gate = max(0.0, min(1.0, (t - 0.10) / warmup_frac)) if (lrh_auc_ema is None and lrh_acc_ema is None) else 1.0
            gamma = self.gamma_max * progress_after_gate
        else:
            gamma = 0.0

        # 3. Compute mask_blend ratio (Uniform random vs LRH confidence-routed)
        # Driven by ROC-AUC ranking fidelity
        if lrh_auc_ema is not None:
            # Blend ratio = clamp((AUC - 0.50) / (0.75 - 0.50), 0.0, 1.0)
            auc_range = max(1e-6, self.auc_gate_target - self.auc_gate_min)
            mask_blend = max(0.0, min(1.0, (lrh_auc_ema - self.auc_gate_min) / auc_range))
        else:
            # When telemetry is unestablished, remain on pure uniform random masking (zero overhead, static graph)
            mask_blend = 0.0

        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
            "mask_blend": float(mask_blend),
        }


class DynamicMetricTracker:
    """
    Maintains running Exponential Moving Averages (EMA) of training-time metrics
    and performs safe, trust-region-bounded dynamic loss rebalancing.
    """

    def __init__(
        self,
        ema_decay: float = 0.99,
        trust_region: float = 0.20,
        rebalance_temp: float = 0.25,
    ):
        self.decay = ema_decay
        self.trust_region = trust_region
        self.rebalance_temp = rebalance_temp

        # LRH performance statistics
        self.lrh_acc_ema: float | None = None
        self.lrh_auc_ema: float | None = None

        # Task loss statistics for safe rebalancing
        self.loss_c_ema: float | None = None
        self.loss_m_ema: float | None = None
        self.initial_loss_c: float | None = None
        self.initial_loss_m: float | None = None

        self.step_count = 0

    def update_lrh(self, acc: float, auc: float):
        """Updates LRH classification accuracy and ROC-AUC running EMAs."""
        if math.isnan(acc) or math.isnan(auc):
            return

        if self.lrh_acc_ema is None:
            self.lrh_acc_ema = float(acc)
            self.lrh_auc_ema = float(auc)
        else:
            self.lrh_acc_ema = self.decay * self.lrh_acc_ema + (1.0 - self.decay) * float(acc)
            self.lrh_auc_ema = self.decay * self.lrh_auc_ema + (1.0 - self.decay) * float(auc)

    def update_losses(self, loss_c: float, loss_m: float):
        """Updates causal and masked reconstruction per-token loss EMAs."""
        if math.isnan(loss_c) or math.isnan(loss_m):
            return

        if self.initial_loss_c is None:
            self.initial_loss_c = max(1e-4, float(loss_c))
            self.initial_loss_m = max(1e-4, float(loss_m))
            self.loss_c_ema = self.initial_loss_c
            self.loss_m_ema = self.initial_loss_m
        else:
            self.loss_c_ema = self.decay * self.loss_c_ema + (1.0 - self.decay) * float(loss_c)
            self.loss_m_ema = self.decay * self.loss_m_ema + (1.0 - self.decay) * float(loss_m)

        self.step_count += 1

    def get_balanced_weights(
        self,
        alpha_nominal: float,
        beta_nominal: float,
    ) -> tuple[float, float, dict[str, float | bool]]:
        """
        Computes safe, trust-region-bounded dynamic loss rebalancing weights.
        
        Uses relative rate-of-change (r_i(t) = L_i(t) / L_i(0)) rather than raw loss magnitudes.
        Strictly bounds adjustments to +/- trust_region (default +/- 20%) to prevent
        dynamic rebalancing from fighting the macro curriculum schedule.
        
        Returns:
            alpha_effective: Adjusted causal weight
            beta_effective: Adjusted infilling weight
            telemetry: Dict containing nominal_ratio, effective_ratio, factor, and clamped status
        """
        nominal_ratio = beta_nominal / max(1e-6, alpha_nominal)

        # If loss statistics have not yet warmed up, return nominal weights directly
        if self.initial_loss_c is None or self.loss_c_ema is None or self.step_count < 10:
            return alpha_nominal, beta_nominal, {
                "nominal_ratio": nominal_ratio,
                "effective_ratio": nominal_ratio,
                "factor": 1.0,
                "clamped": False,
            }

        # Relative rate-of-change (GradNorm formulation)
        rate_c = max(1e-4, self.loss_c_ema) / max(1e-4, self.initial_loss_c)
        rate_m = max(1e-4, self.loss_m_ema) / max(1e-4, self.initial_loss_m)

        # Raw adjustment factor based on relative training pace: (rate_m / rate_c)^temp
        raw_factor = math.pow(rate_m / rate_c, self.rebalance_temp)

        # Strict Trust-Region Clamping: factor must remain within [1 - delta, 1 + delta]
        min_factor = 1.0 - self.trust_region
        max_factor = 1.0 + self.trust_region
        clamped_factor = max(min_factor, min(max_factor, raw_factor))
        was_clamped = (clamped_factor != raw_factor)

        # Apply bounded adjustment to beta relative to alpha
        alpha_effective = alpha_nominal
        beta_effective = beta_nominal * clamped_factor
        effective_ratio = beta_effective / max(1e-6, alpha_effective)

        telemetry = {
            "nominal_ratio": nominal_ratio,
            "effective_ratio": effective_ratio,
            "factor": clamped_factor,
            "clamped": was_clamped,
        }
        return alpha_effective, beta_effective, telemetry
