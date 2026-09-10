"""
Dual-Objective Metric Monitor and Lightweight In-Loop Checkpoint Prober.

Provides:
- DualMetricMonitor: Runs periodic evaluation of both causal and bidirectional infilling
  competence on a small held-out dataset slice (e.g. 200 sequences) at checkpoint intervals.
- Detects causal forgetting / divergence early before spending full training budgets.
"""

import math
import torch
import torch.nn.functional as F

from telos.training.schedule import compute_vectorized_roc_auc


class DualMetricMonitor:
    """
    Lightweight periodic probe evaluator for tracking simultaneous causal and infill progress.
    """

    def __init__(
        self,
        val_dataset_matrix,
        vocab_size: int,
        seq_len: int,
        device: torch.device,
        num_probe_sequences: int = 200,
        mask_token_id: int = 1,
        mask_prob: float = 0.15,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.device = device
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob

        # Prepare static held-out evaluation slice
        n_avail = len(val_dataset_matrix)
        n_eval = min(num_probe_sequences, n_avail)
        probe_np = val_dataset_matrix[:n_eval, :seq_len]
        self.probe_tensor = torch.from_numpy(probe_np.astype("int64")).to(device)

        # Pre-generate deterministic mask for infill evaluation slice so comparison across checkpoints is exact
        B, T = self.probe_tensor.shape
        torch.manual_seed(42)
        rand_probs = torch.rand((B, T), device=device)
        self.eval_mask_positions = (rand_probs < mask_prob)
        self.eval_mask_positions[:, 0] = False  # BOS preserved

        self.history: list[dict[str, float]] = []

    @torch.no_grad()
    def evaluate(self, model, current_step: int, batch_size: int = 16) -> dict[str, float]:
        """
        Runs dual causal and infilling probes across the held-out evaluation slice.
        
        Returns:
            dict containing:
                "step": Current training step
                "causal_top1": Next-token top-1 accuracy
                "causal_ce": Next-token cross entropy loss
                "infill_top1": Masked token reconstruction top-1 accuracy
                "infill_ce": Masked token cross entropy loss
                "lrh_auc": ROC-AUC of Learned Reliability Head
                "lrh_acc": Binary accuracy of Learned Reliability Head
        """
        was_training = model.training
        model.eval()

        raw_model = getattr(model, "module", model)
        has_reliability = (
            getattr(raw_model, "reliability_head", None) is not None
            or getattr(getattr(raw_model, "config", None), "use_reliability_head", False)
        )

        total_causal_correct = 0
        total_causal_tokens = 0
        total_causal_ce = 0.0

        total_infill_correct = 0
        total_infill_tokens = 0
        total_infill_ce = 0.0

        all_r_scores = []
        all_r_labels = []

        N = self.probe_tensor.shape[0]
        for start_idx in range(0, N, batch_size):
            end_idx = min(N, start_idx + batch_size)
            batch = self.probe_tensor[start_idx:end_idx]
            b_sz, T = batch.shape

            # 1. Causal Autoregressive Probe
            if has_reliability:
                causal_logits, r_scores = raw_model(batch, mask_override=True, return_reliability=True)
            else:
                causal_logits = raw_model(batch, mask_override=True, return_reliability=False)
                r_scores = None

            shift_logits = causal_logits[:, :-1, :]
            shift_targets = batch[:, 1:]

            causal_preds = shift_logits.argmax(dim=-1)
            total_causal_correct += (causal_preds == shift_targets).sum().item()
            total_causal_tokens += shift_targets.numel()

            c_ce = F.cross_entropy(shift_logits.reshape(-1, self.vocab_size), shift_targets.reshape(-1), reduction="sum")
            total_causal_ce += c_ce.item()

            if r_scores is not None:
                shift_r = r_scores[:, :-1]
                r_labels = (causal_preds == shift_targets).float()
                all_r_scores.append(shift_r.reshape(-1))
                all_r_labels.append(r_labels.reshape(-1))

            # 2. Bidirectional Infill Probe
            batch_mask = self.eval_mask_positions[start_idx:end_idx]
            corrupted = torch.where(
                batch_mask,
                torch.full((b_sz, T), self.mask_token_id, dtype=batch.dtype, device=self.device),
                batch,
            )

            infill_logits = raw_model(corrupted, mask_override=False, return_reliability=False)
            infill_preds = infill_logits.argmax(dim=-1)

            # Compute accuracy strictly over masked positions
            infill_correct = (infill_preds == batch) & batch_mask
            total_infill_correct += infill_correct.sum().item()
            n_masked = batch_mask.sum().item()
            total_infill_tokens += n_masked

            m_ce = F.cross_entropy(infill_logits.reshape(-1, self.vocab_size), batch.reshape(-1), reduction="none").view(b_sz, T)
            total_infill_ce += (m_ce * batch_mask.float()).sum().item()

        causal_top1 = total_causal_correct / max(1, total_causal_tokens)
        causal_ce = total_causal_ce / max(1, total_causal_tokens)

        infill_top1 = total_infill_correct / max(1, total_infill_tokens)
        infill_ce = total_infill_ce / max(1, total_infill_tokens)

        if all_r_scores:
            cat_r = torch.cat(all_r_scores)
            cat_y = torch.cat(all_r_labels)
            lrh_auc = compute_vectorized_roc_auc(cat_r, cat_y)
            lrh_acc = ((cat_r > 0.0).float() == cat_y).float().mean().item()
        else:
            lrh_auc = 0.50
            lrh_acc = 0.50

        results = {
            "step": current_step,
            "causal_top1": float(causal_top1),
            "causal_ce": float(causal_ce),
            "infill_top1": float(infill_top1),
            "infill_ce": float(infill_ce),
            "lrh_auc": float(lrh_auc),
            "lrh_acc": float(lrh_acc),
        }

        self.history.append(results)

        if was_training:
            model.train()

        return results

    def check_divergence(self) -> str | None:
        """
        Inspects historical trajectory to check if causal generation circuit is collapsing.
        Returns a descriptive warning string if causal CE has spiked while infilling improves.
        """
        if len(self.history) < 2:
            return None

        prev = self.history[-2]
        curr = self.history[-1]

        # Check if causal CE increased by more than 15% from previous checkpoint
        if curr["causal_ce"] > prev["causal_ce"] * 1.15 and curr["causal_top1"] < prev["causal_top1"] * 0.90:
            return (
                f"[CAUSAL DIVERGENCE WARNING] Step {curr['step']}: Causal CE rose from {prev['causal_ce']:.3f} "
                f"to {curr['causal_ce']:.3f} (Top-1: {prev['causal_top1']*100:.1f}% -> {curr['causal_top1']*100:.1f}%), "
                f"while Infill Top-1 is {curr['infill_top1']*100:.1f}%. Consider raising alpha_floor."
            )

        return None
