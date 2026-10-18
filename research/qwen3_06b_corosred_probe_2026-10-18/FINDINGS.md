# Qwen3-0.6B AR vs Converted MDLM COROSred Investigation Report

**Date:** 2026-10-18  
**Hardware:** Apple M5 Pro (24 GB Unified Memory), Apple Silicon MPS  
**Scope:** Direct lineage audit, parameter alignment, and 7-condition COROSred evaluation of base AR (`Qwen/Qwen3-0.6B`) vs converted masked diffusion derivative (`dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`).

---

## 1. Executive Summary

1. **Parameter Lineage Alignment Verified:**
   Unlike the previous DataDecide 530M pair (which had mutually orthogonal weights with $|\cos| < 0.002$), `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` is a direct derivative of `Qwen/Qwen3-0.6B`. Across all 311 weight tensors (751.6M parameters), the mean parameter cosine similarity is **0.9753** (min 0.9242, max 0.99999997), and the embedding cosine is **0.9656**.

2. **COROSred Two-Model Pipeline Shows Strong Synergy:**
   A confidence-routed selective re-diffusion pipeline (`corosred_two_model`)—where AR drafts causal tokens and selectively routes to MDLM bidirectional repair only when AR margin is low (<0.5) or entropy is high (>1.0)—achieves **59/64 (92.2%)** top-1 accuracy on local code reconstruction, with mean NLL dropping to **0.5720**.
   - AR alone: 47/64 (73.4%), NLL = 0.9208
   - MDLM alone: 50/64 (78.1%), NLL = 1.0832
   - Routing triggered on 23/64 cases (35.9%). In 14 cases, AR was wrong, routing detected it, and MDLM fixed it. In only 2 cases did MDLM break a correct AR token. Net improvement: **+18.8 percentage points** over AR alone.

3. **Weight Averaging and Single-Model Interpolation:**
   - With aligned parameter coordinates, direct weight interpolation does *not* completely collapse: `weight_avg_50` reaches **28/64 (43.8%)** and `weight_interp_25` (25% AR + 75% MDLM) reaches **46/64 (71.9%)**.
   - However, simple weight averaging does not match native models, and layer substitution (`layer_subst_half`: layers 0-13 AR, layers 14-27 MDLM) collapsed to **0/64 (0.0%)**, indicating that internal activation manifolds diverge sufficiently to break cross-layer stitching.

4. **Counterfactual Suffix Deduction Remains Hard for 0.6B:**
   Neither native model nor any combination resolved both members of the 24 counterfactual arithmetic/comparison pairs (`0/24 both top-1`). While MDLM showed correct candidate preference on 3/24 pairs, small model capacity limits multi-hop deduction across assert suffixes without specialized fine-tuning.

---

## 2. Models and Checkpoints Audited

| Property | Autoregressive (AR) | Masked Diffusion (MDLM) |
| :--- | :--- | :--- |
| **Hugging Face ID** | `Qwen/Qwen3-0.6B` | `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` |
| **Pinned Revision** | `c1899de289a04d12100db370d81485cdf75e47ca` | `c8d24a3f4adaeef46881b450e1bf7d1005203bd7` |
| **Parameters** | 751,632,384 | 751,632,384 |
| **Layers / Heads** | 28 layers, 16 Q heads, 8 KV heads | 28 layers, 16 Q heads, 8 KV heads |
| **Hidden / Intermediate** | 1024 / 3072 (SwiGLU) | 1024 / 3072 (SwiGLU) |
| **QK-Norm** | RMSNorm per head (dim 128) | RMSNorm per head (dim 128) |
| **RoPE Theta** | 1,000,000.0 | 1,000,000.0 |
| **Vocab / Special** | 151,936 | 151,936 (`<|mask|>` = `151669`) |
| **safetensors Size** | 1,503,300,328 bytes (1.40 GB) | 1,503,300,328 bytes (1.40 GB) |

---

## 3. Parameter Alignment Metrics

Computed in double precision (`float64`) across all 311 weight tensors:

- **Overall Mean Cosine:** 0.9753
- **Min Cosine:** 0.9242 (`layers.19.self_attn.k_proj.weight`)
- **Max Cosine:** 0.99999997 (`layers.26.input_layernorm.weight`)
- **Mean Relative Frobenius Shift:** 0.1854
- **Token Embeddings Cosine:** 0.9656 (Relative shift: 0.2686)

### Component Cosine Breakdown:
- `norm.weight` (LayerNorms): 0.9998 – 0.99999
- `q_norm` / `k_norm`: 0.99993 – 0.99997
- `v_proj`: 0.9685
- `up_proj`: 0.9643
- `gate_proj`: 0.9632
- `o_proj`: 0.9631
- `q_proj`: 0.9611
- `down_proj`: 0.9595
- `k_proj`: 0.9484

---

## 4. Experiment Matrix Results

Evaluated across 112 cases: 64 local Python code reconstruction tasks + 24 counterfactual pairs (48 cases).

| Condition | Local Code Top-1 | Local Code Mean NLL | Counterfactual Pairs Both Top-1 | Counterfactual Prefer Required | Suffix Pred Changes | Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`native_ar`** | **47/64 (73.4%)** | 0.9208 | 0/24 | 0/24 | 0/24 | Frozen AR, causal attention, shifted readout at `hole-1`. Zero leakage control verified. |
| **`native_mdlm`** | **50/64 (78.1%)** | 1.0832 | 0/24 | 3/24 | 1/24 | Frozen MDLM, bidirectional attention, mask token `<|mask|>` at `hole`. |
| **`corosred_two_model`** | **59/64 (92.2%)** | **0.5720** | 0/24 | 3/24 | 1/24 | AR causal draft $\to$ uncertainty routing (margin < 0.5 or entropy > 1.0) $\to$ MDLM bidirectional repair. |
| **`weight_avg_50`** | 28/64 (43.8%) | 2.4472 | 0/24 | 0/24 | 7/24 | $0.5 W_{AR} + 0.5 W_{DIFF}$, bidirectional attention. |
| **`weight_interp_75`** | 2/64 (3.1%) | 9.7810 | 0/24 | 3/24 | 12/24 | $0.75 W_{AR} + 0.25 W_{DIFF}$, bidirectional attention. |
| **`weight_interp_25`** | 46/64 (71.9%) | 1.1515 | 0/24 | 1/24 | 4/24 | $0.25 W_{AR} + 0.75 W_{DIFF}$, bidirectional attention. |
| **`layer_subst_half`** | 0/64 (0.0%) | 6.3157 | 0/24 | 4/24 | 4/24 | Layers 0-13 from AR, layers 14-27 from MDLM, bidirectional attention. |

---

## 5. Detailed Breakdown of Two-Model Confidence Routing

- **Cases Routed to MDLM:** 23/64 (35.9%)
- **AR Errors Fixed by MDLM:** 14 cases (+21.9% absolute gain)
  - Variable names: `prev` $\to$ `curr`, `n` $\to$ `d`, `result` $\to$ `res`, `count` $\to$ `counts`
  - Flow control: `if` $\to$ `sign`, `i` $\to$ `for`, `result` $\to$ `return`, `-` $\to$ `for`
  - Operators: `\n` $\to$ `+`
  - Imports: `if` / `x` $\to$ `import`
- **AR Correct Broken by MDLM:** 2 cases (–3.1% absolute loss)
  - `return` broken to `=` in `local_task_0033`
  - `-` broken to `,` in `local_task_0036`
- **AR Errors Missed by Routing:** 1 case (AR was confident in the wrong token)
- **Net Gain:** +12 correct reconstructions (+18.8 percentage points over AR alone, +14.1 over MDLM alone).

---

## 6. Conclusions and Practical Implications for COROSred

1. **Two-Model Confidence Routing is Immediately Practical:**
   The hypothesis behind COROSred—that AR models can draft rapidly and reliably on unambiguous prefixes, while a matched bidirectional diffusion model can step in selectively to repair high-uncertainty regions using right context—is quantitatively validated by the two-model pipeline. With zero training, top-1 accuracy jumped from 73.4% to 92.2% with a 38% reduction in mean NLL.

2. **Weight Merging vs Two-Model Routing:**
   While direct conversion lineage preserves coordinate basis ($\cos \approx 0.975$) and prevents the catastrophic collapse seen in orthogonal runs, static weight interpolation ($0.5/0.5$) drops accuracy from 78.1% to 43.8%, and layer substitution drops to 0.0%. Thus, without specialized inference techniques (such as task-vector arithmetic, rank-1 LoRA-style delta projections, or dual-path routing), weight merging does not yield a viable single-model checkpoint. A two-model pipeline remains the practical, high-accuracy deployment method.
