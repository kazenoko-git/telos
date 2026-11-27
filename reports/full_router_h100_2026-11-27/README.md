# Full-router H100 pilots, 2026-11-27

All three eight-step pilots completed on a single NVIDIA H100 80GB, one model at a time, using the PyPI package telos-ml==1.9.1rc2. Every router had finite, nonzero selector and token-matrix gradients on the first step. Training checked finite routing gradients at every step and rejected nonfinite total gradient norms. Frozen teacher router parameters received no gradients.

| Model | Routers | Held-out response NLL | Held-out KL to teacher | Peak GPU allocated / reserved GiB | Worker seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| thinkingmachines/Inkling-2-Mini | 24 | 3.8840 → 3.5011 | 1.5537 → 1.4837 | 50.90 / 52.34 | 74.97 |
| Qwen/Qwen3.5-35B-A3B | 40 | 4.2445 → 3.7254 | 1.6176 → 1.4708 | 67.22 / 67.35 | 113.55 |
| google/gemma-4-26B-A4B-it | 30 | 5.2203 → 4.9735 | 1.6752 → 1.5345 | 49.57 / 49.73 | 55.10 |

## Evaluation and reload

Each model used seed 73, W=16, eight optimizer steps, batch size 1, gradient accumulation 2, FP32 full-size routing updates, rank-8 attention LoRA, AdamW at 5e-5, and a 128-token sequence limit. Transformers was pinned to 8e094f8970a60ca83052e2c199783b247b8fc13e. Actual model revisions and prompt lists are in the [raw results](rc2/summary.json).

Dolly examples were deduplicated by instruction/context before splitting into 16 training and eight held-out prompts per model. Tokenization and length filtering are model-specific, so this is a within-model pilot comparison. Scores are response-token cross-entropy averaged within each prompt, then averaged across eight prompts. The baseline is untrained W=16 routing on the same starting model. These short held-out scores do not establish broad instruction quality or free-generation quality.

All three exports passed exact parameter and held-out FP32 logit reload checks. The check zeroed all trainable tensors on the loaded base model, reloaded the Safetensors export, and compared weights and logits against their pre-export values. It did not reload a second copy of the full base checkpoint. [Local artifacts and hashes](artifacts.json)

## Host resources and cost

Initial Inkling and Qwen jobs requested four CPU cores and 128 GiB host RAM. This was overallocated. After the user's correction, the runner requests one physical CPU core (two vCPU) and 8 GiB host RAM for both training and download workers. Gemma completed under the reduced request. A live sample of Gemma's training process showed 5.51 GiB RSS. This was a single sample. [Host sample](gemma-host-sample.json)

Model downloads ran on CPU. Each GPU call had a 1,200-second timeout, automatic retries disabled, one container maximum, and a two-second idle window.
The corrected pilots' in-function compute estimates sum to USD 0.368. Including the initial completed diagnostic gives USD 0.498. Rates were conservatively rounded to USD 0.0016/second for the initial larger hosts and USD 0.0012/second for the reduced host. The published H100 GPU price was USD 0.001097/second, with CPU and memory charged separately. [Modal pricing](https://modal.com/pricing)

These estimates exclude image construction, CPU preparation, container startup, and the interrupted Gemma load. The final account invoice has not been reconciled. Across this task there were five GPU calls, including the initial diagnostic and interrupted load. Charging every call to its entire timeout gives a conservative GPU-worker ceiling of USD 9.12 at the rounded rates, plus CPU preparation/build charges, within the USD 30 task budget.

## Pilot-discovered correction

The initial rc1 Inkling diagnostic showed zero coverage loss. The existing PyTorch distillation code cleared recorded teacher routes when resetting for the student pass. Commit 0a65fdf7 preserves those routes, and tiny fixtures now assert positive coverage loss. The initial run remains [diagnostic evidence](inkling-rc1-diagnostic.json). The table above uses fresh rc2 training from the original base weights with coverage supervision active.

The initial rc1 app was stopped during Qwen's CPU preparation. The second app was stopped during Gemma loading to apply the user's reduced host limits. Gemma then completed in the third app. The first two completed rc2 models were retained. No longer training run was launched.

## Publication and execution

- [PyPI 1.9.1rc2](https://pypi.org/project/telos-ml/1.9.1rc2/) and [distribution hashes](published_distributions.json).
- [rc2 publication workflow](https://github.com/kazenoko-git/telos/actions/runs/45940291584).
- [Initial diagnostic app](https://modal.com/apps/kazenoko-main/main/ap-BUDfxkzAj3CPqsAUndOMvc).
- [Inkling/Qwen rc2 app](https://modal.com/apps/kazenoko-main/main/ap-MfgkGDXCcVc9PUAPswb2XU).
- [Reduced-host Gemma app](https://modal.com/apps/kazenoko-main/main/ap-a8XJIZtfHssTZbZ4ene78R).
- [Bounded pilot runner](../../scripts/modal_router_pilots.py).

Local routing/distillation tests passed 40 tests after the coverage fix. GitHub CI and both prerelease publication workflows passed. The artifacts and raw metrics support a handover for a larger held-out evaluation before choosing longer training settings. Future runs retain the user's two-vCPU/eight-GiB host limits.
