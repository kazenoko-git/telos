# Documented XLA/CUDA differences and relevance to Telos — 18 October 2026

## Answer

The class of problem is documented: backend-dependent numerical behavior, historical XLA autocast defects, and user reports of worse TPU convergence/accuracy. No reviewed source establishes a universal TPU training-quality penalty or identifies Telos's particular cause. Published reports do not make the affected runtime versions or mechanisms identical to ours.

## Confirmed implementation history

- [XLA PR #8094](https://github.com/pytorch/xla/pull/8094), merged 30 September 2024, enabled FP32 autocasting of cross-entropy. Its example shows why an FP32 final loss did not necessarily imply FP32 intermediate computation. The shown HLO was from AWS Neuron; this is a shared XLA autocast-policy fix, not a reproduction of the Telos TPU failure. The reviewed v2.8 policy already contains the fix; this is not evidence that our original run lacked it. Exact original runtime/build remains unestablished.
- [XLA PR #7878](https://github.com/pytorch/xla/pull/7878), merged 19 August 2024, fixed norm-result dtype handling under AMP. Tagged XLA GPU, it demonstrates a real XLA policy/metadata defect, not a confirmed TPU clipping bug in Telos. The fix is historical and must be matched to versions before assigning relevance.
- [PyTorch numerical-accuracy documentation](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html) documents platform/kernel differences and reduced-precision GEMM/SDPA reductions. [SDPA documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) warns of backend-dependent numerical results. These describe supported behavior, not proof that ordinary numerical differences explain a large sustained quality gap.

## Similar user reports

- [timm/DeiT discussion #934](https://github.com/huggingface/pytorch-image-models/discussions/934), 2021–2022: author reported 73.1 versus 74.6 top-1 and roughly two-point short-run deficits on TPU. Longer training reportedly caught up around epoch 150. Maintainer identified LR-scaling differences between the codebases and discussed pure-BF16 versus mixed-precision sensitivity. Consequently this is precedent for a convergence complaint, not a clean controlled proof that TPU hardware causes inferior convergence. It does not reopen Telos's previously audited batch/LR accounting or recommend blanket FP32.
- [PyTorch/XLA issue #5755](https://github.com/pytorch/xla/issues/5755), opened 2023: author reported a large probing accuracy drop after loading/saving ViT weights via TPU, even without training. No confirmed resolution was visible in the retrieved issue. This is principally a checkpoint/load-save report, not evidence of the same training failure or of a general TPU quality penalty.

## Why extensive TPU usage does not settle this

TPU hardware, the XLA compiler, and the PyTorch/XLA integration are different layers. Google discusses its separate TorchTPU integration and the constraints of predecessor PyTorch/XLA in its [2026 engineering article](https://developers.googleblog.com/torchtpu-running-pytorch-natively-on-tpus-at-google-scale/). Hardware success at scale does not certify every version of each framework integration or custom operator path. This also provides no evidence of the cause of model response style differences.

## Remaining candidates in Telos

1. A runtime/version-dependent operator or autocast/lowering discrepancy in forward/backward or gradient handling, with attention being a concrete shared API backed by different implementations. Unmeasured, not established.
2. The exact training code/environment differed across historical experiments in a way not captured by YAML. Current code audits cannot retroactively certify old execution. Mixed checkpoint lineage has been observed; it does not erase the measured endpoint deficit.
3. Different optimization trajectories due to hardware arithmetic are possible, but their size and directional effect have not been demonstrated. Calling them generic BF16 drift would not explain this persistent gap.

Saved checkpoints and the local CPU optimizer replay do not isolate these candidates. Without accelerator access, the useful remaining audit is recovering exact historical torch, torch_xla, libtpu and source revisions and matching them to specific documented fixes. No Kaggle session, cloud job, code change, commit or push was performed for this literature check.
