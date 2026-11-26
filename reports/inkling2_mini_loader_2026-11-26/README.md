# Inkling 2 Mini checkpoint and native expert loader

Checkpoint revision `9c0e2960609ba7cb40daf92c6f7bf8ed56e1dd62` of the MLX community conversion was downloaded. The four weight shards total 17,308,326,283 bytes and match the published SHA-256 digests. The local headers matched the earlier HTTP range samples. Native layout inspection used only the local Safetensors headers, index and configuration, with tensor loads blocked. [Pinned checkpoint](https://huggingface.co/mlx-community/Inkling-2-Mini-4bit/tree/9c0e2960609ba7cb40daf92c6f7bf8ed56e1dd62)

| Header measurement | Value |
| --- | ---: |
| Tensors across four shards | 1,559 |
| Physical text layers and expert banks | 24 |
| Routed experts per bank | 128 |
| Experts selected per token and visit | 8 |
| Layer visits and distinct state slots per token | 48 |
| NVFP4 banks with FP32 per-expert global scales | 20 |
| MXFP8 banks | 4 |
| Packed expert payload across all banks | 14.59 GiB |
| Non-expert text payload, including the output head | 954.77 MiB |

The model visits layers 0–5, then layers 6–17 three times, then layers 18–23. Telos maps each repeat to the same physical expert bank and separate state slots. Each NVFP4 expert occupies about 4.27 MiB of packed payload and each MXFP8 expert about 7.83 MiB. These are file sizes. They exclude runtime workspace, activations, and KV and convolution state. [Thinking Machines architecture description](https://thinkingmachines.ai/news/inkling-2-mini/) · [vLLM loop and cache description](https://vllm.ai/blog/2026-11-20-inkling2-mini)

The native code validates the fused interleaved gate/up and down ranges, reads selected experts through Telos's existing SSD cache, and applies the per-expert NVFP4 scale using MLX. It also validates FP32 router tensors and keeps the selection bias out of the mixture weights. Tiny sharded fixtures check exact resident and SSD-sliced expert outputs for NVFP4 and MXFP8 in FP32 and BF16, plus an exact routed mixture and separate loop slot assignments. The real weight payload was used for transfer integrity checks only. No generation or MLX-LM was used. [Interleaved projection layout](https://github.com/vllm-project/vllm/blob/main/vllm/models/inkling/nvidia/mlp.py) · [MLX quantization API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.dequantize.html)

The initial checkpoint and expert-loader milestone reported 25 passes and one skip across the targeted Inkling, LFM, Qwen3 and inference fixtures.

## Native text path

The native path handles sliding and full grouped-query attention with learned relative position bias and a query-dependent sink. It uses static causal convolution for sliding-layer keys and values and dynamic per-head taps for full attention. Block residual reads and four gated message streams feed each repeat from the same prelude, using the previous pass's completed core blocks and last partial block. Per-visit KV and convolution state stays separate. [Inkling implementation reference](https://github.com/huggingface/transformers/blob/main/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py) · [Loop and cache design](https://vllm.ai/blog/2026-11-20-inkling2-mini)

An eight-layer sharded fixture exercised a three-pass core with 16 state slots, affine-quantized attention and output weights, and both expert formats. Resident and SSD modes produced identical prefill and decode hidden states, decode logits, KV and convolution state, and greedy token IDs and text. Separate tiny tests verify relative bias, sinks, sliding-window truncation and input-dependent convolution taps against explicit expected values. The fixture also verifies that an insufficient budget is rejected before opening tensor payloads.

Inspection of the pinned checkpoint's headers accounted for all 1,137 text tensors and verified that the tokenizer matches the unpadded output width. With a 2 GiB budget, the header-only planner admitted about 449 MiB of experts at 128 tokens of context and 251 MiB at 512 tokens. It rejected a 2,048-token context at the same budget. These are planner estimates. All inference tests used tiny fixtures.
