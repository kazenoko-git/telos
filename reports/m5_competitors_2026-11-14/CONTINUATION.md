# Nanochat, LitGPT, and evaluator checks

Continuation of the M5 Pro checks on 14 November 2026. All observations are
smokes or scorer tests; none establish a product-wide speed or quality ranking.

## MPS backend smokes

The pinned nanochat model at `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`
completed a tiny 2-layer PyTorch MPS forward/backward pass on M5 Pro. Its logits
had shape `[2, 64, 256]`; the MPS device synchronized successfully. The test
environment has PyTorch 2.15.0, while nanochat declares 2.9.1. This did not run
the nanochat trainer, data pipeline, optimizer recipe, or quality evaluation.

The pinned LitGPT model at `5341568a65e918a85cb284fac038de2157eebf3f` first
failed to import because `psutil` was missing. Installing `psutil==7.2.2` in the
isolated benchmark environment fixed the import. Its 2-layer model then passed
MPS forward, backward, AdamW update, and synchronization with logits shape
`[2, 32, 256]`. This environment also uses PyTorch 2.15.0. These tiny model
smokes establish MPS execution only; neither invoked LitGPT's native pretraining
CLI or establishes training throughput.

Environment additions and locks are in [the updated requirements list](requirements.lock.txt).
Structured results are in [nanochat](nanochat_mps_smoke.json) and
[LitGPT](litgpt_mps_smoke.json). No repository source code was modified.

## Evaluator scoring rules

I installed `lm-eval==0.4.13` into the isolated environment and ran its actual
GSM8K strict and flexible extraction filters beside Télos's
`evaluate_gsm8k_sample`, using eight identical fixed completion strings and
gold answers. The task configuration was taken from the pinned upstream
GSM8K YAML. This checks scorer behavior; it is not an official benchmark run
and includes no model-generated outputs.

Télos and lm-eval's flexible filter agreed on the tested answer-only, unmarked
final-number, currency, and negative-decimal cases. Strict lm-eval extraction
requires a `####` marker, so those same outputs failed its strict path while
Télos accepted them. Conversely, lm-eval's flexible filter selected a later
unrelated number or revised answer in three fixtures where Télos preferred the
earlier `####` answer. Fraction-style `1/2` was not recognized as 0.5 by any
tested path. See [all completions and outcomes](evaluator_scoring_comparison.json).

These differences make prompt format and filter choice material to reported
accuracy. A useful next evaluation comparison needs immutable model-generated
completions on the same public GSM8K test IDs, then both official score paths,
with differences reviewed task by task. It should record the prompt/few-shot
configuration because the upstream lm-eval task uses five examples and
Télos's runner uses its own prompt construction.
