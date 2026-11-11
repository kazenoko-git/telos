# MLX accumulation experiments — 2026-11-11

No candidate established a useful throughput gain. Production training remains on the existing bounded-memory path.

| Variant | Baseline tok/s | Candidate tok/s | Change | Peak GiB, baseline / candidate |
|---|---:|---:|---:|---:|
| One-microbatch lookahead | 29,021 | 25,074 | -13.60% | 1.925 / 1.925 |
| Compiled additions | 37,598 | 36,710 | -2.36% | 1.724 / 1.724 |
| Defer final wait | 37,613 | 37,721 | +0.29% | 1.725 / 1.774 |

Throughput is total measured tokens divided by total measured update time across each pair of trials. The lookahead row is inconclusive because the baseline trials drifted from 23.8k to 37.1k tok/s. The other two comparisons were steadier.

## Method and limitations

- 50M AR (49,166,848 parameters), real corpus, seq256, MBS8, accumulation8, BF16 with FP32 master weights.
- Each trial: fresh process, seed42, four warmup updates, sixteen measured updates. Order baseline/candidate/candidate/baseline.
- Earlier exploratory trials 0-5 excluded: different lookahead variant and potential overlap with GPU correctness tests.
- Trials6-17 ran without simultaneous GPU tests; desktop applications remained active. Lookahead comparison experienced substantial timing drift.
- Independent baseline runs have different final hashes and small loss differences, so full-transformer bitwise parity is not established. The worst lookahead loss difference against trial6 was0.01755; a later baseline differed by0.02302. Cause of run-to-run variation is not isolated.
- Deterministic small-model tests show exact equality of loss, parameters, optimizer states, FP32 masters, and subsequent random samples for all three variants, FP32/BF16, accumulation2/8, changing learning rates.
- Production accumulation code restored. Experiments remain in scripts/mlx_accumulation_experiments.py and are used only by benchmark/tests.
- The four-GiB MLX limit is advisory; benchmarks additionally check observed MLX peak, memory headroom, and swap headroom each update.

System swap did not increase during the retained trials (5,860.56 MiB at the first start; 5,852.56 MiB at the final end). All measured MLX peaks were below 1.93 GiB.

## Reproduction

Run each command in a fresh process with `PYTHONPATH=.` and the project virtualenv. Use `--steps 16` and a different output path for every trial. The default selects the production baseline; select one experiment with `--pipeline`, `--compiled-add`, or `--defer-last`. For example:

```sh
PYTHONPATH=. .venv/bin/python scripts/benchmark_mlx_accumulation.py --steps 16 --output /tmp/mlx-baseline.json
PYTHONPATH=. .venv/bin/python scripts/benchmark_mlx_accumulation.py --steps 16 --defer-last --output /tmp/mlx-defer.json
```

Correctness: `PYTHONPATH=. .venv/bin/python -m pytest tests/test_mlx_step.py tests/test_mlx_accumulation_experiments.py -q` — 17 passed.

The results suggest these host scheduling/addition changes are not a major throughput opportunity at MBS8. They do not identify the dominant GPU kernel; that requires finer GPU timing.
