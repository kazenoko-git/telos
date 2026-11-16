# Investigation complete

See [REPORT.md](REPORT.md) for final results and reproduction.

The confirmed result is +18.58% warmed pre-training throughput for the tested
49.17M AR/context1024 configuration with isolated MLX 0.32.4, at higher peak
allocation. No repeatable context256 gain or new prototype-kernel gain was
established. Production defaults and the shared runtime remain unchanged.

Research is recorded on `codex/mlx-pretrain-optimization`; package version and
release tags remain unchanged.
