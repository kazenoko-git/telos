# Reproduce the study

## Packaged evaluation

`telos-ml==1.6.2` includes the frozen suite, calculator plans, generic runners and historical response snapshots:

```sh
telos eval documents replay --reference --output replay.json --html-report replay.html
```

See `docs/document-evaluation.md` on the main branch for new inference. This study directory also preserves the original runner environments and records. `package_audit.json` is the historical inspection of version 1.6.1, before integration.

## Recompute saved scores (offline)

Extract `Reader-1-reproduction.zip`, enter the extracted root, then run:

```sh
python3 reports/reader1_consolidated_2026-11-15/replay_saved_scores.py
```

This uses only the Python standard library and the included scoring source. It checks all 6,144 consolidated records (8 direct-QA tracks and 4 extraction/pipeline tracks), including exact-field and calculator replay. It does not run models. Original model artifacts and source SHA-256 values are included for inspection.

## Run new inference

The package supports new inference through generic adapters. The original study runners below preserve the historical setups:

- `reports/xerj_reader_2026-11-14/run_comparative_suite.py`: Reader-1/API and local LM Studio runners; see `--help` and `model_adapters.py`.
- `reports/xerj_reader_2026-11-14/rerun_qwen08_http500.py`: the explicitly labelled 53-request recovery pass.
- `reports/xerj_reader_2026-11-14/run_tiny_specialists.py`: native SmolLM2/NuExtract Tiny tracks.
- `reports/specialist_extraction_2026-11-15/run_gliner_base.py`: native GLiNER CPU extraction; source-order canonicalization is documented separately.
- `reports/nuextract3_2026-11-15/run_nuextract3.py`: NuExtract3 native-template direct/extraction tracks.

Inspect each runner and its saved metadata before running. Several use machine-specific weight/environment paths; configure those for the downloaded model revision and your machine. MLX runners require compatible Apple Silicon hardware and runtimes. Local API runs require the matching LM Studio checkpoint and server. Reader-1 requires an XERJ API key and available service; Gemini requires a Google key and sufficient quota. API keys, weights, virtual environments and private email content are not bundled.

Fixed cloud models, parameter-matched checkpoints, cross-platform timing parity, and bit-for-bit regeneration of all original responses are not guaranteed. Some earlier runs lack startup weight hashes; the NuExtract3 runner declares a revision but originally loaded by repository name. Use a pinned local snapshot and isolated output path for a new run; validate settings before resuming. Keep new results separate from original artifacts.

`manifest.json` lists source file SHA-256 values. `package_audit.json` records the wheel inspection. `report.html` is self-contained and searchable; the two-page PDF is the concise overview. The archive is a study artifact bundle, not an installable Télos package release.

## Rebuild the display

Use the current Télos source checkout with its optional `eval-pdf` dependencies and Chromium installed. Run `build_report.py` in this directory. It converts the saved records into `viewer.json`, then calls `telos.eval.display` for both HTML and PDF. There is no separate PDF layout. `--html-only` skips PDF export.
