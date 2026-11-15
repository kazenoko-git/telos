# Document evaluation

Télos 1.6.2 bundles `document512`: 512 synthetic document tasks, frozen operand plans, a SHA-256 manifest and normalized historical responses. It requires no repository checkout or separately downloaded question/answer data.

## Replay recorded scores offline

```bash
pip install telos-ml==1.6.2
telos eval documents replay --reference --output replay.json --html-report replay.html
telos eval documents export --output suite.json
```

Replay recomputes grades from saved responses. It does not run the original models or the provider's program executor. The bundled snapshots retain historical error/recovery labels; new runs use the prompt and attempt policy recorded in their own output. They are not bit-for-bit recreations of older inference environments.

## Run new predictions

Use any existing general-purpose adapter. For an OpenAI-compatible server:

```bash
telos eval --suite document512 --model MODEL --backend openai_api \
  --api-base http://localhost:1234/v1 --track all --max-new-tokens 256 \
  --output run.json --html-report run.html
```

API keys use the existing adapter's environment settings (`OPENAI_API_KEY` or `GEMINI_API_KEY`); never put credentials in result files or Git. Hosted providers need their own credentials and may charge for inference. Local model weights are downloaded separately.

On Apple Silicon, MLX evaluation requires Python 3.11+:

```bash
pip install 'telos-ml[eval-mlx]==1.6.2'
telos eval --suite document512 --model numind/NuExtract3-mlx-4bits \
  --backend mlx_lm --model-revision 29c38269f94054282bf9ea97a20dfc6bb8bbefea \
  --document-prompt native-template --track all --max-new-tokens 256 \
  --output nuextract3.json --html-report nuextract3.html
```

`native-template` passes the schema and instructions to a compatible tokenizer template and fails before inference if they are ignored. Other chat models should use the default `instruction` format. `--thinking` enables thinking where supported. HF models use `pip install 'telos-ml[eval-hf]==1.6.2'`, `--backend huggingface` and an appropriate device. Pin remote HF/MLX weights with `--model-revision`; local weight hashes are recorded. Generation settings, rendered prompts, completions, client timing and observable HTTP attempts are saved.

Use `--max-tasks 2` for a smoke run. `--case-ids scalar-01 ...` selects known IDs. `--resume` continues an existing output only when its model/settings/suite fingerprint matches; existing failures remain recorded. Recovery runs need a new output. Built-in HTTP adapters make one request per prediction, without hidden retries. Three consecutive backend errors stop a run and preserve its partial checkpoint. Coverage is reported separately from accuracy.

## Three separate tracks

- `direct_qa`: model supplies the final status and typed answer.
- `extraction`: every requested source field must match the frozen literal-span contract. Models receive field descriptions, not reference operands or answers.
- `calculator`: the extracted fields enter fixed, human-specified arithmetic, date, percentage and resolution operations. This is not autonomous program generation.

`--track all` shares one extraction completion between the latter two tracks. Scores are never pooled. Evidence validation proves literal matching, not that evidence entails an answer. Currency is not inferred when missing. Malformed JSON, unsupported statuses and runtime errors remain distinct outcomes.

## Import another extractor's responses

Encoders or custom runtimes can produce this provider-neutral JSON format, then use the packaged scorer without an adapter:

```json
{"records": [{"model": "my-model", "case_id": "scalar-01", "track": "direct_qa", "response": {"status": "ANSWER", "value": 19}}]}
```

For source extraction, replace `track` with `extraction` and `response` with `fields`, mapping the exported plan's field names to lists of verbatim strings. Replay produces extraction and calculator tracks independently:

```bash
telos eval documents replay predictions.json --output scored.json --html-report scored.html
```

A package run can also be replayed. Raw completions take precedence over cached parsed values and grades. Missing cases stay unattempted.

## Scope

The original 240 tasks were inspected during development; 272 were added later. This frozen set is a diagnostic suite with repeated templates, not a general-document benchmark or a continuing held-out test. Historical response replay reproduces the reported scores. Fresh inference can differ with weights, quantization, prompts, decoding, runtime and service changes. No vendor-specific endpoint adapter is included.

## Comparison viewer and PDF

The current viewer supports per-model leaderboards, separate track tabs, model/category filters, source inspection and paginated case lists. It embeds the Télos logo and Outfit font, including for offline viewing. Multiple tracks are never combined into a headline accuracy.

The PDF uses the viewer's print stylesheet, not a second report layout:

```bash
pip install '.[eval-pdf]'
playwright install chromium
telos eval report scored.json --title 'Document evaluation' --output report.html --pdf report.pdf
```

`pip install '.[eval-pdf]'` installs from the current source checkout. Browser PDF export is optional; the HTML viewer's Print action also uses the same print layout. `TELOS_CHROMIUM_EXECUTABLE` can select an existing Chrome/Chromium executable instead of a Playwright download.
