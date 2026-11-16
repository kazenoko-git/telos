# Evaluation Beautification Display

Télos 1.6.0 adds a portable evaluation notebook and live terminal progress. The notebook follows the warm ivory, parchment, espresso and minimal sans-serif design used in the Télos research figures. It contains its own styles, scripts and logo; opening a generated report makes no network calls.

## View a saved evaluation

```sh
telos eval report logs/evaluation.json --output reports/evaluation.html
```

The command reads saved JSON and writes one standalone HTML file. It does not load a model, rerun tasks, send data to a server, or open a browser automatically. Open the resulting file in a browser to inspect it.

For Reader-1 archives, include the cases file to show expected answers and highlight source evidence:

```sh
telos eval report reports/xerj_reader_2026-11-14/extended-run.json \
  --cases reports/xerj_reader_2026-11-14/extended_cases.json \
  --title 'Reader-1, under the microscope' \
  --output reports/reader-evaluation.html
```

Filter by model, suite, category or outcome, or search the recorded data. Select a case to compare its expected and observed values, source document, Unicode evidence offsets, formula and replay result. Summary cards and bars reflect the filtered selection, identified by the “Filtered selection” badge. Export JSON saves the normalized report; CSV exports the visible rows. Print uses a simplified layout.

Native functional, tool-use and contextual-probe reports are supported, including multi-model results, dual hint tracks, and the audited historical `results.json` archive. Summary-only reports keep their suite scores and do not invent individual outcomes. Historical saved success rates remain saved success rates; a chart does not make different budgets or retry policies comparable. Missing confidence, timing or usage is shown as not recorded.

Multiple files may be provided when their filenames have distinct stems. Keep diagnostic repeats separate from a primary evaluation when interpreting accuracy; selecting a suite makes the selection explicit. Do not treat a combined collection of selected tests as population accuracy.

## Save HTML while evaluating

```sh
telos eval --checkpoint path/to/checkpoint --mode functional --suite humaneval \
  --output logs/evaluation.json --html-report reports/evaluation.html
```

The original JSON report remains the source record. HTML is generated after evaluation finishes, without changing the scorer or the saved outcome schema.

## Terminal progress

```sh
telos eval --checkpoint path/to/checkpoint --display auto
telos eval --checkpoint path/to/checkpoint --display plain
telos eval --checkpoint path/to/checkpoint --display off
```

`auto` uses a compact line with elapsed time, estimated remaining time and completed/passed counts on a terminal. Pipes and CI receive sparse plain records. `plain` always emits those records. `off` disables the new progress display while retaining existing evaluation logs and summaries. `NO_COLOR` disables color. The progress display covers contextual probes, functional suites, and tool-use tasks. It does not change generation, scoring or sandboxing.

The CLI and programmatic report writer are both available:

```python
from telos.eval.display import write_html_report

write_html_report(saved_results, 'evaluation.html', title='My evaluation')
```

Reports can contain prompts, source documents and model responses supplied in the source files. Review their contents before sharing the generated HTML or exports.
