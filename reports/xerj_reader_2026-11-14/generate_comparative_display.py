#!/usr/bin/env python3
"""Generate standalone comparative HTML evaluation dashboard.

Produces a self-contained, offline HTML report featuring:
- Warm cream/parchment palette and restrained typography
- Embedded base64 Télos logo
- Detailed model cards with checkpoint identifiers, quantization, and runtimes
- Disaggregated reporting across combined 512, inspected 240, and new 272 subsets
- Category comparison matrix across all 12 categories
- Case-level failure inspection and character-offset evidence highlighting
"""
import base64
import json
from pathlib import Path
from typing import Any


def load_suite_cases(root: Path) -> dict[str, dict[str, Any]]:
    """Load cases from suite_512_cases.json indexed by id."""
    cases_path = root / "reports/xerj_reader_2026-11-14/suite_512_cases.json"
    if not cases_path.exists():
        return {}
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    return {str(c["id"]): c for c in cases}


def load_all_runs(root: Path) -> dict[str, Any]:
    """Merge model runs from comparative_512_v2_run.json, v1, or constituent run files."""
    r_dir = root / "reports/xerj_reader_2026-11-14"
    for cand in ["comparative_512_v2_run.json", "comparative_512_run.json"]:
        merged_path = r_dir / cand
        if merged_path.exists():
            try:
                return json.loads(merged_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass

    merged: dict[str, Any] = {"benchmark_meta": {}, "model_runs": {}}
    sources = ["comparative_512_xerj.json", "comparative_512_local.json", "comparative_512_gemini.json"]
    for src in sources:
        p = r_dir / src
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "benchmark_meta" in data and not merged["benchmark_meta"]:
            merged["benchmark_meta"] = data["benchmark_meta"]
        for m_id, m_run in data.get("model_runs", {}).items():
            if m_id == "gemini-3.5-flash-lite" and src == "comparative_512_local.json":
                continue
            merged["model_runs"][m_id] = m_run
    return merged


MODEL_METADATA = {
    "xerj-reader-1": {
        "display_name": "XERJ Reader-1",
        "tag": "API Baseline",
        "checkpoint": "xerj-reader-1",
        "quantization": "FP16 / BF16 (Hosted)",
        "runtime": "XERJ Platform API (https://platform.xerj.ai/v1)",
        "architecture": "Document Reader with Symbolic Python Executor Replay (1/4 sample)",
        "cost_model": "$0.01 per 1M input tokens; execution free",
    },
    "gemini-3.5-flash-lite": {
        "display_name": "Google Gemini 3.5 Flash-Lite (Incomplete)",
        "tag": "Incomplete (107/512)",
        "checkpoint": "gemini-3.5-flash-lite",
        "quantization": "Hosted Cloud Service",
        "runtime": "Google Generative Language API",
        "architecture": "Multimodal Flash LLM with Reconstructed Quote Grounding (Incomplete: 107/512 completed; 25 quota errors; 380 unattempted due to 500 RPD daily free-tier cap; 106/107 correct)",
        "cost_model": "Google AI Studio free tier (500 RPD cap hit)",
    },
    "ministral-3-8b-instruct-2512": {
        "display_name": "Ministral 3 8B Instruct",
        "tag": "Local 8B",
        "checkpoint": "ministral-3-8b-instruct-2512",
        "quantization": "4-bit Quantized (GGUF Q4_K_M)",
        "runtime": "LM Studio Local Endpoint (http://localhost:1234/v1)",
        "architecture": "Dense 8B Transformer with JSON Schema Constrained Decoding",
        "cost_model": "Local compute (Apple Silicon / Metal)",
    },
    "ternary-bonsai-4b-mlx": {
        "display_name": "Ternary Bonsai 4B MLX",
        "tag": "1.58-bit Local",
        "checkpoint": "ternary-bonsai-4b-mlx",
        "quantization": "1.58-bit Ternary Weights / MLX 4-bit",
        "runtime": "LM Studio Local Endpoint (http://localhost:1234/v1)",
        "architecture": "BitNet Ternary Architecture for extreme edge efficiency",
        "cost_model": "Local compute (Apple Silicon / Metal)",
    },
    "google/gemma-4-e4b": {
        "display_name": "Google Gemma 4 E4B",
        "tag": "Local 4B",
        "checkpoint": "google/gemma-4-e4b",
        "quantization": "4-bit Quantized (MLX / GGUF)",
        "runtime": "LM Studio Local Endpoint (http://localhost:1234/v1)",
        "architecture": "Gemma 4 Edge Architecture with Structured JSON Output",
        "cost_model": "Local compute (Apple Silicon / Metal)",
    },
    "lfm3-omni-2b": {
        "display_name": "LiquidAI LFM3 Omni 2B",
        "tag": "Local 2B MLX",
        "checkpoint": "LiquidAI/LFM3-Omni-2B-MLX-4bit",
        "quantization": "4-bit MLX (Direct Weights)",
        "runtime": "Direct MLX Resident Subprocess Worker (testrepos/mlx-vlm)",
        "architecture": "Non-Transformer Liquid Foundation Model; Autoregressive Greedy",
        "cost_model": "Local compute (Apple Silicon / Metal)",
    },
}


def build_normalized_payload(benchmark_data: dict[str, Any], case_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Assemble clean, normalized JSON data structure for client dashboard."""
    model_runs = benchmark_data.get("model_runs", {})
    categories = [
        "arithmetic", "conflicts", "corrections", "dates",
        "irrelevant_instructions", "lists", "missing", "percentages",
        "scalar", "table_lookup", "unicode", "units"
    ]

    models_summary = []
    cases_rows = []

    for m_id, r in model_runs.items():
        meta = MODEL_METADATA.get(m_id, {
            "display_name": m_id,
            "tag": "Model",
            "checkpoint": m_id,
            "quantization": "Unknown",
            "runtime": "Local / API",
            "architecture": "Standard",
            "cost_model": "Unspecified",
        })

        summary = r.get("summary", {})
        warmup = r.get("warm_up_notes") or {}
        replay_info = r.get("replay_evaluation") or {}

        models_summary.append({
            "id": m_id,
            "name": meta["display_name"],
            "tag": meta["tag"],
            "checkpoint": meta["checkpoint"],
            "quantization": meta["quantization"],
            "runtime": meta["runtime"],
            "architecture": meta["architecture"],
            "cost_model": meta["cost_model"],
            "warmup": warmup,
            "replay": replay_info,
            "subsets": {
                "combined_512": summary.get("combined_512", {}),
                "inspected_240": summary.get("inspected_240", {}),
                "new_272": summary.get("new_272", {}),
            },
        })

        # Process case outcomes
        for item in r.get("cases", []):
            cid = str(item.get("case_id"))
            case_def = case_map.get(cid, {})
            grade = item.get("grade") or {}
            resp = item.get("response") or {}
            doc_text = case_def.get("document", "")

            # Expected answer or status
            expected = case_def.get("expected")
            if expected is None:
                expected = case_def.get("statuses")

            cases_rows.append({
                "id": cid,
                "model": m_id,
                "model_name": meta["display_name"],
                "category": item.get("category", case_def.get("category", "general")),
                "subset": item.get("subset", case_def.get("subset", "unknown")),
                "question": case_def.get("question", ""),
                "correct": grade.get("answer_correct"),
                "classification": grade.get("outcome_classification", "unknown"),
                "status": resp.get("status", "ERROR" if item.get("error") else "UNKNOWN"),
                "value": resp.get("value"),
                "value_text": resp.get("value_text"),
                "expected": expected,
                "confidence": resp.get("confidence"),
                "latency_ms": item.get("latency_ms"),
                "client_ms": item.get("client_ms"),
                "evidence": resp.get("evidence", []),
                "evidence_valid": grade.get("evidence_offsets_valid"),
                "has_evidence": grade.get("answer_has_evidence"),
                "document": doc_text,
                "program": item.get("symbolic_formula") or resp.get("symbolic", {}).get("program_compact", ""),
                "replay_matches": item.get("replay_matches"),
                "error": item.get("error") or resp.get("reason"),
            })

    return {
        "schema": "telos.evaluation.comparative.v1",
        "title": "Reader-1 comparative evaluation",
        "date": "14 November 2026",
        "benchmark_meta": benchmark_data.get("benchmark_meta", {}),
        "categories": categories,
        "models": models_summary,
        "cases": cases_rows,
        "note": "Controlled comparative diagnostics across 12 structured document extraction categories. Synthetic diagnostics; not an estimate of general web accuracy.",
    }


def generate_html(payload: dict[str, Any], logo_base64: str) -> str:
    """Generate the full, self-contained HTML page."""
    # Escape JSON payload safely to prevent parsing boundary breaks
    json_str = (
        json.dumps(payload, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>Reader-1 comparative evaluation · Télos</title>
<style>
:root {{
  --page: #FDF9F5;
  --card: #F4EBE1;
  --inner: #EFE5D9;
  --border: #E8DFD5;
  --ink: #2D2A26;
  --muted: #5C554D;
  --rust: #9B5B3E;
  --green: #23765D;
  --red: #A33F36;
  --blue: #2563AB;
  --card-radius: 10px;
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
  margin: 0;
  background: var(--page);
  color: var(--ink);
  font-family: Outfit, "Avenir Next", "Helvetica Neue", Arial, sans-serif;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}}
header {{
  max-width: 1480px;
  margin: auto;
  padding: 28px 48px 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--border);
}}
.brand {{
  display: flex;
  align-items: center;
  gap: 22px;
  text-decoration: none;
  color: inherit;
}}
.brand img {{
  width: 132px;
  height: 58px;
  object-fit: contain;
  mix-blend-mode: multiply;
}}
.brand-title {{
  border-left: 1px solid #d3c4b5;
  padding-left: 20px;
  display: flex;
  flex-direction: column;
}}
.brand-title strong {{
  font-size: 15px;
  font-weight: 500;
  letter-spacing: -0.01em;
}}
.brand-title span {{
  font-size: 11px;
  color: var(--muted);
}}
nav {{ display: flex; gap: 8px; }}
button, select, input {{
  font: inherit;
  color: var(--ink);
}}
button {{
  cursor: pointer;
  background: transparent;
  border: 1px solid #d3c4b5;
  border-radius: 6px;
  padding: 8px 14px;
  font-size: 12.5px;
  transition: background 0.15s ease;
}}
button:hover {{ background: var(--inner); }}
button:focus-visible, select:focus-visible, input:focus-visible {{
  outline: 2px solid var(--rust);
  outline-offset: 2px;
}}
main {{
  max-width: 1480px;
  margin: auto;
  padding: 34px 48px 60px;
}}
.eyebrow {{
  font-size: 11px;
  letter-spacing: 0.14em;
  font-weight: 600;
  color: var(--rust);
  text-transform: uppercase;
  margin: 0 0 10px;
}}
.title-row {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 20px;
  flex-wrap: wrap;
}}
h1 {{
  font-size: clamp(28px, 3.6vw, 44px);
  font-weight: 500;
  letter-spacing: -0.035em;
  line-height: 1.15;
  margin: 0 0 10px;
}}
.lede {{
  font-size: 15px;
  color: var(--muted);
  max-width: 980px;
  margin: 0 0 24px;
}}
.subset-tabs {{
  display: inline-flex;
  background: var(--card);
  padding: 4px;
  border-radius: 8px;
  border: 1px solid var(--border);
  gap: 4px;
  margin-bottom: 28px;
}}
.subset-tab {{
  border: 0;
  border-radius: 6px;
  padding: 8px 18px;
  font-size: 13px;
  font-weight: 500;
  color: var(--muted);
  background: transparent;
}}
.subset-tab.active {{
  background: var(--page);
  color: var(--ink);
  box-shadow: 0 1px 3px rgba(45, 42, 38, 0.08);
}}
.metrics-strip {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  border: 1px solid var(--border);
  background: var(--card);
  border-radius: var(--card-radius);
  overflow: hidden;
  margin-bottom: 34px;
}}
.metric-box {{
  padding: 20px 24px;
  border-right: 1px solid #dfd1c3;
}}
.metric-box:last-child {{ border-right: 0; }}
.metric-label {{ font-size: 11.5px; color: var(--muted); margin-bottom: 4px; }}
.metric-val {{ font-size: 32px; font-weight: 500; letter-spacing: -0.03em; line-height: 1.2; margin-bottom: 4px; }}
.metric-sub {{ font-size: 11.5px; color: var(--muted); }}

h2 {{
  font-size: 22px;
  font-weight: 500;
  letter-spacing: -0.02em;
  margin: 0 0 14px;
}}
.section-desc {{
  font-size: 13px;
  color: var(--muted);
  margin: -8px 0 20px;
}}

.models-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 20px;
  margin-bottom: 44px;
}}
.model-card {{
  border: 1px solid var(--border);
  border-radius: var(--card-radius);
  background: var(--card);
  padding: 22px 24px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}}
.model-head {{
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 12px;
}}
.model-name {{
  font-size: 18px;
  font-weight: 500;
  letter-spacing: -0.015em;
  margin: 0;
}}
.badge {{
  font-size: 11px;
  font-weight: 500;
  padding: 3px 8px;
  border-radius: 4px;
  background: var(--inner);
  border: 1px solid #dcd1c3;
  color: var(--muted);
  white-space: nowrap;
}}
.model-stats {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin: 14px 0 18px;
  padding: 14px 0;
  border-top: 1px solid #dfd1c3;
  border-bottom: 1px solid #dfd1c3;
}}
.stat-box .lbl {{ font-size: 11px; color: var(--muted); }}
.stat-box .val {{ font-size: 24px; font-weight: 500; letter-spacing: -0.02em; margin-top: 2px; }}
.stat-box .sub {{ font-size: 11px; color: var(--muted); margin-top: 1px; }}
.model-details {{
  font-size: 11.5px;
  color: var(--muted);
  line-height: 1.6;
  display: grid;
  gap: 4px;
}}
.model-details span strong {{ color: var(--ink); font-weight: 500; }}

.category-section {{
  border: 1px solid var(--border);
  border-radius: var(--card-radius);
  background: var(--card);
  padding: 26px 28px;
  margin-bottom: 44px;
}}
.matrix-wrap {{
  overflow-x: auto;
  margin-top: 18px;
}}
table.matrix {{
  width: 100%;
  border-collapse: collapse;
  font-size: 12.5px;
  min-width: 820px;
}}
table.matrix th {{
  text-align: left;
  font-size: 11.5px;
  font-weight: 500;
  color: var(--muted);
  padding: 10px 14px;
  border-bottom: 1px solid #d3c4b5;
  background: var(--card);
}}
table.matrix th.num, table.matrix td.num {{
  text-align: right;
}}
table.matrix td {{
  padding: 11px 14px;
  border-bottom: 1px solid #dfd1c3;
  vertical-align: middle;
}}
.mini-bar {{
  display: inline-block;
  height: 6px;
  border-radius: 2px;
  background: var(--rust);
  vertical-align: middle;
  margin-left: 8px;
}}
.acc-cell {{
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 6px;
}}

.explorer-section {{
  border: 1px solid var(--border);
  border-radius: var(--card-radius);
  background: var(--card);
  padding: 26px 28px;
}}
.filter-bar {{
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  align-items: flex-end;
  padding: 16px 0;
  border-top: 1px solid #dfd1c3;
  border-bottom: 1px solid #dfd1c3;
  margin: 14px 0 18px;
}}
.filter-item {{
  display: grid;
  gap: 5px;
  font-size: 11px;
  color: var(--muted);
  min-width: 130px;
  flex: 1;
}}
.filter-item.search-item {{
  flex: 2.2;
  min-width: 240px;
}}
input, select {{
  background: var(--page);
  border: 1px solid #d3c4b5;
  border-radius: 6px;
  padding: 8px 10px;
  font-size: 12.5px;
  min-height: 38px;
}}
.table-meta {{
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: var(--muted);
  margin-bottom: 12px;
}}
.cases-wrap {{
  overflow-x: auto;
  max-height: 720px;
  border: 1px solid #dfd1c3;
  border-radius: 6px;
}}
table.cases-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 12.5px;
  min-width: 900px;
  background: var(--page);
}}
table.cases-table th {{
  position: sticky;
  top: 0;
  background: var(--card);
  padding: 10px 12px;
  font-size: 11px;
  font-weight: 500;
  color: var(--muted);
  border-bottom: 1px solid #d3c4b5;
  text-align: left;
  z-index: 1;
}}
table.cases-table td {{
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}}
.btn-case {{
  border: 0;
  background: none;
  padding: 0;
  text-align: left;
  font-size: 12.5px;
  line-height: 1.4;
  color: var(--ink);
}}
.btn-case:hover {{ text-decoration: underline; }}
.case-sub {{
  display: block;
  font-size: 10.5px;
  color: var(--muted);
  margin-top: 3px;
}}
.tag-outcome {{
  display: inline-block;
  padding: 2px 7px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
}}
.tag-outcome.pass {{ background: #e0ebe2; color: #235c47; }}
.tag-outcome.fail {{ background: #f3ddd6; color: #8d382d; }}
.tag-outcome.error {{ background: #eae3d9; color: #5c554d; }}

dialog#detail {{
  background: var(--page);
  color: var(--ink);
  border: 1px solid #c9b9a9;
  border-radius: 12px;
  width: min(960px, 94vw);
  max-height: 90vh;
  padding: 28px;
  overflow-y: auto;
}}
dialog::backdrop {{
  background: rgba(45, 42, 38, 0.4);
}}
.dialog-header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--border);
  padding-bottom: 14px;
  margin-bottom: 18px;
}}
.detail-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 16px;
}}
.detail-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 16px;
}}
.detail-card h3 {{
  margin: 0 0 8px;
  font-size: 11.5px;
  font-weight: 500;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}}
pre {{
  font: 11.5px/1.7 ui-monospace, SFMono-Regular, Consolas, monospace;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
}}
mark.quote {{
  background: #f1cfb3;
  color: var(--ink);
  border-bottom: 1.5px solid var(--rust);
  padding: 0 2px;
  border-radius: 2px;
}}
footer {{
  margin-top: 50px;
  padding-top: 24px;
  border-top: 1px solid var(--border);
  font-size: 11.5px;
  color: var(--muted);
  display: flex;
  justify-content: space-between;
  gap: 20px;
  flex-wrap: wrap;
}}
@media (max-width: 900px) {{
  header, main {{ padding-left: 20px; padding-right: 20px; }}
  .metrics-strip {{ grid-template-columns: 1fr 1fr; }}
  .metric-box:nth-child(2) {{ border-right: 0; }}
  .metric-box:nth-child(-n+2) {{ border-bottom: 1px solid #dfd1c3; }}
  .detail-grid {{ grid-template-columns: 1fr; }}
}}
@media print {{
  body {{ background: #fff; }}
  header, main {{ padding: 10px; max-width: none; }}
  nav, .subset-tabs, .filter-bar, footer button {{ display: none; }}
  dialog {{ display: none; }}
}}
</style>
</head>
<body>

<header>
  <a class="brand" href="#top" aria-label="Télos Evaluation Home">
    <img src="data:image/png;base64,{logo_base64}" alt="Télos Logo">
    <div class="brand-title">
      <strong>Reader-1 Comparative Evaluation</strong>
      <span>Télos Research · 14 November 2026</span>
    </div>
  </a>
  <nav aria-label="Dashboard actions">
    <button id="btn-export-json">Export JSON</button>
    <button id="btn-export-csv">Export Visible CSV</button>
    <button id="btn-print">Print ↗</button>
  </nav>
</header>

<main id="top">
  <div class="eyebrow">Benchmark Diagnostics</div>
  <div class="title-row">
    <h1>Comparative Evaluation Display</h1>
  </div>
  <p class="lede">
    Rigorous evaluation of structured document extraction across 6 model architectures on a deterministic 512-case frozen test suite.
    Includes exact character-offset evidence verification, execution replay, and disaggregated subset performance.
  </p>

  <!-- Subset selector tabs -->
  <div class="subset-tabs" role="tablist" aria-label="Subset selector">
    <button class="subset-tab active" data-subset="combined_512">All Cases (512 total)</button>
    <button class="subset-tab" data-subset="inspected_240">Historical Subset (240 cases)</button>
    <button class="subset-tab" data-subset="new_272">New Expansion Subset (272 cases)</button>
  </div>

  <!-- Executive Summary Metrics -->
  <section class="metrics-strip" id="top-metrics" aria-label="Summary metrics"></section>

  <!-- Model Cards Section -->
  <h2>Evaluated Models & Runtimes</h2>
  <p class="section-desc">Detailed hardware, runtime configuration, quantization formats, and observed performance across subsets.</p>
  <section class="models-grid" id="models-container"></section>

  <!-- Category Comparison Section -->
  <section class="category-section">
    <h2>Category Comparison (All 12 Categories)</h2>
    <p class="section-desc">Accuracy breakdown across extraction capabilities, multi-condition lookups, conflict handling, and arithmetic.</p>
    <div class="matrix-wrap">
      <table class="matrix" id="category-table">
        <thead>
          <tr id="category-head-row">
            <th scope="col">Category</th>
          </tr>
        </thead>
        <tbody id="category-body"></tbody>
      </table>
    </div>
  </section>

  <!-- Case Explorer Section -->
  <section class="explorer-section">
    <h2>Case Explorer & Failure Inspection</h2>
    <p class="section-desc">Filter by model, evaluation outcome, category, or subset to inspect exact outputs, source evidence, and formulas.</p>

    <div class="filter-bar">
      <div class="filter-item search-item">
        <label for="filter-search">Search query</label>
        <input id="filter-search" type="search" placeholder="Filter by question, ID, answer, formula, or snippet…">
      </div>
      <div class="filter-item">
        <label for="filter-model">Model</label>
        <select id="filter-model"><option value="">All Models</option></select>
      </div>
      <div class="filter-item">
        <label for="filter-outcome">Outcome</label>
        <select id="filter-outcome">
          <option value="">All Outcomes</option>
          <option value="fail">Incorrect / Failures</option>
          <option value="pass">Correct</option>
          <option value="error">Errors / Transport</option>
        </select>
      </div>
      <div class="filter-item">
        <label for="filter-category">Category</label>
        <select id="filter-category"><option value="">All Categories</option></select>
      </div>
      <div class="filter-item">
        <label for="filter-subset">Subset</label>
        <select id="filter-subset">
          <option value="">Current Selection</option>
          <option value="inspected_240">Inspected (240)</option>
          <option value="new_272">New (272)</option>
        </select>
      </div>
      <button id="btn-reset-filters">Reset Filters</button>
    </div>

    <div class="table-meta">
      <span id="case-count-display">Loading cases…</span>
    </div>

    <div class="cases-wrap">
      <table class="cases-table">
        <thead>
          <tr>
            <th scope="col" style="width: 32%;">Case / Question</th>
            <th scope="col" style="width: 14%;">Model</th>
            <th scope="col" style="width: 12%;">Category</th>
            <th scope="col" style="width: 11%;">Outcome</th>
            <th scope="col" style="width: 15%;">Observed vs Expected</th>
            <th scope="col" style="width: 8%;">Evidence</th>
            <th scope="col" style="width: 8%;">Client Time</th>
          </tr>
        </thead>
        <tbody id="cases-body"></tbody>
      </table>
    </div>
  </section>

  <footer>
    <span>Controlled synthetic diagnostics; not an estimate of production accuracy on general documents.</span>
    <span>Frozen SHA-256: <code>7aa446c23aa360a783b2ba5935e98e9305d67528ea55a7fea9a8f4cd30db529a</code></span>
  </footer>
</main>

<!-- Case Inspection Modal -->
<dialog id="detail">
  <div class="dialog-header">
    <span class="eyebrow" id="modal-eyebrow">Case Inspection</span>
    <button id="modal-close" aria-label="Close dialog">Close ✕</button>
  </div>
  <h2 id="modal-title" style="margin-top: 0; line-height: 1.35;"></h2>

  <div class="detail-grid">
    <div class="detail-card">
      <h3>Expected Target</h3>
      <pre id="modal-expected"></pre>
    </div>
    <div class="detail-card">
      <h3>Observed Model Response</h3>
      <pre id="modal-observed"></pre>
    </div>
  </div>

  <div class="detail-card" id="modal-formula-card" style="margin-bottom: 16px;" hidden>
    <h3>Generated Formula / Program</h3>
    <pre id="modal-formula"></pre>
  </div>

  <div class="detail-card" id="modal-doc-card" style="margin-bottom: 16px;">
    <h3>Source Document · Exact Quoted Evidence Highlighted</h3>
    <pre id="modal-document"></pre>
  </div>

  <div class="detail-grid">
    <div class="detail-card">
      <h3>Evidence Character Offsets</h3>
      <pre id="modal-evidence"></pre>
    </div>
    <div class="detail-card">
      <h3>Verification & Diagnostics</h3>
      <pre id="modal-verification"></pre>
    </div>
  </div>
</dialog>

<script id="report-data" type="application/json">{json_str}</script>

<script>
'use strict';
const report = JSON.parse(document.getElementById('report-data').textContent);
const byId = id => document.getElementById(id);
const make = (tag, text, cls) => {{
  const el = document.createElement(tag);
  if (text !== undefined && text !== null) el.textContent = text;
  if (cls) el.className = cls;
  return el;
}};

let currentSubset = 'combined_512';
let filteredCases = report.cases;

// Initialize model and category filter dropdowns
const modelSelect = byId('filter-model');
for (const m of report.models) {{
  const opt = make('option', m.name);
  opt.value = m.id;
  modelSelect.append(opt);
}}

const catSelect = byId('filter-category');
for (const c of report.categories) {{
  const opt = make('option', c.replace('_', ' '));
  opt.value = c;
  catSelect.append(opt);
}}

// Tab switching
document.querySelectorAll('.subset-tab').forEach(tab => {{
  tab.addEventListener('click', () => {{
    document.querySelectorAll('.subset-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    currentSubset = tab.dataset.subset;
    renderAll();
  }});
}});

function renderMetrics() {{
  const topBar = byId('top-metrics');
  topBar.replaceChildren();

  // Find best accuracy model for current subset
  let bestModel = null;
  let bestAcc = -1;
  let fastestModel = null;
  let fastestLat = Infinity;

  for (const m of report.models) {{
    const s = m.subsets[currentSubset] || {{}};
    const acc = s.accuracy_pct || 0;
    const lat = (s.latencies_ms && s.latencies_ms.median) || Infinity;
    if (acc > bestAcc) {{ bestAcc = acc; bestModel = m; }}
    if (lat < fastestLat) {{ fastestLat = lat; fastestModel = m; }}
  }}

  const totalCount = currentSubset === 'inspected_240' ? 240 : currentSubset === 'new_272' ? 272 : 512;

  const m1 = make('div', null, 'metric-box');
  m1.append(make('div', 'Evaluation Dataset', 'metric-label'),
            make('div', totalCount + ' Cases', 'metric-val'),
            make('div', '12 categories · Frozen SHA-256', 'metric-sub'));

  const m2 = make('div', null, 'metric-box');
  m2.append(make('div', 'Evaluated Architectures', 'metric-label'),
            make('div', report.models.length + ' Models', 'metric-val'),
            make('div', 'Cloud APIs + Local 1.58b/4b/8b MLX', 'metric-sub'));

  const m3 = make('div', null, 'metric-box');
  m3.append(make('div', 'Highest Overall Accuracy', 'metric-label'),
            make('div', (bestAcc >= 0 ? bestAcc.toFixed(1) + '%' : '—'), 'metric-val'),
            make('div', bestModel ? bestModel.name : '', 'metric-sub'));

  const m4 = make('div', null, 'metric-box');
  m4.append(make('div', 'Lowest Response Latency', 'metric-label'),
            make('div', (fastestLat < Infinity ? fastestLat.toFixed(0) + ' ms' : '—'), 'metric-val'),
            make('div', fastestModel ? fastestModel.name : '', 'metric-sub'));

  topBar.append(m1, m2, m3, m4);
}}

function renderModelCards() {{
  const container = byId('models-container');
  container.replaceChildren();

  for (const m of report.models) {{
    const s = m.subsets[currentSubset] || {{}};
    const acc = s.accuracy_pct !== undefined ? s.accuracy_pct.toFixed(1) + '%' : '—';
    const corr = s.correct_count !== undefined ? s.correct_count : '—';
    const tot = s.total_cases !== undefined ? s.total_cases : '—';
    const lat = s.latencies_ms && s.latencies_ms.median ? s.latencies_ms.median.toFixed(0) + ' ms' : '—';
    const clientLat = s.latencies_ms && s.latencies_ms.client_median ? (s.latencies_ms.client_median / 1000).toFixed(2) + 's' : '—';
    const validOffsets = s.evidence && s.evidence.valid_offsets_pct !== undefined ? s.evidence.valid_offsets_pct.toFixed(1) + '%' : '—';

    const card = make('div', null, 'model-card');

    const head = make('div', null, 'model-head');
    const headText = make('div');
    headText.append(make('h3', m.name, 'model-name'),
                    make('span', m.checkpoint, 'case-sub'));
    const badge = make('span', m.tag, 'badge');
    head.append(headText, badge);

    const stats = make('div', null, 'model-stats');
    const st1 = make('div', null, 'stat-box');
    st1.append(make('div', 'Accuracy (' + currentSubset.replace('_', ' ') + ')', 'lbl'),
               make('div', acc, 'val'),
               make('div', `${{corr}} / ${{tot}} correct`, 'sub'));

    const st2 = make('div', null, 'stat-box');
    st2.append(make('div', 'Median Latency', 'lbl'),
               make('div', lat, 'val'),
               make('div', `Client: ${{clientLat}}`, 'sub'));
    stats.append(st1, st2);

    const details = make('div', null, 'model-details');
    details.innerHTML = `
      <span><strong>Runtime:</strong> ${{htmlEscape(m.runtime)}}</span>
      <span><strong>Quantization:</strong> ${{htmlEscape(m.quantization)}}</span>
      <span><strong>Offset Validity:</strong> ${{validOffsets}}</span>
      ${{m.replay && m.replay.sampled_replays ? `<span><strong>Executor Replay:</strong> ${{m.replay.matched_replays}}/${{m.replay.sampled_replays}} matches (${{m.replay.match_pct}}%)</span>` : ''}}
      <span><strong>Economics:</strong> ${{htmlEscape(m.cost_model)}}</span>
    `;

    card.append(head, stats, details);
    container.append(card);
  }}
}}

function renderCategoryTable() {{
  const headRow = byId('category-head-row');
  headRow.replaceChildren(make('th', 'Category'));
  for (const m of report.models) {{
    const th = make('th', m.name, 'num');
    headRow.append(th);
  }}

  const body = byId('category-body');
  body.replaceChildren();

  for (const cat of report.categories) {{
    const tr = make('tr');
    tr.append(make('td', cat.replace('_', ' ')));

    for (const m of report.models) {{
      const catData = (m.subsets[currentSubset] && m.subsets[currentSubset].categories && m.subsets[currentSubset].categories[cat]) || {{}};
      const acc = catData.accuracy_pct !== undefined ? catData.accuracy_pct : null;
      const corr = catData.correct !== undefined ? catData.correct : 0;
      const tot = catData.total !== undefined ? catData.total : 0;

      const td = make('td', null, 'num');
      if (acc !== null) {{
        const cell = make('div', null, 'acc-cell');
        cell.append(make('span', `${{acc.toFixed(0)}}% (${{corr}}/${{tot}})`));
        const bar = make('span', null, 'mini-bar');
        bar.style.width = Math.max(3, Math.round(acc * 0.4)) + 'px';
        if (acc >= 80) bar.style.background = 'var(--green)';
        else if (acc >= 50) bar.style.background = 'var(--rust)';
        else bar.style.background = 'var(--red)';
        cell.append(bar);
        td.append(cell);
      }} else {{
        td.textContent = '—';
      }}
      tr.append(td);
    }}
    body.append(tr);
  }}
}}

function renderCases() {{
  const q = byId('filter-search').value.trim().toLowerCase();
  const fModel = byId('filter-model').value;
  const fOutcome = byId('filter-outcome').value;
  const fCat = byId('filter-category').value;
  const fSub = byId('filter-subset').value;

  filteredCases = report.cases.filter(c => {{
    // Subset filter logic
    if (fSub) {{
      if (c.subset !== fSub) return false;
    }} else if (currentSubset !== 'combined_512') {{
      if (c.subset !== currentSubset) return false;
    }}
    if (fModel && c.model !== fModel) return false;
    if (fCat && c.category !== fCat) return false;

    // Outcome filter
    if (fOutcome === 'pass' && c.correct !== true) return false;
    if (fOutcome === 'fail' && c.correct !== false) return false;
    if (fOutcome === 'error' && c.status !== 'ERROR' && c.correct !== null) return false;

    if (q) {{
      const textBlock = `${{c.id}} ${{c.question}} ${{c.document}} ${{c.value_text || ''}} ${{c.expected || ''}} ${{c.program || ''}}`.toLowerCase();
      if (!textBlock.includes(q)) return false;
    }}
    return true;
  }});

  byId('case-count-display').textContent = `Showing ${{filteredCases.length}} of ${{report.cases.length}} cases`;

  const tbody = byId('cases-body');
  tbody.replaceChildren();

  // Render first 250 visible cases to prevent DOM overload
  const displaySlice = filteredCases.slice(0, 250);
  for (const c of displaySlice) {{
    const tr = make('tr');

    // Case / Question
    const tdQ = make('td');
    const btn = make('button', c.question, 'btn-case');
    btn.addEventListener('click', () => openCaseModal(c));
    tdQ.append(btn);
    const sub = make('span', `${{c.id}} · ${{c.subset}}`, 'case-sub');
    tdQ.append(sub);
    tr.append(tdQ);

    // Model
    tr.append(make('td', c.model_name));

    // Category
    tr.append(make('td', c.category.replace('_', ' ')));

    // Outcome Badge
    const tdOut = make('td');
    let outClass = 'error';
    let outText = c.status || 'ERROR';
    if (c.correct === true) {{
      outClass = 'pass';
      outText = 'Correct';
    }} else if (c.correct === false) {{
      outClass = 'fail';
      outText = 'Incorrect';
    }}
    tdOut.append(make('span', outText, `tag-outcome ${{outClass}}`));
    tr.append(tdOut);

    // Observed vs Expected
    const tdVal = make('td');
    const valDiv = make('div', null);
    valDiv.innerHTML = `<div><strong>Obs:</strong> ${{htmlEscape(String(c.value_text !== null && c.value_text !== undefined ? c.value_text : c.status))}}</div>` +
                       `<div class="case-sub"><strong>Exp:</strong> ${{htmlEscape(String(c.expected || '—'))}}</div>`;
    tdVal.append(valDiv);
    tr.append(tdVal);

    // Evidence
    const tdEv = make('td');
    if (c.evidence_valid === true) {{
      tdEv.append(make('span', 'Valid offsets', 'tag-outcome pass'));
    }} else if (c.evidence_valid === false) {{
      tdEv.append(make('span', 'Invalid offsets', 'tag-outcome fail'));
    }} else {{
      tdEv.textContent = 'None';
    }}
    tr.append(tdEv);

    // Client Latency
    const tdLat = make('td');
    tdLat.textContent = c.client_ms ? (c.client_ms / 1000).toFixed(2) + 's' : (c.latency_ms ? c.latency_ms.toFixed(0) + 'ms' : '—');
    tr.append(tdLat);

    tbody.append(tr);
  }}

  if (filteredCases.length === 0) {{
    const tr = make('tr');
    const td = make('td', 'No cases match your filters.');
    td.colSpan = 7;
    td.style.textAlign = 'center';
    td.style.padding = '30px';
    td.style.color = 'var(--muted)';
    tr.append(td);
    tbody.append(tr);
  }}
}}

function openCaseModal(c) {{
  byId('modal-eyebrow').textContent = `${{c.model_name}} · ${{c.id}} · ${{c.category.replace('_', ' ')}} (${{c.subset}})`;
  byId('modal-title').textContent = c.question;

  byId('modal-expected').textContent = typeof c.expected === 'object' ? JSON.stringify(c.expected, null, 2) : String(c.expected || '—');
  byId('modal-observed').textContent = JSON.stringify({{
    status: c.status,
    value: c.value,
    value_text: c.value_text,
    confidence: c.confidence,
    classification: c.classification,
    error: c.error,
  }}, null, 2);

  // Formula inspection
  const formulaCard = byId('modal-formula-card');
  if (c.program) {{
    formulaCard.hidden = false;
    byId('modal-formula').textContent = c.program;
  }} else {{
    formulaCard.hidden = true;
  }}

  // Document highlighting with character offsets
  const docPre = byId('modal-document');
  docPre.replaceChildren();
  if (c.document) {{
    const spans = [...(Array.isArray(c.evidence) ? c.evidence : [])]
      .filter(s => typeof s.start === 'number' && typeof s.end === 'number' && s.end > s.start)
      .sort((a, b) => a.start - b.start);

    const chars = Array.from(c.document);
    let at = 0;
    for (const span of spans) {{
      if (span.start < at || span.end > chars.length) continue;
      const before = chars.slice(at, span.start).join('');
      const quote = chars.slice(span.start, span.end).join('');
      docPre.append(document.createTextNode(before));
      const mark = make('mark', quote, 'quote');
      docPre.append(mark);
      at = span.end;
    }}
    docPre.append(document.createTextNode(chars.slice(at).join('')));
  }} else {{
    docPre.textContent = 'No document text attached.';
  }}

  byId('modal-evidence').textContent = JSON.stringify(c.evidence || [], null, 2);

  byId('modal-verification').textContent = JSON.stringify({{
    answer_correct: c.correct,
    outcome_classification: c.classification,
    evidence_offsets_valid: c.evidence_valid,
    answer_has_evidence: c.has_evidence,
    replay_matches: c.replay_matches,
    client_latency_ms: c.client_ms,
    model_latency_ms: c.latency_ms,
  }}, null, 2);

  byId('detail').showModal();
}}

byId('modal-close').addEventListener('click', () => byId('detail').close());

function htmlEscape(str) {{
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}}

// Filter event listeners
['filter-search', 'filter-model', 'filter-outcome', 'filter-category', 'filter-subset'].forEach(id => {{
  byId(id).addEventListener(id === 'filter-search' ? 'input' : 'change', renderCases);
}});

byId('btn-reset-filters').addEventListener('click', () => {{
  byId('filter-search').value = '';
  byId('filter-model').value = '';
  byId('filter-outcome').value = '';
  byId('filter-category').value = '';
  byId('filter-subset').value = '';
  renderCases();
}});

byId('btn-export-json').addEventListener('click', () => {{
  const blob = new Blob([JSON.stringify(report, null, 2)], {{ type: 'application/json' }});
  downloadBlob(blob, 'telos-reader-comparative-512.json');
}});

byId('btn-export-csv').addEventListener('click', () => {{
  const headers = ['id', 'model', 'category', 'subset', 'question', 'correct', 'status', 'value', 'expected', 'latency_ms'];
  const rows = [headers.join(',')];
  for (const c of filteredCases) {{
    const row = headers.map(h => {{
      let val = c[h];
      if (val === null || val === undefined) val = '';
      val = String(val).replace(/"/g, '""');
      return `"${{val}}"`;
    }});
    rows.push(row.join(','));
  }}
  const blob = new Blob([rows.join('\\r\\n')], {{ type: 'text/csv' }});
  downloadBlob(blob, 'telos-comparative-cases.csv');
}});

function downloadBlob(blob, filename) {{
  const url = URL.createObjectURL(blob);
  const a = make('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}}

byId('btn-print').addEventListener('click', () => window.print());

function renderAll() {{
  renderMetrics();
  renderModelCards();
  renderCategoryTable();
  renderCases();
}}

renderAll();
</script>
</body>
</html>
"""


def main():
    root = Path(__file__).resolve().parents[2]
    out_path = root / "reports/xerj_reader_2026-11-14/comparative_display.html"
    logo_path = root / "telos/eval/display_assets/logo.png"

    logo_bytes = logo_path.read_bytes() if logo_path.exists() else b""
    logo_base64 = base64.b64encode(logo_bytes).decode("ascii")

    cases_map = load_suite_cases(root)
    benchmark_data = load_all_runs(root)

    payload = build_normalized_payload(benchmark_data, cases_map)
    html_content = generate_html(payload, logo_base64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")
    print(f"Generated standalone comparative display dashboard at {out_path} ({len(html_content)} bytes)")


if __name__ == "__main__":
    main()
