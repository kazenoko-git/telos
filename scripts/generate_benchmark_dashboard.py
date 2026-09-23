#!/usr/bin/env python3
"""
Télos Benchmark Comparison Dashboard Generator.

Compiles evaluation results from `logs/` into a sleek, self-contained HTML dashboard
featuring interactive Chart.js graphs, capability radar charts, a detailed leaderboard table,
and dynamic drag-and-drop JSON file upload capabilities. Includes embedded Télos branding logo.
"""

import os
import sys
import json
import base64
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGO_PATH = PROJECT_ROOT / "logos" / "telos_logo.png"
OUTPUT_HTML = PROJECT_ROOT / "reports" / "benchmark_comparison_dashboard.html"


def load_logo_base64() -> str:
    """Encodes the Télos logo as a Base64 data URI for zero-dependency standalone HTML embedding."""
    if LOGO_PATH.exists():
        with open(LOGO_PATH, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
    return ""


SUITE_KEYS = [
    "humaneval",
    "tooluse",
    "cyber",
    "gpqa_diamond",
    "mmlu_science",
    "competition_math",
    "arc"
]


def get_model_display_name(key: str, data: Dict[str, Any]) -> str:
    """Maps internal model keys or filenames to clean human-readable labels."""
    k = key.lower()
    if "afm3" in k:
        return "AFM-3 Core Advanced"
    elif "granite" in k:
        return "IBM Granite 4.2 3B"
    elif "lfm8b" in k or "lfm_8b" in k:
        return "LiquidAI LFM 8B A1B"
    elif "phi4" in k:
        return "Microsoft Phi-4 Mini"
    elif "gemma" in k:
        return "Google Gemma 4 E4B"
    elif "qwen" in k:
        return "Qwen 2.5 32B Instruct"
    
    # Fallback to model_checkpoint if present
    for v in data.values():
        if isinstance(v, dict) and "model_checkpoint" in v:
            return str(v["model_checkpoint"])
    return key.replace("_master_summary.json", "").replace("eval_report_", "").title()


def get_model_color(model_name: str, index: int) -> str:
    """Returns curated neon hex color palettes for models."""
    colors = [
        "#00F2FE",  # Cyan Glow
        "#00F5A0",  # Emerald Green
        "#FFD200",  # Amber Gold
        "#FF007F",  # Neon Pink
        "#7F00FF",  # Electric Violet
        "#00D2FF",  # Sky Blue
        "#FF5722",  # Bright Coral
    ]
    name = model_name.lower()
    if "afm-3" in name:
        return "#00F2FE"
    elif "granite" in name:
        return "#FFD200"
    elif "liquid" in name or "lfm" in name:
        return "#00F5A0"
    elif "phi" in name:
        return "#FF007F"
    elif "gemma" in name:
        return "#7F00FF"
    return colors[index % len(colors)]


def collect_evaluation_data() -> Dict[str, Any]:
    """
    Scans `logs/` for master summary files and individual benchmark JSON reports.
    Returns structured hierarchy of models, benchmark suites, and metrics.
    Excludes unfinished models that have not completed all 7 suites.
    """
    master_files = sorted(glob.glob(str(LOGS_DIR / "*master_summary.json")))
    models_data = {}

    for mf in master_files:
        path = Path(mf)
        key = path.name.replace("_master_summary.json", "").replace("eval_report_", "")
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                continue

        model_label = get_model_display_name(key, data)
        suites = {}

        for suite_key, info in data.items():
            if isinstance(info, dict) and "pass_at_1_pct" in info:
                outcomes = info.get("execution_outcomes") or {}
                ci = info.get("pass_at_1_95ci") or [0.0, 0.0]
                suites[suite_key] = {
                    "label": info.get("label", suite_key),
                    "pass_at_1_pct": float(info.get("pass_at_1_pct", 0.0)),
                    "ci_low": float(ci[0]) if len(ci) > 0 else 0.0,
                    "ci_high": float(ci[1]) if len(ci) > 1 else 0.0,
                    "total_tasks": int(info.get("total_tasks", 0)),
                    "passed_count": int(outcomes.get("PASSED", 0)),
                    "failed_assertion": int(outcomes.get("FAILED_ASSERTION", 0)),
                    "syntax_error": int(outcomes.get("SYNTAX_ERROR", 0)),
                    "timeout": int(outcomes.get("TIMEOUT", 0)),
                    "runtime_exception": int(outcomes.get("RUNTIME_EXCEPTION", 0)),
                    "ast_validity_pct": float(info.get("ast_validity_pct", 100.0)),
                }

        # Filter out unfinished models lacking all 7 suites
        completed_suites = [s for s in SUITE_KEYS if s in suites]
        if len(completed_suites) != len(SUITE_KEYS):
            print(f"ℹ Skipping unfinished model in dashboard: '{model_label}' ({len(completed_suites)}/{len(SUITE_KEYS)} suites)")
            continue

        models_data[model_label] = {
            "key": key,
            "display_name": model_label,
            "suites": suites
        }

    return models_data



def generate_html_dashboard(eval_data: Dict[str, Any], logo_b64: str) -> str:
    """Generates the single-file HTML/CSS/JS dashboard template."""
    json_payload = json.dumps(eval_data, indent=2)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TÉLOS Model Benchmark Leaderboard & Comparison Dashboard</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        :root {{
            --bg-dark: #07090E;
            --card-bg: rgba(18, 22, 33, 0.75);
            --card-border: rgba(255, 255, 255, 0.08);
            --card-hover: rgba(255, 255, 255, 0.12);
            --accent-cyan: #00F2FE;
            --accent-green: #00F5A0;
            --accent-purple: #7F00FF;
            --accent-gold: #FFD200;
            --text-main: #F0F4F8;
            --text-muted: #8E9BAE;
            --font-heading: 'Outfit', sans-serif;
            --font-body: 'Inter', sans-serif;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(0, 242, 254, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(127, 0, 255, 0.05) 0%, transparent 40%);
            color: var(--text-main);
            font-family: var(--font-body);
            line-height: 1.6;
            padding: 30px 40px;
            min-height: 100vh;
        }}

        /* Header Header */
        header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 25px;
            border-bottom: 1px solid var(--card-border);
            margin-bottom: 30px;
        }}

        .brand-container {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}

        .brand-logo {{
            height: 52px;
            width: auto;
            filter: drop-shadow(0 0 12px rgba(0, 242, 254, 0.4));
        }}

        .brand-title {{
            font-family: var(--font-heading);
            font-size: 28px;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #FFFFFF 0%, #00F2FE 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .brand-subtitle {{
            font-size: 13px;
            color: var(--text-muted);
            font-weight: 400;
            margin-top: 2px;
        }}

        .header-meta {{
            text-align: right;
        }}

        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(0, 245, 160, 0.1);
            border: 1px solid rgba(0, 245, 160, 0.3);
            color: var(--accent-green);
            font-size: 12px;
            font-weight: 600;
            padding: 6px 14px;
            border-radius: 20px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .status-dot {{
            width: 8px;
            height: 8px;
            background-color: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--accent-green);
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0% {{ transform: scale(0.95); opacity: 0.8; }}
            50% {{ transform: scale(1.15); opacity: 1; }}
            100% {{ transform: scale(0.95); opacity: 0.8; }}
        }}

        .last-updated {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 6px;
        }}

        /* Controls Section */
        .controls-card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 20px 24px;
            margin-bottom: 30px;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
        }}

        .control-group {{
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }}

        .control-label {{
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .btn-toggle {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            color: var(--text-main);
            padding: 8px 16px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .btn-toggle:hover {{
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
        }}

        .btn-toggle.active {{
            background: linear-gradient(135deg, rgba(0, 242, 254, 0.2) 0%, rgba(127, 0, 255, 0.2) 100%);
            border-color: var(--accent-cyan);
            color: #FFFFFF;
            box-shadow: 0 0 12px rgba(0, 242, 254, 0.2);
        }}

        /* Drag & Drop Upload Zone */
        .upload-zone {{
            border: 2px dashed rgba(0, 242, 254, 0.3);
            border-radius: 12px;
            padding: 10px 20px;
            background: rgba(0, 242, 254, 0.02);
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }}

        .upload-zone:hover {{
            border-color: var(--accent-cyan);
            background: rgba(0, 242, 254, 0.06);
        }}

        .upload-text {{
            font-size: 12px;
            color: var(--accent-cyan);
            font-weight: 500;
        }}

        /* Charts Grid */
        .charts-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 24px;
            margin-bottom: 30px;
        }}

        @media (max-width: 1100px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .chart-card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 24px;
            position: relative;
        }}

        .chart-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }}

        .chart-title {{
            font-family: var(--font-heading);
            font-size: 18px;
            font-weight: 700;
            color: #FFFFFF;
        }}

        .chart-container {{
            position: relative;
            height: 380px;
            width: 100%;
        }}

        /* Table Section */
        .table-card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 24px;
            overflow: hidden;
        }}

        .table-wrapper {{
            overflow-x: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 14px;
        }}

        th {{
            font-family: var(--font-heading);
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0.5px;
            padding: 14px 16px;
            border-bottom: 1px solid var(--card-border);
        }}

        td {{
            padding: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            color: var(--text-main);
        }}

        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}

        .model-cell {{
            font-weight: 700;
            font-family: var(--font-heading);
            color: #FFFFFF;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .model-color-badge {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }}

        .score-pill {{
            font-weight: 700;
            font-family: var(--font-heading);
            font-size: 15px;
            padding: 4px 12px;
            border-radius: 8px;
            display: inline-block;
        }}

        .score-high {{
            background: rgba(0, 245, 160, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(0, 245, 160, 0.3);
        }}

        .score-mid {{
            background: rgba(0, 242, 254, 0.15);
            color: var(--accent-cyan);
            border: 1px solid rgba(0, 242, 254, 0.3);
        }}

        .score-low {{
            background: rgba(255, 210, 0, 0.15);
            color: var(--accent-gold);
            border: 1px solid rgba(255, 210, 0, 0.3);
        }}

        /* Search input */
        .search-input {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            color: var(--text-main);
            padding: 8px 16px;
            border-radius: 10px;
            font-size: 13px;
            outline: none;
            width: 220px;
            transition: all 0.2s ease;
        }}

        .search-input:focus {{
            border-color: var(--accent-cyan);
            background: rgba(255, 255, 255, 0.08);
        }}
    </style>
</head>
<body>

    <!-- Header -->
    <header>
        <div class="brand-container">
            {'<img src="' + logo_b64 + '" alt="Télos Logo" class="brand-logo">' if logo_b64 else ''}
            <div>
                <h1 class="brand-title">TÉLOS EVALUATION DASHBOARD</h1>
                <p class="brand-subtitle">Multi-Model Cross-Benchmark Capabilities & Accuracy Comparison</p>
            </div>
        </div>
        <div class="header-meta">
            <div class="status-badge">
                <div class="status-dot"></div> Live Telemetry Synchronized
            </div>
            <div class="last-updated">Report Generated: {current_time}</div>
        </div>
    </header>

    <!-- Controls & Drag/Drop -->
    <section class="controls-card">
        <div class="control-group">
            <span class="control-label">Metric:</span>
            <button class="btn-toggle active" onclick="setMetric('pass_at_1_pct')">Pass@1 (%)</button>
            <button class="btn-toggle" onclick="setMetric('passed_count')">Solved Count</button>
            <button class="btn-toggle" onclick="setMetric('ast_validity_pct')">AST Valid (%)</button>
        </div>

        <div class="control-group">
            <div class="upload-zone" onclick="document.getElementById('fileInput').click()" ondragover="event.preventDefault()" ondrop="handleFileDrop(event)">
                <span class="upload-text">+ Drag & Drop JSON Report to Add Model</span>
                <input type="file" id="fileInput" accept=".json" style="display: none" onchange="handleFileSelect(event)">
            </div>
        </div>
    </section>

    <!-- Charts Grid -->
    <section class="charts-grid">
        <!-- Bar Chart -->
        <div class="chart-card">
            <div class="chart-header">
                <h2 class="chart-title">Benchmark Pass@1 Score Comparison</h2>
            </div>
            <div class="chart-container">
                <canvas id="barChart"></canvas>
            </div>
        </div>

        <!-- Radar Chart -->
        <div class="chart-card">
            <div class="chart-header">
                <h2 class="chart-title">Capability Fingerprint</h2>
            </div>
            <div class="chart-container">
                <canvas id="radarChart"></canvas>
            </div>
        </div>
    </section>

    <!-- Detailed Leaderboard Table -->
    <section class="table-card">
        <div class="chart-header">
            <h2 class="chart-title">Comprehensive Leaderboard</h2>
            <input type="text" id="tableSearch" class="search-input" placeholder="Search suite or model..." oninput="filterTable()">
        </div>
        <div class="table-wrapper">
            <table id="leaderboardTable">
                <thead>
                    <tr>
                        <th>Model / Architecture</th>
                        <th>Benchmark Suite</th>
                        <th>Pass@1 Score</th>
                        <th>95% Confidence Interval</th>
                        <th>Solved / Total</th>
                        <th>AST Validity</th>
                    </tr>
                </thead>
                <tbody id="tableBody">
                    <!-- Populated dynamically via JS -->
                </tbody>
            </table>
        </div>
    </section>

    <script>
        // Embedded Evaluation Data
        let rawData = {json_payload};

        let currentMetric = 'pass_at_1_pct';
        let barChartInst = null;
        let radarChartInst = null;

        const MODEL_COLORS = {{
            "AFM-3 Master Model": "#00F2FE",
            "IBM Granite 4.2 3B MLX": "#00F5A0",
            "Liquid LFM 8B": "#FFD200",
            "Microsoft Phi 4 Mini Reasoning": "#FF007F",
            "Google Gemma 4 E4B": "#7F00FF"
        }};

        function getColor(modelName, idx) {{
            if (MODEL_COLORS[modelName]) return MODEL_COLORS[modelName];
            const fallback = ["#00F2FE", "#00F5A0", "#FFD200", "#FF007F", "#7F00FF", "#00D2FF"];
            return fallback[idx % fallback.length];
        }}

        function getAllSuites() {{
            const suitesSet = new Set();
            Object.values(rawData).forEach(model => {{
                Object.keys(model.suites || {{}}).forEach(s => suitesSet.add(s));
            }});
            return Array.from(suitesSet);
        }}

        function formatSuiteLabel(s) {{
            const labels = {{
                "humaneval": "HumanEval (Code)",
                "tooluse": "Tool-Use (BFCL)",
                "cyber": "Cybersecurity",
                "gpqa_diamond": "GPQA Diamond",
                "mmlu_science": "MMLU Science",
                "competition_math": "MATH (Hendrycks)",
                "arc": "ARC-Challenge"
            }};
            return labels[s] || s.toUpperCase();
        }}

        function renderBarChart() {{
            const ctx = document.getElementById('barChart').getContext('2d');
            const suites = getAllSuites();
            const models = Object.keys(rawData);

            const datasets = models.map((modelName, idx) => {{
                const color = getColor(modelName, idx);
                const dataPoints = suites.map(s => {{
                    const suiteData = rawData[modelName].suites[s];
                    return suiteData ? suiteData[currentMetric] : 0;
                }});

                return {{
                    label: modelName,
                    data: dataPoints,
                    backgroundColor: color + 'BB',
                    borderColor: color,
                    borderWidth: 2,
                    borderRadius: 6
                }};
            }});

            if (barChartInst) barChartInst.destroy();

            barChartInst = new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: suites.map(formatSuiteLabel),
                    datasets: datasets
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            grid: {{ color: 'rgba(255, 255, 255, 0.06)' }},
                            ticks: {{ color: '#8E9BAE' }}
                        }},
                        x: {{
                            grid: {{ display: false }},
                            ticks: {{ color: '#8E9BAE' }}
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            labels: {{ color: '#F0F4F8', font: {{ family: 'Outfit', size: 13 }} }}
                        }},
                        tooltip: {{
                            backgroundColor: 'rgba(18, 22, 33, 0.95)',
                            borderColor: 'rgba(255, 255, 255, 0.1)',
                            borderWidth: 1
                        }}
                    }}
                }}
            }});
        }}

        function renderRadarChart() {{
            const ctx = document.getElementById('radarChart').getContext('2d');
            const suites = getAllSuites();
            const models = Object.keys(rawData);

            const datasets = models.map((modelName, idx) => {{
                const color = getColor(modelName, idx);
                const dataPoints = suites.map(s => {{
                    const suiteData = rawData[modelName].suites[s];
                    return suiteData ? suiteData.pass_at_1_pct : 0;
                }});

                return {{
                    label: modelName,
                    data: dataPoints,
                    borderColor: color,
                    backgroundColor: color + '22',
                    borderWidth: 2,
                    pointBackgroundColor: color
                }};
            }});

            if (radarChartInst) radarChartInst.destroy();

            radarChartInst = new Chart(ctx, {{
                type: 'radar',
                data: {{
                    labels: suites.map(formatSuiteLabel),
                    datasets: datasets
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        r: {{
                            angleLines: {{ color: 'rgba(255, 255, 255, 0.08)' }},
                            grid: {{ color: 'rgba(255, 255, 255, 0.08)' }},
                            pointLabels: {{ color: '#8E9BAE', font: {{ size: 11 }} }},
                            ticks: {{ display: false, beginAtZero: true }}
                        }}
                    }},
                    plugins: {{
                        legend: {{ display: false }}
                    }}
                }}
            }});
        }}

        function renderTable() {{
            const tbody = document.getElementById('tableBody');
            tbody.innerHTML = '';

            let modelIdx = 0;
            Object.entries(rawData).forEach(([modelName, modelData]) => {{
                const color = getColor(modelName, modelIdx++);

                Object.entries(modelData.suites || {{}}).forEach(([suiteKey, s]) => {{
                    const tr = document.createElement('tr');
                    
                    const score = s.pass_at_1_pct.toFixed(2);
                    let scoreClass = 'score-low';
                    if (s.pass_at_1_pct >= 65) scoreClass = 'score-high';
                    else if (s.pass_at_1_pct >= 45) scoreClass = 'score-mid';

                    tr.innerHTML = `
                        <td class="model-cell">
                            <span class="model-color-badge" style="background-color: ${{color}}"></span>
                            ${{modelName}}
                        </td>
                        <td>${{formatSuiteLabel(suiteKey)}}</td>
                        <td><span class="score-pill ${{scoreClass}}">${{score}}%</span></td>
                        <td>[${{s.ci_low.toFixed(1)}}%, ${{s.ci_high.toFixed(1)}}%]</td>
                        <td>${{s.passed_count}} / ${{s.total_tasks}}</td>
                        <td>${{s.ast_validity_pct.toFixed(1)}}%</td>
                    `;
                    tbody.appendChild(tr);
                }});
            }});
        }}

        function setMetric(metricKey) {{
            currentMetric = metricKey;
            document.querySelectorAll('.control-group .btn-toggle').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            renderBarChart();
        }}

        function filterTable() {{
            const query = document.getElementById('tableSearch').value.toLowerCase();
            const rows = document.querySelectorAll('#tableBody tr');
            rows.forEach(r => {{
                const text = r.innerText.toLowerCase();
                r.style.display = text.includes(query) ? '' : 'none';
            }});
        }}

        // Dynamic JSON File Drag & Drop Loader
        function handleFileSelect(event) {{
            const file = event.target.files[0];
            if (file) parseUploadedJson(file);
        }}

        function handleFileDrop(event) {{
            event.preventDefault();
            const file = event.dataTransfer.files[0];
            if (file) parseUploadedJson(file);
        }}

        function parseUploadedJson(file) {{
            const reader = new FileReader();
            reader.onload = function(e) {{
                try {{
                    const data = JSON.parse(e.target.result);
                    const modelName = file.name.replace('.json', '').replace('eval_report_', '').replace('_master_summary', '').title();
                    
                    // Simple parser for custom summary format
                    const suites = {{}};
                    Object.entries(data).forEach(([k, v]) => {{
                        if (typeof v === 'object' && v !== null && 'pass_at_1_pct' in v) {{
                            suites[k] = {{
                                label: v.label || k,
                                pass_at_1_pct: parseFloat(v.pass_at_1_pct || 0),
                                ci_low: parseFloat((v.pass_at_1_95ci || [0])[0]),
                                ci_high: parseFloat((v.pass_at_1_95ci || [0, 0])[1]),
                                total_tasks: parseInt(v.total_tasks || 0),
                                passed_count: parseInt((v.execution_outcomes || {{}}).PASSED || 0),
                                ast_validity_pct: parseFloat(v.ast_validity_pct || 100)
                            }};
                        }}
                    }});

                    if (Object.keys(suites).length > 0) {{
                        rawData[modelName] = {{ display_name: modelName, suites: suites }};
                        renderBarChart();
                        renderRadarChart();
                        renderTable();
                        alert(`Successfully loaded '${{modelName}}' into leaderboard!`);
                    }} else {{
                        alert("Invalid summary format: missing 'pass_at_1_pct' fields.");
                    }}
                }} catch (err) {{
                    alert("Error parsing JSON file: " + err.message);
                }}
            }};
            reader.readAsText(file);
        }}

        // Initialize Dashboard Charts
        document.addEventListener('DOMContentLoaded', () => {{
            renderBarChart();
            renderRadarChart();
            renderTable();
        }});
    </script>
</body>
</html>
"""
    return html


def main():
    """Generates the benchmark comparison dashboard HTML file."""
    print("=" * 80)
    print("  TÉLOS BENCHMARK DASHBOARD GENERATOR")
    print("=" * 80 + "\n")

    logo_b64 = load_logo_base64()
    if logo_b64:
        print(f"[OK] Embedded Télos Logo from {LOGO_PATH} (Base64 length: {len(logo_b64)})")
    else:
        print("! Warning: Télos logo not found at logos/telos_logo.png")

    eval_data = collect_evaluation_data()
    print(f"[OK] Collected evaluation data for {len(eval_data)} models:")
    for model_name, info in eval_data.items():
        print(f"  · {model_name:<32}: {len(info['suites'])} suites")

    html_content = generate_html_dashboard(eval_data, logo_b64)

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[OK] Dashboard generated successfully at:")
    print(f"  file://{OUTPUT_HTML.resolve()}")


if __name__ == "__main__":
    main()
