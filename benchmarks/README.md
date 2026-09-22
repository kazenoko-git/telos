# Télos Benchmark Suite

This directory contains evaluation results, comparison graphs, and hardware benchmarks across edge models and Télos architectures.

<p align="center">
  <a href="https://telos.research.wingit.tech"><strong>Research Portal: telos.research.wingit.tech</strong></a>
  <br>
  <em>Note: The research portal is currently under active construction and is not finished yet.</em>
</p>

## Benchmark Diagrams

### 1. Overall Capability Radar

<p align="center">
  <img src="../figures/benchmark_radar_graph.png" alt="Benchmark Radar Graph" width="600">
</p>

This radar graph compares **Apple AFM-3 Core Advanced**, **IBM Granite 4.2 3B**, and **LiquidAI LFM 8B** across 7 test suites:
- **Apple AFM-3 Core Advanced** leads in tool-use (BFCL: 73.33%), coding (HumanEval: 62.80%), and hard questions (GPQA Diamond: 39.39%).
- **IBM Granite 4.2 3B** leads in math (Competition MATH: 64.71%) and reasoning (ARC-Challenge: 86.60%).
- **LiquidAI LFM 8B** leads in general science (MMLU Science: 73.11%) and security auditing (24.00%).

### 2. Runaway Loops Across Tasks

When small models run on devices, they can get caught in repeating loops where they print the same tokens over and over until context runs out.

<p align="center">
  <img src="../figures/benchmark_repetition_rate.png" alt="Benchmark Runaway Loop Rate" width="800">
</p>

Across all 3,067 evaluated tasks:
- **IBM Granite 4.2 3B** had the most runaway loops at **36.8%** (1,129 tasks stuck in loops). We spent significant time isolating and analyzing these loops.
- **LiquidAI LFM 8B** had a runaway loop rate of **28.5%** (875 tasks).
- **Apple AFM-3 Core Advanced** had the lowest loop rate at **23.4%** (718 tasks).

### 3. Throughput and Memory Usage

<p align="center">
  <img src="../figures/benchmark_throughput_memory.png" alt="Benchmark Speed and Memory" width="800">
</p>

Speed and memory footprint on Apple Silicon:
- **Apple AFM-3 Core Advanced** (Swift IPC): **58.7 tokens/sec**, uses **2.4 GB** RAM.
- **IBM Granite 4.2 3B** (MLX): **41.2 tokens/sec**, uses **3.2 GB** RAM.
- **LiquidAI LFM 8B A1B** (MLX): **22.4 tokens/sec**, uses **7.8 GB** RAM.

## Master Leaderboard Summary (7-Suite Completion)

| Model | HumanEval (Pass@1) | BFCL Tool-Use | Cyber Auditing | GPQA Diamond | MMLU Science | Competition MATH | ARC Challenge | 7-Suite Average |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Apple AFM-3 Core Advanced** | **62.80%** | **73.33%** | 12.00% | **39.39%** | 69.75% | 52.00% | 85.32% | **56.37%** |
| **IBM Granite 4.2 3B (MLX)** | 60.37% | 70.00% | 22.00% | 28.79% | 61.82% | **64.71%** | **86.60%** | 56.33% |
| **LiquidAI LFM 8B A1B** | 45.73% | 63.33% | **24.00%** | 20.71% | **73.11%** | 43.57% | 59.56% | 47.14% |

## Detailed Reports

| Report | Description |
| :--- | :--- |
| **[AFM vs LFM vs IBM Comparison](afm_vs_lfm_vs_ibm.md)** | Full 7-suite comparison of Apple AFM-3, IBM Granite 4.2 3B, and LiquidAI LFM 8B. |
| **[Apple Foundation Models (AFM)](afm.md)** | Evaluation report for AFM-3 Core & Core Advanced over native Swift IPC. |
| **[IBM Granite 4.2 3B](ibm_granite.md)** | Evaluation report for IBM Granite 4.2 3B on Apple Silicon. |
| **[LiquidAI LFM 8B A1B](liquid_lfm.md)** | Performance report for LiquidAI LFM 8B A1B. |

## Dashboard Generator

To generate the standalone HTML comparison dashboard:

```bash
python scripts/generate_benchmark_dashboard.py
```
