#!/usr/bin/env python3
"""Validate and report a completed probe; never imports a model backend."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def analyze(folder):
    cases = json.loads((folder / "cases.json").read_text())
    prepared = json.loads((folder / "prepared.json").read_text())
    runtime = json.loads((folder / "runtime.json").read_text())
    rows = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
    expected = len(cases) * len(prepared["modes"]) * 4
    assert len(rows) == expected, (len(rows), expected)
    assert len({(r['id'], r['mode'], r['placeholder'], r['readout']) for r in rows}) == expected
    cases_by_id = {c["id"]: c for c in cases}
    grouped = defaultdict(list)
    for row in rows:
        assert row["id"] in cases_by_id
        assert row["target"] == cases_by_id[row["id"]]["target"]
        grouped[row["mode"], row["placeholder"], row["readout"]].append(row)
    stats = {}
    for key, subset in grouped.items():
        paired = [r for r in subset if r["pair"]]
        natural = [r for r in subset if not r["pair"]]
        pairs = defaultdict(list)
        for r in paired:
            pairs[r["pair"]].append(r)
        assert all(len(p) == 2 for p in pairs.values())
        deltas = [abs(p[0]["candidate_a_logp"]-p[1]["candidate_a_logp"]) for p in pairs.values()]
        record = dict(n=len(subset), paired_n=len(paired), pairs=len(pairs),
             paired_top1=sum(r["correct"] for r in paired),
             both_top1=sum(all(r["correct"] for r in p) for p in pairs.values()),
             both_top1_strict_vs_alternative=sum(all(r["correct"] and r["candidate_margin"] > 0
                                                   for r in p) for p in pairs.values()),
             both_prefer_required=sum(all(r["candidate_margin"] > 0 for r in p) for p in pairs.values()),
             prediction_changes=sum(p[0]["prediction"] != p[1]["prediction"] for p in pairs.values()),
             max_suffix_logp_change=max(deltas, default=0), natural_n=len(natural),
             natural_top1=sum(r["correct"] for r in natural),
             natural_top5=sum(r["rank"] <= 5 for r in natural),
             natural_nll=statistics.mean(-r["target_logp"] for r in natural) if natural else None)
        if key[0] == "causal":
            assert record["max_suffix_logp_change"] <= 1e-4, "Causal independence failed"
            assert record["both_top1"] == 0 and record["both_prefer_required"] == 0
        stats["/".join(key)] = record
    text = ["# Frozen Qwen3-4B attention probe: measured results", "",
            f"Model revision: `{runtime['revision']}`. Original BF16 weights, no training or FIM.", "",
            f"Validated {len(rows)} score records from {len(cases)} examples.",
            f"Native/custom causal maximum logit difference: {runtime['native_causal_max_logit_delta']:.6g}.",
            "Paired causal suffix-independence controls passed.", "",
            "Numbers below are descriptive results for this fixed, small probe. The paired",
            "examples use two templates; they are not a broad code benchmark. Local code",
            "examples are not verified absent from Qwen pretraining.", ""]
    for placeholder in ("zero", "underscore"):
        for readout in ("shifted", "same_position"):
            text += [f"## {placeholder} placeholder; {readout} readout", "",
                     "| Attention | Paired token top-1 | Both top-1 (strict vs alternative) | Both prefer required operator | Local token top-1 | Local top-5 | Local NLL |",
                     "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for mode in prepared["modes"]:
                s = stats[f"{mode}/{placeholder}/{readout}"]
                nll = f"{s['natural_nll']:.3f}" if s['natural_nll'] is not None else "n/a"
                text.append(f"| {mode} | {s['paired_top1']}/{s['paired_n']} | {s['both_top1_strict_vs_alternative']}/{s['pairs']} | "
                            f"{s['both_prefer_required']}/{s['pairs']} | {s['natural_top1']}/{s['natural_n']} | "
                            f"{s['natural_top5']}/{s['natural_n']} | {nll} |")
            text.append("")
    text += ["## Interpretation limits", "",
        "A changed prediction demonstrates sensitivity, not necessarily correct use of right context.",
        "Both pair members favoring their respective required operator is a stronger diagnostic;",
        "The table excludes apparent paired top-1 successes that tie with the alternative operator.",
        "Recovering both as vocabulary-wide top-1 is stronger still. Neither proves a working",
        "multi-token diffusion sampler, iterative convergence, calibration, or an inference speedup.",
        "Any promising condition needs a new evaluation set before claiming generalization.", ""]
    (folder / "analysis.json").write_text(json.dumps(stats, indent=2)+"\n")
    (folder / "RESULTS.md").write_text("\n".join(text))
    (folder / "completion.json").write_text(json.dumps(dict(status="complete", records=len(rows),
        examples=len(cases), causal_controls="passed",
        results_sha256=hashlib.sha256((folder / 'results.jsonl').read_bytes()).hexdigest()), indent=2)+"\n")
    print("\n".join(text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    analyze(parser.parse_args().folder)
