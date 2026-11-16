"""Summarize complete ABBA trials without treating update samples as replicas."""
import argparse
import json
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    results = json.loads((args.directory / 'results.json').read_text())
    validation = json.loads((args.directory / 'checkpoint_validation.json').read_text())
    assert len(results) == 8 and all(r['valid'] for r in results)
    assert all(not r['summary']['errors'] for r in results), 'Guard/telemetry errors occurred'
    assert len(validation['heldout']) == 8 and len(validation['backward_graphs']) == 4
    assert len({r['metadata']['initial_weights_sha256'] for r in results}) == 1
    summary = dict(initial_weights_sha256=results[0]['metadata']['initial_weights_sha256'], shapes=[],
        limitations=['Two adjacent fresh-process pairs per shape; pair ranges are not confidence intervals.',
                     'Completed optimizer-update throughput excludes process/model setup and final checkpoint serialization.',
                     'Held-out checks are short-run diagnostics, not convergence or general quality equivalence.'])
    for length in (2048, 256):
        runs = [r for r in results if r['length'] == length]
        assert [r['mode'] for r in runs] == ['native', 'fallback', 'fallback', 'native']
        assert all(r['metadata']['memory_plan'] == runs[0]['metadata']['memory_plan'] for r in runs)
        rates = {mode: [r['tokens_per_second'] for r in runs if r['mode'] == mode] for mode in ('native', 'fallback')}
        gains = [100 * (runs[0]['tokens_per_second']/runs[1]['tokens_per_second'] - 1),
                 100 * (runs[3]['tokens_per_second']/runs[2]['tokens_per_second'] - 1)]
        row = dict(length=length, memory_plan=runs[0]['metadata']['memory_plan'], rates=rates,
                   paired_native_gain_percent=gains,
                   geometric_native_gain_percent=100 * (math.exp(statistics.mean(math.log1p(g/100) for g in gains))-1),
                   runs=[])
        for run in runs:
            trial = Path(run['trial_dir'])
            steps = [json.loads(line) for line in (trial/'steps.jsonl').read_text().splitlines()]
            warm = steps[run['warmup']:]
            effective = run['metadata']['actual_batch_size'] * run['metadata']['actual_grad_accum']
            rate = len(warm)*effective*length/sum(r['training_step_seconds'] for r in warm)
            assert abs(rate/run['tokens_per_second']-1) < 1e-12
            system = [json.loads(line) for line in (trial/'system.jsonl').read_text().splitlines()]
            baseline = run['metadata']['swap_growth_baseline']
            growth = [max(0, s['swap_used_mb']-baseline['swap_used_mb']) for s in system if s['time'] >= baseline['time']]
            heldout = next(h for h in validation['heldout'] if h['name'] == run['name'])
            middle = len(warm)//2
            halves = [part for part in (warm[:middle], warm[middle:]) if part]
            row['runs'].append(dict(name=run['name'], mode=run['mode'], updates=len(steps), warmup=run['warmup'],
                tokens_per_second=rate, peak_mlx_gib=run['peak_mlx_bytes']/1024**3,
                half_window_tokens_per_second=[len(part)*effective*length/sum(s['training_step_seconds'] for s in part) for part in halves],
                system_free_percent_min=min(s['memory_free_percent'] for s in system),
                system_swap_free_mib_min=min(s['swap_free_mb'] for s in system),
                post_warmup_swap_growth_mib_max=max(growth, default=0), heldout_mean_ce=heldout['mean_ce']))
        summary['shapes'].append(row)
    (args.directory/'comparison_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
