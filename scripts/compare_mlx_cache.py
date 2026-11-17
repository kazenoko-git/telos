"""Reversed-order cache comparisons with separate throughput and trace runs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from capture_mlx_timeline import load_profile, XCTRACE, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    args.output=args.output.resolve();args.source=args.source.resolve();args.data=args.data.resolve()
    args.output.mkdir(parents=True,exist_ok=args.resume)
    scripts=Path(__file__).resolve().parent
    source_files=list((args.source/'telos').rglob('*.py'))+[args.source/'scripts/profile_mlx_training.py']
    manifest=dict(source=str(args.source),data=str(args.data),
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        source_sha256={str(f.relative_to(args.source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files},
        harness_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [scripts/'capture_mlx_timeline.py',scripts/'analyze_mlx_timeline.py',scripts/'correlate_mlx_timeline.py',Path(__file__)]})
    if args.resume:
        previous=json.loads((args.output/'manifest.json').read_text())
        assert previous['source_sha256']==manifest['source_sha256'] and previous['data_sha256']==manifest['data_sha256']
        assert previous['harness_sha256']['capture_mlx_timeline.py']==manifest['harness_sha256']['capture_mlx_timeline.py']
        write_json(args.output/'resume_manifest.json',manifest)
    else:
        write_json(args.output/'manifest.json',manifest)
    profile=load_profile(args.source)
    results=json.loads((args.output/'results.json').read_text()) if args.resume and (args.output/'results.json').exists() else []
    for mode in ('untraced','traced'):
        for index,cache in enumerate((128,256,512,512,256,128)):
            if any(r['mode']==mode and r['index']==index and r['valid_training'] for r in results):
                continue
            # Let previous trace processing settle without weakening memory guards.
            for attempt in range(25):
                s=profile.system_sample()
                if s['memory_free_percent']>=(30 if mode=='untraced' else 35) and s['swap_free_mb']>=512:
                    break
                print(json.dumps(dict(event='waiting_for_memory',mode=mode,cache_mib=cache,sample=s)),flush=True)
                time.sleep(5)
            trial=args.output/f'{mode}_{index:02d}_{cache}'
            command=[sys.executable,str(scripts/'capture_mlx_timeline.py'),
                '--source',str(args.source),'--data',str(args.data),'--output',str(trial),
                '--updates',str(64 if mode=='untraced' else 96),'--warmup','20',
                '--seconds','2','--cache-mib',str(cache)]
            if mode=='untraced':command.append('--no-trace')
            print(json.dumps(dict(event='starting',mode=mode,index=index,cache_mib=cache)),flush=True)
            if not trial.exists():
                with (args.output/f'{mode}_{index:02d}_{cache}.log').open('w') as log:
                    run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                    returncode=run.returncode
            else:
                assert args.resume
                returncode=0
            capture=json.loads((trial/'capture.json').read_text())
            metadata=trial/'training/metadata.json';steps=trial/'host_steps.jsonl'
            rows=[json.loads(line) for line in steps.read_text().splitlines()] if steps.exists() else []
            warm=rows[16:]
            valid=returncode==0 and capture.get('worker_returncode')==0 and capture.get('training_summary',{}).get('status')=='complete'
            result=dict(mode=mode,index=index,cache_mib=cache,valid_training=valid,
                capture_status=capture['status'],updates=len(rows),
                initial_system=capture['initial'],final_system=capture.get('final'),
                warmed_step_seconds=sum((r['end_monotonic_ns']-r['start_monotonic_ns'])/1e9 for r in warm),
                warmed_updates=len(warm),directory=str(trial))
            result['warmed_tokens_per_second']=len(warm)*16384/result['warmed_step_seconds'] if warm else None
            if metadata.exists():
                meta=json.loads(metadata.read_text());result['initial_weights_sha256']=meta.get('initial_weights_sha256')
            training_steps=trial/'training/steps.jsonl'
            if training_steps.exists():
                data=[json.loads(line) for line in training_steps.read_text().splitlines()]
                result['peak_mlx_bytes']=max((r['mlx_peak_bytes'] for r in data),default=0)
                result['warmed_cache_mean_bytes']=sum(r['mlx_cache_bytes'] for r in data[16:])/len(data[16:]) if len(data)>16 else None
            if mode=='traced' and capture.get('trace_returncode')==0:
                for schema,filename in [('time-profile','cpu_samples.xml'),
                        ('metal-application-command-buffer-submissions','submissions.xml')]:
                    if args.resume and (trial/filename).exists():
                        continue
                    with (trial/'extra_exports.log').open('a') as log:
                        export=subprocess.run([XCTRACE,'export','--input',str(trial/'metal.trace'),
                            '--xpath',f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]',
                            '--output',str(trial/filename)],stdout=log,stderr=subprocess.STDOUT,timeout=60)
                        if export.returncode:raise RuntimeError(f'Export failed: {trial}/{filename}')
                for script in ('analyze_mlx_timeline.py','correlate_mlx_timeline.py'):
                    with (trial/'analysis.log').open('a') as log:
                        subprocess.run([sys.executable,str(scripts/script),str(trial)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=45)
                result['analysis']=json.loads((trial/'analysis.json').read_text())
                result['correlation']={k:v for k,v in json.loads((trial/'correlation.json').read_text()).items() if k!='top_gaps'}
            results.append(result);write_json(args.output/'results.json',results)
            print(json.dumps(result),flush=True)
            if not valid or (mode=='traced' and 'analysis' not in result):
                raise RuntimeError('Trial did not complete; stopping comparison, preserving partial evidence')
    assert len({r['initial_weights_sha256'] for r in results})==1
    print('CACHE_COMPARISON_COMPLETE',flush=True)


if __name__=='__main__':
    main()
