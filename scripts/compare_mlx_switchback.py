"""Long, complementary sync/async switchbacks with measured thermal/clock controls.

This does not lock GPU clocks. It balances treatment within one training process,
cools before replay, and records enough data for matched-frequency analysis.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

if __package__:
    from .capture_mlx_timeline import load_profile,write_json
    from .mlx_hardware_monitor import HardwareMonitor
else:
    from capture_mlx_timeline import load_profile,write_json
    from mlx_hardware_monitor import HardwareMonitor


def schedule_modes(pairs,seed):
    if pairs<2 or pairs%2:raise ValueError('Use a positive even number of pairs')
    first=[False]*(pairs//2)+[True]*(pairs//2)
    random.Random(seed).shuffle(first)
    return [mode for lead in first for mode in (lead,not lead)]


def wait_for_cooldown(root,trial,monitor,profile,seconds,timeout):
    # Refresh after the preceding trial: don't average its workload into idle power.
    monitor.sample()
    began=time.monotonic();quiet_since=None
    with (root/'cooldown.jsonl').open('a',buffering=1) as stream:
        while True:
            sample=monitor.sample();system=profile.system_sample()
            power=subprocess.check_output(['pmset','-g','batt'],text=True,timeout=5)
            text=subprocess.check_output(['ps','-axo','pid=,pcpu=,comm='],text=True,timeout=5)
            processes=[]
            for line in text.splitlines():
                parts=line.strip().split(None,2)
                if len(parts)==3 and int(parts[0])!=os.getpid():
                    processes.append((float(parts[1]),parts[2]))
            top=max(processes,default=(0,''))
            watts=sample['energy_model'].get('GPU Energy',{}).get('estimated_watts')
            quiet=(sample['thermal_state']==0 and watts is not None and 0<=watts<=1.0
                and "Now drawing from 'AC Power'" in power
                and system['memory_free_percent']>=35 and system['swap_free_mb']>=512)
            if quiet:
                if quiet_since is None:quiet_since=time.monotonic()
            else:quiet_since=None
            record=dict(trial=trial,sample=sample,system=system,power_source=power,
                top_other_cpu_estimate=top,ready_seconds=time.monotonic()-quiet_since if quiet_since else 0)
            stream.write(json.dumps(record)+'\n')
            if record['ready_seconds']>=seconds:return
            if time.monotonic()-began>=timeout:raise RuntimeError('Matched nominal cooldown conditions were not reached')
            time.sleep(2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pairs',type=int,default=24)
    parser.add_argument('--block-updates',type=int,default=32)
    parser.add_argument('--warmup-updates',type=int,default=128)
    parser.add_argument('--seed',type=int,default=20261117)
    parser.add_argument('--cooldown-seconds',type=int,default=30)
    parser.add_argument('--cooldown-timeout',type=int,default=600)
    args=parser.parse_args()
    modes=schedule_modes(args.pairs,args.seed)
    updates=len(modes)*args.block_updates
    if args.block_updates<1 or not 0<args.warmup_updates<updates or args.warmup_updates%(2*args.block_updates):
        parser.error('Warmup must cover whole pairs and be shorter than the run')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    source=args.source.resolve();data=args.data.resolve();scripts=Path(__file__).resolve().parent
    policy=subprocess.check_output(['pmset','-g','custom'],text=True,timeout=5)
    (root/'power-policy.txt').write_text(policy)
    plan=dict(updates=updates,pairs=args.pairs,block_updates=args.block_updates,warmup_updates=args.warmup_updates,
        seed=args.seed,cache_mib=256,hardware_clock_lock=False,
        cooldown=dict(seconds=args.cooldown_seconds,thermal_state='nominal',gpu_estimated_power_max_w=1.0),
        matching=dict(max_pair_frequency_relative_difference=.01,max_block_frequency_cv=.02,
            min_full_hardware_intervals_per_block=6,same_single_thermal_state=True,
            max_pair_other_cpu_pct_difference=15,
            minimum_matched_pairs_per_trial=8,minimum_matched_pairs_total=16),
        inference='Overall switchback estimates net effect; matched-clock analysis estimates conditional effect, not a hardware-locked intervention.')
    write_json(root/'plan.json',plan)
    write_json(root/'manifest.json',dict(source=str(source),data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'telos').rglob('*.py')},
        harness_sha256={name:hashlib.sha256((scripts/name).read_bytes()).hexdigest() for name in
            ('capture_mlx_timeline.py','mlx_hardware_monitor.py','compare_mlx_switchback.py')}))
    profile=load_profile(source);monitor=HardwareMonitor();results=[]
    for trial in range(2):
        selected=modes if trial==0 else [not mode for mode in modes]
        schedule=root/f'schedule-{trial}.json'
        write_json(schedule,dict(block_updates=args.block_updates,modes=selected))
        print(json.dumps(dict(event='cooling',trial=trial)),flush=True)
        wait_for_cooldown(root,trial,monitor,profile,args.cooldown_seconds,args.cooldown_timeout)
        if subprocess.check_output(['pmset','-g','custom'],text=True,timeout=5)!=policy:
            raise RuntimeError('Power policy changed before trial')
        directory=root/f'trial-{trial}'
        command=[sys.executable,str(scripts/'capture_mlx_timeline.py'),'--source',str(source),'--data',str(data),
            '--output',str(directory),'--updates',str(updates),'--warmup',str(args.warmup_updates),
            '--cache-mib','256','--no-trace','--hardware-monitor','--save-training-state',
            '--mode-schedule',str(schedule),'--require-ac-power','--max-thermal-state','1',
            '--timeout-seconds',str(updates*1.5+120)]
        print(json.dumps(dict(event='starting',trial=trial,updates=updates)),flush=True)
        with (root/f'trial-{trial}.log').open('w') as log:
            run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=updates*1.5+180)
        capture=json.loads((directory/'capture.json').read_text())
        result=dict(trial=trial,returncode=run.returncode,capture_status=capture['status'],
            worker_returncode=capture.get('worker_returncode'),training=capture.get('training_summary'),
            power_policy_unchanged=subprocess.check_output(['pmset','-g','custom'],text=True,timeout=5)==policy)
        results.append(result);write_json(root/'results.json',results);print(json.dumps(result),flush=True)
        if (run.returncode or capture['status']!='untraced_complete' or capture.get('worker_returncode')!=0
                or not result['power_policy_unchanged'] or capture.get('training_summary',{}).get('updates')!=updates):
            raise RuntimeError('Controlled trial invalid; stopping without replacing it')
    print('SWITCHBACK_COMPLETE',flush=True)


if __name__=='__main__':main()
