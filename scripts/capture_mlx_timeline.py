"""Capture a short, warmed Metal timeline separately from throughput benchmarks.

Uses an existing frozen profile_mlx_training.py without changing its source.
The trace is diagnostic: its overhead invalidates throughput comparisons.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

XCTRACE = '/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def load_profile(source):
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('frozen_profile', source / 'scripts/profile_mlx_training.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(args):
    profile = load_profile(args.source)
    schedule=None
    if args.mode_schedule:
        schedule=json.loads(args.mode_schedule.read_text())
        if (schedule['block_updates']<=0 or len(schedule['modes'])*schedule['block_updates']!=args.updates
                or any(type(mode) is not bool for mode in schedule['modes'])):
            raise ValueError('Mode schedule must cover every update with boolean modes')
    import mlx.core as mx
    set_cache_limit=mx.set_cache_limit
    cache_calls=[]
    def cache_limit_override(requested):
        cache_calls.append(dict(requested_bytes=requested,applied_bytes=args.cache_mib*1024**2))
        write_json(args.output/'cache_override.json',cache_calls)
        return set_cache_limit(args.cache_mib*1024**2)
    mx.set_cache_limit=cache_limit_override
    from telos.training.mlx_step import MLXTrainingStep
    if schedule is not None:
        import inspect
        if 'async_accumulation' not in inspect.signature(MLXTrainingStep.__init__).parameters:
            raise ValueError('Scheduled source must support async_accumulation')
    if args.async_accumulation:
        original_init=MLXTrainingStep.__init__
        def async_init(step,*positional,**keywords):
            keywords['async_accumulation']=True
            original_init(step,*positional,**keywords)
        MLXTrainingStep.__init__=async_init
    original = MLXTrainingStep.__call__
    count = 0
    last_step = None
    with (args.output / 'host_steps.jsonl').open('w', buffering=1) as stream:
        def measured(step, *positional, **keywords):
            nonlocal count, last_step
            last_step = step
            if schedule is not None:
                step.async_accumulation=schedule['modes'][count//schedule['block_updates']]
            begin, wall = time.perf_counter_ns(), time.time_ns()
            cpu = time.process_time_ns()
            result = original(step, *positional, **keywords)
            count += 1
            stream.write(json.dumps(dict(step=count, start_monotonic_ns=begin,
                end_monotonic_ns=time.perf_counter_ns(), start_epoch_ns=wall,
                end_epoch_ns=time.time_ns(), process_cpu_ns=time.process_time_ns()-cpu,
                async_accumulation=bool(getattr(step,'async_accumulation',False)),
                schedule_block=(count-1)//schedule['block_updates'] if schedule else None,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))+'\n')
            return result
        MLXTrainingStep.__call__ = measured
        sys.argv = [str(args.source/'scripts/profile_mlx_training.py'),
            '--output', str(args.output/'training'), '--tokens', str(args.updates*16384),
            '--microbatch', '8', '--grad-accum', '8', '--seq-len', '256',
            '--data', str(args.data), '--max-mlx-gib', '3.2',
            '--critical-memory-free-percent', '25', '--critical-swap-free-mib', '256',
            '--critical-swap-growth-mib', '512', '--experimental-packed-swiglu',
            '--force-fused-attention', '--compiled-accumulation']
        profile.main()
        if args.save_training_state and last_step is not None:
            from mlx.utils import tree_flatten
            state=[last_step.model.parameters(),last_step.optimizer.state,
                getattr(last_step.optimizer,'master_params',{}),mx.random.state]
            arrays={key:value for key,value in tree_flatten(state) if isinstance(value,mx.array)}
            mx.eval(*arrays.values())
            mx.save_safetensors(str(args.output/'final_training_state.safetensors'),arrays)


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=160)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--seconds', type=float, default=2)
    parser.add_argument('--cache-mib', type=int, choices=[128,256,512], default=256)
    parser.add_argument('--no-trace', action='store_true')
    parser.add_argument('--hardware-monitor', action='store_true')
    parser.add_argument('--async-accumulation', action='store_true')
    parser.add_argument('--save-training-state', action='store_true')
    parser.add_argument('--timeout-seconds',type=float,default=150)
    parser.add_argument('--mode-schedule',type=Path)
    parser.add_argument('--require-ac-power',action='store_true')
    parser.add_argument('--max-thermal-state',type=int,choices=[0,1,2,3])
    parser.add_argument('--template', default='Time Profiler')
    parser.add_argument('--instrument', action='append', default=None)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.instrument is None:
        args.instrument=['GPU','Metal Application'] if args.template=='Time Profiler' else []
    args.source, args.data, args.output = args.source.resolve(), args.data.resolve(), args.output.resolve()
    if args.mode_schedule:args.mode_schedule=args.mode_schedule.resolve()
    if args.worker:
        return worker(args)
    if not 0 < args.seconds <= 5 or not 0 < args.warmup < args.updates:
        parser.error('Use a trace of at most five seconds and leave updates after warmup')
    if args.timeout_seconds<=0:parser.error('--timeout-seconds must be positive')
    if (args.require_ac_power or args.max_thermal_state is not None) and not args.hardware_monitor:
        parser.error('Power/thermal controls require --hardware-monitor')
    args.output.mkdir(parents=True, exist_ok=False)
    profile = load_profile(args.source)
    initial = profile.system_sample()
    report = dict(status='preflight', initial=initial, trace_is_diagnostic=True,
        limitations=['System utilization counters include other apps.',
            'Host step windows are not GPU kernel intervals.',
            'Trace overhead changes scheduling and throughput.',
            'Missing shader rows or counters mean unavailable evidence, never zero activity.'],
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    write_json(args.output/'capture.json', report)
    if initial['memory_free_percent'] < (30 if args.no_trace else 35) or initial['swap_free_mb'] < 512:
        report['status']='insufficient_memory_headroom'
        write_json(args.output/'capture.json',report)
        print(json.dumps(report))
        return
    options={'Time Profiler': {'recordWaitingThreads': True,
        'contextSwitchSampling': True, 'highFrequencySampling': True,
        'recordKernelStacks': False}}
    write_json(args.output/'recording_options.json',options)
    command=[sys.executable,str(Path(__file__).resolve()),'--worker',
        '--source',str(args.source),'--data',str(args.data),'--output',str(args.output),
        '--updates',str(args.updates),'--cache-mib',str(args.cache_mib)]
    if args.async_accumulation:command.append('--async-accumulation')
    if args.save_training_state:command.append('--save-training-state')
    if args.mode_schedule:command.extend(['--mode-schedule',str(args.mode_schedule)])
    child = trace = None
    hardware=None
    if args.hardware_monitor:
        from mlx_hardware_monitor import HardwareMonitor
        try:
            hardware=HardwareMonitor()
        except Exception as error:
            report['hardware_monitor_error']=str(error)
    try:
        with (args.output/'training.log').open('w') as log, (args.output/'xctrace.log').open('w') as trace_log, (args.output/'observer.jsonl').open('w',buffering=1) as observer:
            env=os.environ.copy()
            for key in ('MLX_MAX_OPS_PER_BUFFER','MLX_MAX_MB_PER_BUFFER','MLX_METAL_FAST_SYNCH'):
                env.pop(key,None)
            env.update(PYTHONPATH=str(args.source),MLX_ENABLE_TF32='0')
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            report['worker_pid']=child.pid
            deadline=time.monotonic()+args.timeout_seconds
            rows=[]
            while child.poll() is None:
                steps=args.output/'host_steps.jsonl'
                if steps.exists():
                    rows=[json.loads(line) for line in steps.read_text().splitlines()]
                sample=profile.system_sample()
                sample['monotonic_ns']=time.perf_counter_ns()
                sample['completed_steps']=len(rows)
                observer.write(json.dumps(sample)+'\n')
                if hardware is not None:
                    record=hardware.sample()
                    record['completed_steps']=len(rows)
                    record['processes']=subprocess.check_output(['ps','-axo','pid=,pcpu=,rss=,comm='],text=True,timeout=5)
                    if args.require_ac_power or len(rows)%5==0:
                        record['power_source']=subprocess.check_output(['pmset','-g','batt'],text=True,timeout=5)
                    with (args.output/'hardware.jsonl').open('a') as stream:
                        stream.write(json.dumps(record)+'\n')
                    if args.require_ac_power and "Now drawing from 'AC Power'" not in record['power_source']:
                        report['status']='power_source_changed'
                        break
                    if args.max_thermal_state is not None and record['thermal_state']>args.max_thermal_state:
                        report['status']='thermal_guard'
                        break
                if (sample['memory_free_percent'] < 25 or sample['swap_free_mb'] < 256
                        or sample['swap_used_mb']-initial['swap_used_mb'] > 512):
                    report['status']='observer_memory_guard'
                    break
                if not args.no_trace and trace is None and len(rows)>=args.warmup:
                    trace_command=[XCTRACE,'record','--template',args.template,
                        '--recording-options',str(args.output/'recording_options.json'),
                        '--attach',str(child.pid),'--time-limit',f'{args.seconds}s',
                        '--output',str(args.output/'metal.trace'),'--no-prompt']
                    for instrument in args.instrument:
                        trace_command.extend(['--instrument',instrument])
                    report.update(trace_command=trace_command,trace_request_epoch_ns=time.time_ns(),
                        steps_at_trace_request=len(rows))
                    trace=subprocess.Popen(trace_command,stdout=trace_log,stderr=subprocess.STDOUT)
                    write_json(args.output/'capture.json',report)
                if trace is not None and trace.poll() is not None:
                    report['trace_returncode']=trace.returncode
                    report['status']='trace_recorded' if trace.returncode==0 else 'trace_failed'
                    # Let the already-running training finish unless a guard fires.
                if time.monotonic()>deadline:
                    report['status']='timeout'
                    break
                time.sleep(1)
            if child.poll() is not None:
                report['worker_returncode']=child.returncode
            stop_process(trace)
            stop_process(child)
            report['worker_returncode']=child.returncode
            if trace is not None:
                report['trace_returncode']=trace.returncode
        report['final']=profile.system_sample()
        summary=args.output/'training/summary.json'
        if summary.exists():
            report['training_summary']=json.loads(summary.read_text())
            if (args.no_trace and report['status'] not in ('observer_memory_guard','power_source_changed','thermal_guard','timeout')
                    and report['training_summary']['status']=='complete' and report['worker_returncode']==0):
                report['status']='untraced_complete'
        if report.get('trace_returncode')==0:
            with (args.output/'export.log').open('w') as log:
                result=subprocess.run([XCTRACE,'export','--input',str(args.output/'metal.trace'),
                    '--toc','--output',str(args.output/'toc.xml')],stdout=log,stderr=subprocess.STDOUT,timeout=45)
                report['toc_returncode']=result.returncode
                # Export only compact execution/encoding tables. Full counter exports
                # can consume gigabytes and are intentionally a separate operation.
                for schema,filename in [('metal-gpu-intervals','gpu_intervals.xml'),
                        ('metal-application-intervals','application_intervals.xml')]:
                    result=subprocess.run([XCTRACE,'export','--input',str(args.output/'metal.trace'),
                        '--xpath',f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]',
                        '--output',str(args.output/filename)],stdout=log,stderr=subprocess.STDOUT,timeout=45)
                    report[filename+'_returncode']=result.returncode
    finally:
        stop_process(trace)
        stop_process(child)
        write_json(args.output/'capture.json',report)
    print(json.dumps(report))


if __name__=='__main__':
    main()
