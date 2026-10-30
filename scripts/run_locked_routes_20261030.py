"""Suite entry for matched locked-parent dual and pure-input route validation."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import _bootstrap as boot

ROOT = boot.ROOT
OUT = ROOT / 'outputs/locked_routes_20261030'


def active_workers():
    names = {'validate_locked_dual_20261030.py', 'validate_locked_plugin_20261030.py'}
    found = []
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = p.read_bytes().split(b'\0')
            selected = any(Path(os.fsdecode(v)).name in names or
                           ('locked_routes_20261030/plugin/' in os.fsdecode(v) and
                            os.fsdecode(v).endswith('.py')) for v in args[1:] if v)
            if args and b'python' in args[0] and selected:
                found.append(int(p.parent.name))
        except (OSError, ValueError):
            continue
    return found


def completed_counts():
    return {route: len(list((OUT / route / 'runs').glob('*/*/f*/COMPLETE.json')))
            for route in ('dual', 'plugin')}


def command(script, *args, gpu='', log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(ROOT/'src'))
    with (OUT/log).open('a') as f:
        subprocess.run([sys.executable, str(ROOT/'scripts'/script), *args], cwd=ROOT,
                       env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def dual_neural():
    command('validate_locked_dual_20261030.py', '--inner', 'ArcFENet', gpu='1', log='suite_dual_neural.log')
    for model in ('ArcFENet', 'ArcFusionNet'):
        command('validate_locked_dual_20261030.py', '--heads', model, gpu='1', log='suite_dual_neural.log')


def dual_cpu():
    command('validate_locked_dual_20261030.py', '--inner', 'HYDRA', log='suite_dual_cpu.log')
    command('validate_locked_dual_20261030.py', '--heads', 'HYDRA', log='suite_dual_cpu.log')


def run_study(suite, execute=False):
    assert set(suite['datasets']) == {'continuous5m', 'zenodo5501527'}
    assert suite['protocol']['seeds'] == [0] and suite['protocol']['folds'] == 5
    assert suite['execution']['physical_gpu_ids'] == [1, 2, 3]
    from check_baseline_release import verify
    verify()
    workers = active_workers()
    counts = completed_counts()
    OUT.mkdir(parents=True, exist_ok=True)
    status = dict(status='running' if workers else 'complete' if counts == {'dual':30,'plugin':30} else 'ready_or_partial',
                  completed=counts, expected_per_route=30, active_workers=workers,
                  seed=0, unchanged_parent=True, gpu_allowed=[1,2,3])
    (OUT/'live_status.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    if not execute or workers:
        return
    if counts != {'dual':30, 'plugin':30}:
        for route in ('dual', 'plugin'):
            if not (OUT/route/'protocol.json').exists():
                command(f'validate_locked_{route}_20261030.py', '--prepare', log='suite_prepare.log')
        with ThreadPoolExecutor(4) as pool:
            jobs = [pool.submit(dual_neural), pool.submit(dual_cpu)]
            for physical in (2, 3):
                jobs.append(pool.submit(command, 'validate_locked_plugin_20261030.py', '--worker',
                                        gpu=str(physical), log=f'suite_plugin_gpu{physical}.log'))
            for job in jobs:
                job.result()
    audit_path = OUT/'plugin/AUDIT.json'
    if not audit_path.exists() or json.loads(audit_path.read_text()).get('status') != 'pass':
        command('audit_locked_plugin_20261030.py', gpu='2', log='suite_plugin_replay.log')
    command('summarize_locked_routes_20261030.py', '--final', log='suite_summary.log')
    command('report_locked_routes_20261030.py', log='suite_summary.log')
    verify()
    status.update(status='complete', completed=completed_counts(), active_workers=[])
    (OUT/'live_status.json').write_text(json.dumps(status, indent=2))
