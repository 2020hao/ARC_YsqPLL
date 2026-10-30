"""Suite dispatch for the twelve-arm head-only mechanism study."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import _bootstrap as boot

ROOT=boot.ROOT
OUT=ROOT/'outputs/phase_dynamic_evidence_20261030'


def active():
    result=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=p.read_bytes().split(b'\0')
            if args and b'python' in args[0] and any(Path(os.fsdecode(v)).name=='validate_phase_dynamic_evidence_20261030.py' for v in args[1:] if v):result.append(int(p.parent.name))
        except OSError:pass
    return result


def command(script,*args,gpu='',log):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(ROOT/'src'))
    with (OUT/log).open('a') as f:
        subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)


def run_study(suite,execute=False):
    from check_baseline_release import verify
    verify()
    assert set(suite['datasets'])=={'continuous5m','zenodo5501527'} and len(suite['arms'])==12
    OUT.mkdir(parents=True,exist_ok=True)
    workers=active();count=len(list((OUT/'runs').glob('*/*/f*/COMPLETE.json')))
    state=dict(status='running' if workers else 'complete' if (OUT/'COMPLETE.json').exists() else 'heads_complete_pending_audit' if count==30 else 'ready_or_partial',
               completed_cell_folds=count,expected_cell_folds=30,configurations=12,new_configurations=8,
               planned_head_fits=420,parity_head_fits=1,new_backbone_fits=0,active_workers=workers)
    (OUT/'live_status.json').write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2))
    if not execute or workers:return
    command('prepare_phase_dynamic_evidence_20261030.py',log='suite_prepare.log')
    for flag,name in [('--parity','parity.json'),('--prepare','protocol.json')]:
        if not (OUT/name).exists():command('validate_phase_dynamic_evidence_20261030.py',flag,log='suite_prepare.log')
    if count!=30:
        with ThreadPoolExecutor(3) as pool:
            tasks=[pool.submit(command,'validate_phase_dynamic_evidence_20261030.py','--worker',log=f'suite_worker{i}.log') for i in range(3)]
            for task in tasks:task.result()
    if not (OUT/'REAL_MODEL_REPLAY.json').exists():command('validate_phase_dynamic_evidence_20261030.py','--replay',gpu='1',log='suite_replay.log')
    if not (OUT/'independent_audit.json').exists():command('audit_phase_dynamic_evidence_20261030.py',log='suite_audit.log')
    command('summarize_phase_dynamic_evidence_20261030.py',log='suite_summary.log')
    command('report_phase_dynamic_evidence_20261030.py',log='suite_summary.log')
    verify()
