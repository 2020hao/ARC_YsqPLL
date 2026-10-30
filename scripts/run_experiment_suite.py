"""Default study entry point: require both datasets, preflight all, then train/evaluate separately."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import _bootstrap as boot
from arc_model.configuration import load_config
from arc_model.data.dataset import PreparedDataset
from arc_model.training.pipeline import run
from arc_model.evaluation.multilevel import evaluate

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--suite',default='configs/experiment_suite.yaml')
 p.add_argument('--execute',action='store_true',help='Default preflight only; never partial single-dataset execution')
 p.add_argument('--device',default='cpu')
 p.add_argument('--model',help='One model config; still runs BOTH datasets')
 p.add_argument('--seed',type=int);p.add_argument('--mode');p.add_argument('--variant',type=int)
 a=p.parse_args();suite=yaml.safe_load((boot.ROOT/a.suite).read_text())
 if suite.get('specialized_study'):
  # Specialized coordinate experiments retain both datasets and their own reviewed snapshot.
  if suite['specialized_study']=='fenet_qualification_20261030':
   from qualify_fenet_20261030 import run_study
  elif suite['specialized_study']=='fusion_qualification_20261030':
   from qualify_fusion_current_20261030 import run_study
  elif suite['specialized_study']=='grouped_baseline_standard_20261029':
   from freeze_grouped_baselines import run_study
  elif suite['specialized_study']=='hydra_alternating_protocol_20261029':
   from validate_hydra_alternating import run_study
  elif suite['specialized_study']=='paper_baseline_completion_20261029':
   from complete_paper_baselines import run_study
  elif suite['specialized_study']=='dynamic_restore_20261029':
   from validate_dynamic_restore import run_study
  elif suite['specialized_study']=='shared_eight_e2e_20261029':
   from train_shared_eight_e2e import run_study
  elif suite['specialized_study']=='shared_eight_study_20261029':
   from validate_shared_eight import run_study
  elif suite['specialized_study']=='input_frontend_pilot_20261029':
   from validate_input_frontend import run_study
  elif suite['specialized_study']=='rcr_cft_detector_20261028':
   from run_rcr_cft_study import run_study
  elif suite['specialized_study']=='rcr_state_phase_adapter_20261028':
   from run_state_phase_adapter_study import run_study
  elif suite['specialized_study']=='rcr_three_backbones_20261028':
   from run_tcv_study import run_study
  elif suite['specialized_study']=='reference_recovery_20261026':
   from run_reference_recovery_study import run_study
  elif suite['specialized_study']=='reference_native_full_20261026':
   from run_reference_native_full_study import run_study
  elif suite['specialized_study']=='head_fusion_validation_20261028':
   from run_head_fusion_validation_study import run_study
  elif suite['specialized_study']=='continuous_pll_validation_20261027_v2':
   from run_continuous_pll_validation_study_v2 import run_study
  elif suite['specialized_study']=='continuous_pll_validation_20261027':
   from run_continuous_pll_validation_study import run_study
  elif suite['specialized_study']=='dynamic_mask_ablation_20261027':
   from run_dynamic_mask_ablation_study import run_study
  elif suite['specialized_study']=='local_phase_balanced_20261027':
   from run_local_phase_balanced_study import run_study
  elif suite['specialized_study']=='protected_phase_branch_20261027':
   from run_protected_phase_branch_study import run_study
  elif suite['specialized_study']=='selective24_harmonic_20261027':
   from run_selective_harmonic_study import run_study
  elif suite['specialized_study']=='shared_clock_held_20261026':
   from run_shared_clock_held_study import run_study
  elif suite['specialized_study']=='shared_clock_original_control_20261026':
   from run_shared_clock_original_control_study import run_study
  elif suite['specialized_study']=='shared_clock_fusion_20261026':
   from run_shared_clock_fusion_study import run_study
  else:raise ValueError('Unregistered specialized study')
  run_study(suite,execute=a.execute)
  return
 if set(suite['datasets'])!={'new5m','zenodo5501527'} or suite['pool_datasets']:raise ValueError('Both registered datasets must run separately')
 if a.model:suite['models']=[a.model]
 if a.seed is not None:suite['seeds']=[a.seed]
 if a.mode is not None:suite['modes']=[a.mode]
 if a.variant is not None:suite['variants']=[a.variant]
 errors=[];jobs=[]
 for name,entry in suite['datasets'].items():
  try:
   if name=='zenodo5501527':
    receipt=json.loads((boot.ROOT/entry['download_receipt']).read_text())
    assert receipt['status']=='complete' and len(receipt['files'])==27
    annotations=pd.read_csv(boot.ROOT/entry['annotations'])
    if not len(annotations) or not (annotations.status=='reviewed').all():raise ValueError('公共数据下载完整；需审阅annotations.csv区间标签，再生成公共模型缓存')
   d=PreparedDataset(boot.ROOT/yaml.safe_load((boot.ROOT/entry['pipeline']).read_text())['paths']['prepared'],boot.ROOT/yaml.safe_load((boot.ROOT/entry['pipeline']).read_text())['paths']['manifests'])
   timing=pd.read_csv(boot.ROOT/entry['window_metadata'])
   if len(timing)!=len(d.record_ids):raise ValueError('Timing and windows must match row-for-row')
   if not np.array_equal(timing.record_id,d.record_ids):raise ValueError('Timing record order mismatch')
   for model in suite['models']:
    for seed in suite['seeds']:
     for variant in suite['variants']:
      for mode in suite['modes']:
       c=load_config(boot.ROOT/model,boot.ROOT/entry['pipeline']);c.seed=seed;c.variant=variant;c.mode=mode;c.device=a.device;c.paths['output']=str(Path(suite['output'])/name);c.validate()
       required=['features','record_ids']+(['mbwvd'] if c.model=='ArcFENet' else ['fusion_temporal','fusion_spectrum'] if c.model=='ArcFusionNet' else [f'hydra_features_seed{seed}'])
       for key in required:d.array(key)
       folder=c.path('output')/f'{c.model}_v{c.variant}_{c.mode}_s{c.seed}'
       if any(folder.glob('fold*.json')):raise ValueError(f'Existing completed outputs: {folder}; version suite output before rerun')
       jobs.append((name,entry,c,d,timing))
  except (OSError,ValueError,AssertionError,KeyError) as e:errors.append(dict(dataset=name,reason=str(e) or 'Incomplete receipt/cache'))
 report=dict(status='ready' if not errors else 'prerequisites_missing',configured_runs=len(suite['datasets'])*len(suite['models'])*len(suite['seeds'])*len(suite['variants'])*len(suite['modes']),preflight_ready_runs=len(jobs),errors=errors)
 out=boot.ROOT/suite['output'];out.mkdir(parents=True,exist_ok=True);(out/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 if errors:
  if a.execute:raise SystemExit('No model launched: complete both datasets first')
  return
 if not a.execute:return
 summaries=[]
 for name,entry,c,d,timing in jobs:
  folder=run(c);rows=[]
  for fold in range(len(d.splits)):
   z=np.load(folder/f'fold{fold}_windows.npz');ix=z['indices'];frame=timing.iloc[ix].copy()
   frame['stream_id']=[d.ledger[int(r)].get('stream_id', d.ledger[int(r)]['source_file']) for r in z['record_ids']]
   frame['condition_key']=[str(d.ledger[int(r)]['condition_key']) for r in z['record_ids']]
   
   if c.variant==0:
    frame['start_seconds']=frame.fixed_start_seconds
    frame['end_seconds']=frame.fixed_start_seconds+.02
   frame['label']=z['labels'];frame['score']=z['scores'] if c.model=='HYDRA' else z['probabilities'];rows.append(frame)
  frame=pd.concat(rows,ignore_index=True);frame.to_csv(folder/'cycle_predictions.csv',index=False)
  def intervals(key):
   path=entry.get(key)
   table=pd.read_csv(boot.ROOT/path) if path and (boot.ROOT/path).exists() else None
   return table if table is not None and len(table) else None
  alarm=suite['evaluation']['alarm']
  threshold=suite['evaluation']['threshold']['HYDRA' if c.model=='HYDRA' else 'neural']
  metrics=evaluate(frame,threshold,intervals('events'),intervals('normal_intervals'),alarm['consecutive_cycles'],alarm['event_deadline_seconds'])
  (folder/'multilevel_metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
  summaries.append(dict(dataset=name,model=c.model,variant=c.variant,mode=c.mode,seed=c.seed,metrics=metrics))
 (out/'suite_results.json').write_text(json.dumps(summaries,indent=2,allow_nan=False))
if __name__=='__main__':main()
