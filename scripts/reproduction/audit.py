from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
O=Path(__file__).resolve().parents[2]/'outputs/current_correction_20261109';R=O.parents[1];sys.path.insert(0,str(R/'src'))
from arc_model.modules.rcr_dynamic_branch_1mhz.fusion import corrected_margin
from arc_model.modules.rcr_dynamic_branch_1mhz.fitting import replay
rows=[]
selected=set(json.loads((O/'protocol.json').read_text()).get('selected_datasets',['self462','zenodo5501527']))
expected_folds=sum(4 if ds=='self462' else 5 for ds in selected)*3
for ds,nf in [('self462',4),('zenodo5501527',5)]:
 if ds not in selected:continue
 t=pd.read_csv(O/'data'/ds/'cycles.csv');x=np.load(O/'data'/ds/'dynamic.npy');keep=np.load(O/'data'/ds/'reviewed.npy');assert len(keep)=={'self462':125051,'zenodo5501527':1357}[ds]
 for model in ['ArcFENet','HYDRA','Fusion-current']:
  coverage=np.zeros(len(t),int)
  for f in range(nf):
   d=O/'runs'/ds/model/f'f{f}'
   if not (d/'HEAD_COMPLETE.json').exists():continue
   r=np.load(d/'outer/indices.npz');tr,te=r['train'],r['test'];coverage[te]+=1;p=np.load(d/'paired_predictions.npz');np.testing.assert_array_equal(p['indices'],te);np.testing.assert_array_equal(p['labels'],t.label.to_numpy()[te]);state=dict(np.load(d/'correction_state.npz'));np.testing.assert_array_equal(state['fit_indices'],tr)
   for col in ['record_id','source_group','condition_key']:assert not set(t.iloc[tr][col])&set(t.iloc[te][col])
   np.testing.assert_allclose(state['mu'],x[tr][:,state['keep']].mean(0));np.testing.assert_allclose(state['sd'],np.maximum(x[tr][:,state['keep']].std(0),1e-6))
   raw=p['baseline']
   if ds=='self462':
    names={'ArcFENet':'ArcFENet_Baseline','HYDRA':'HYDRA_Baseline','Fusion-current':'ArcFusionNet_Current_Baseline'}
    old=np.load(R/'outputs/self462_1m_baselines_20261105/runs'/names[model]/f'f{f}_s0/predictions.npz')
    np.testing.assert_array_equal(raw>=0,old['prediction'])
   delta,dist=replay(state,x[te]);expected=corrected_margin(state,raw,x[te]);np.testing.assert_allclose(expected,p['RCR_dynamic_soft_gate'],atol=1e-5,rtol=2e-5);np.testing.assert_allclose(float(state['calibration_slope'])*raw+float(state['calibration_intercept'])+delta,p['RCR_dynamic_no_gate'],atol=1e-5,rtol=2e-5)
   rows.append(dict(dataset=ds,model=model,fold=f,test=len(te),replay_max_abs=float(abs(expected-p['RCR_dynamic_soft_gate']).max()),train_only_statistics=True,group_disjoint=True))
  if len([r for r in rows if r['dataset']==ds and r['model']==model])==nf:assert np.all(coverage[keep]==1) and coverage.sum()==len(keep)
files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest()for pkg in ['baseline_arc_fenet','baseline_arc_fusion','baseline_hydra']for p in (R/'src/arc_model/implementations'/pkg).glob('*.py')}
report=dict(status='passed_complete' if len(rows)==expected_folds else 'passed_partial',folds=len(rows),rows=rows,sealed_source_hashes=files,metric_scope='condition-equal BAcc; other classification metrics pooled; seed0',limitations=['exposed development folds','no fixed20 same-information comparator','no independent SNN or PLL attribution'])
(O/'AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(report['status'],len(rows))
