from pathlib import Path
import sys,json,time,os,traceback
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
O=Path(__file__).resolve().parents[2]/'outputs/current_correction_20261109';ROOT=O.parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_self462_1m_baselines_20261105 import metrics
from arc_model.modules.rcr_dynamic_branch_1mhz import fitting
from arc_model.modules.rcr_dynamic_branch_1mhz.fusion import corrected_margin
import torch
torch.set_num_threads(3)
def save(p,x):
 p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
def calibration(z,y):
 def fun(v):
  a=np.exp(v[0]);b=v[1];s=a*z+b;err=expit(s)-y
  return np.mean(np.logaddexp(0,s)-y*s),np.array([np.mean(err*a*z),err.mean()])
 r=minimize(fun,[0.,0.],jac=True,method='L-BFGS-B',bounds=[(-5,5),(-20,20)]);assert r.success,r.message
 return float(np.exp(r.x[0])),float(r.x[1])
def fit(ds,model,f):
 d=O/'runs'/ds/model/f'f{f}';t=pd.read_csv(O/'data'/ds/'cycles.csv');x=np.load(O/'data'/ds/'dynamic.npy');y=t.label.to_numpy();g=t.condition_key.to_numpy();role=np.load(d/'outer/indices.npz');tr,te=role['train'],role['test'];raw=np.full(len(t),np.nan);seen=np.zeros(len(t),int)
 for i in range(2):
  ids=np.load(d/f'inner{i}/indices.npz');p=np.load(d/f'inner{i}/predictions.npz');np.testing.assert_array_equal(p['indices'],ids['test']);assert not set(g[ids['train']])&set(g[p['indices']]);raw[p['indices']]=p['margin'];seen[p['indices']]+=1
 assert np.all(seen[tr]==1) and np.all(seen[te]==0)
 pred=np.load(d/'outer/predictions.npz');np.testing.assert_array_equal(pred['indices'],te);raw[te]=pred['margin'];assert np.isfinite(raw[np.r_[tr,te]]).all()
 # Excluded candidates never enter fitted statistics or loss.
 raw=np.nan_to_num(raw);a,b=calibration(raw[tr],y[tr]);cal=a*raw+b
 delta,dist,state=fitting.fit(x,y,cal,tr,np.arange(36),f,device='cpu',epochs=80)
 gate_th=max(float(np.quantile(dist[tr[y[tr]==0]],.95)),1e-6)
 state.update(calibration_slope=a,calibration_intercept=b,gate_threshold=gate_th,fusion='whole_path',strength=1.)
 soft=corrected_margin(state,raw,x);ungated=cal+delta
 np.savez(d/'correction_state.npz',**state);np.savez(d/'paired_predictions.npz',indices=te,labels=y[te],baseline=raw[te],calibration_only=cal[te],RCR_dynamic_no_gate=ungated[te],RCR_dynamic_soft_gate=soft[te]);np.savez(d/'honest_train_scores.npz',indices=tr,margin=raw[tr]);save(d/'HEAD_COMPLETE.json',dict(status='complete',train=len(tr),test=len(te),calibration_slope=a,calibration_intercept=b,gate_threshold=gate_th,coefficient_l2=float(np.linalg.norm(state['coef'])),outer_labels_used_for_fitting=False,head_epochs=80))
def summary():
 selected=set(json.loads((O/'protocol.json').read_text())['selected_datasets'])
 rows=[];groups=[];deltas=[]
 for ds,nf in [('self462',4),('zenodo5501527',5)]:
  if ds not in selected or not (O/'data'/ds/'cycles.csv').is_file():continue
  t=pd.read_csv(O/'data'/ds/'cycles.csv');keep=np.load(O/'data'/ds/'reviewed.npy')
  for model in ['ArcFENet','HYDRA','Fusion-current']:
   paths=[O/'runs'/ds/model/f'f{f}'/'paired_predictions.npz'for f in range(nf)]
   if not all(p.exists()for p in paths):continue
   records=[dict(np.load(p)) for p in paths];ids=np.concatenate([r['indices']for r in records]);order=np.argsort(ids);ids=ids[order];np.testing.assert_array_equal(ids,keep);y=t.iloc[ids].label.to_numpy();g=t.iloc[ids].condition_key.to_numpy();scores={arm:np.concatenate([r[arm]for r in records])[order]for arm in ['baseline','calibration_only','RCR_dynamic_no_gate','RCR_dynamic_soft_gate']}
   for arm,s in scores.items():
    row=dict(dataset=ds,model=model,arm=arm,N=len(ids),seed=0,**metrics(y,s,g,0));row['FNR']=1-row['Recall'];rows.append(row)
    for gr in np.unique(g):
     m=g==gr;groups.append(dict(dataset=ds,model=model,arm=arm,condition=gr,**metrics(y[m],s[m],g[m],0)))
   for arm in list(scores)[1:]:
    db=[]
    for gr in np.unique(g):
     m=g==gr;db.append(metrics(y[m],scores[arm][m],g[m],0)['BAcc']-metrics(y[m],scores['baseline'][m],g[m],0)['BAcc'])
    db=np.array(db);rng=np.random.default_rng(20261109);boot=db[rng.integers(0,len(db),(10000,len(db)))].mean(1);lo,hi=np.quantile(boot,[.025,.975]);before=scores['baseline']>=0;after=scores[arm]>=0
    deltas.append(dict(dataset=ds,model=model,arm=arm,BAcc_delta_pp=db.mean()*100,CI_low_pp=lo*100,CI_high_pp=hi*100,improved_groups=int(sum(db>0)),worse_groups=int(sum(db<0)),tied_groups=int(sum(db==0)),corrected_errors=int(sum((before!=y)&(after==y))),new_errors=int(sum((before==y)&(after!=y)))))
 pd.DataFrame(rows).to_csv(O/'完整性能指标.csv',index=False);pd.DataFrame(groups).to_csv(O/'逐工况指标.csv',index=False);pd.DataFrame(deltas).to_csv(O/'配对增益与错误转移.csv',index=False)
 lines=['# 当前完整队列：三基准输出校正验证','', '原骨干与预处理不变；自采462条/125051窗口/156工况四折，公共1357窗口/13组五折；seed0开发验证。自采外层得分逐项复用定版结果，公共按当前1357队列重训。所有新增校准、标准化、PCA和线性校正均使用外层训练组，骨干分数由内层两折组隔离产生。','', '固定80轮线性头、强度1、正常训练距离95%分位、门控软度0.15，不用测试集选择融合方式。无门控对照使用同一校准与动态系数。区间按条件组配对重采样，未作多重比较校正。']
 if rows:
  df=pd.DataFrame(rows)
  for ds,sub in df.groupby('dataset'):
   lines += ['', '## '+ds,'','|模型|接入方式|BAcc（条件等权）|Acc|Precision|Recall|FPR|F1|PR-AUC|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
   for _,r in sub.iterrows():lines.append('|'+r.model+'|'+r.arm+'|'+'|'.join(f'{r[k]*100:.2f}'for k in ['BAcc','Acc','Precision','Recall','FPR','F1','AP'])+'|')
  lines+=['','## 配对分析','']
  for r in deltas:
   if r['arm']=='RCR_dynamic_soft_gate':lines.append(f"- {r['dataset']} / {r['model']}：BAcc变化{r['BAcc_delta_pp']:+.2f} pp，组配对95%区间[{r['CI_low_pp']:+.2f}, {r['CI_high_pp']:+.2f}]；修正原错误{r['corrected_errors']}个，新增错误{r['new_errors']}个。")
 lines+=['','尚未完成的组合不填入结果；历史38374/1361队列结果不混入。完整动态校正相对基准的变化不能单独归因于PLL；本次没有固定窗同信息对照。FENet仍为本地数值重建，Fusion为仅电流适配。']
 (O/'结果与分析.md').write_text('\n'.join(lines))
 return len(rows)
if __name__=='__main__':
 selected=set(json.loads((O/'protocol.json').read_text()).get('selected_datasets',['self462','zenodo5501527']))
 while True:
  pending=0
  for ds,nf in [('self462',4),('zenodo5501527',5)]:
   if ds not in selected:continue
   for model in ['ArcFENet','HYDRA','Fusion-current']:
    for f in range(nf):
     d=O/'runs'/ds/model/f'f{f}'
     if (d/'HEAD_COMPLETE.json').exists():continue
     pending+=1
     if not all((d/sub/'COMPLETE.json').exists()for sub in ['outer','inner0','inner1']):continue
     try:fit(ds,model,f);print('HEAD_DONE',ds,model,f,flush=True)
     except Exception:save(d/'HEAD_FAILED.json',dict(traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True);raise
  n=summary();save(O/'STATUS.json',dict(pending_head_folds=pending,complete_metric_rows=n,updated=time.time()))
  if pending==0:
   import subprocess
   subprocess.run([sys.executable,str(Path(__file__).with_name('finalize.py'))],check=True)
   save(O/'COMPLETE.json',dict(status='complete',datasets=sorted(selected),pairs=3*len(selected),folds=3*sum(4 if ds=='self462' else 5 for ds in selected),metric_rows=n));break
  time.sleep(25)
