"""Frozen-baseline inputs and native-band CNN--SNN views from real1MHz CH0."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,time,hashlib
import numpy as np,pandas as pd
from scipy.signal import firwin,upfirdn,sosfilt,butter,resample_poly
import _bootstrap as boot
from arc_model.preprocessing.rcr_cycle_v2 import estimate
from prepare_self462_12800_20261105 import entries,select
ROOT=boot.ROOT;OUT=ROOT/'outputs/self462_1m_baselines_20261105';PARENT=ROOT/'outputs/self462_12800_20261105';ORIGINAL=ROOT/'outputs/self462_feature_study_20261105'

def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def one(row):
 d=OUT/'records'/row['id'];d.mkdir(parents=True,exist_ok=True)
 if (d/'COMPLETE.json').exists():return json.loads((d/'COMPLETE.json').read_text())
 tick=time.time();s=OUT/'signals/records'/row['id'];src=json.loads((s/'COMPLETE.json').read_text());old=json.loads((ORIGINAL/'data/records'/row['id']/'COMPLETE.json').read_text());assert src['source_CH0_sha256']==old['CH0_sha256'] and src['source_header_sha256']==old['header_sha256']
 raw=np.load(s/'current1m.npy',mmap_mode='r');assert np.isfinite(raw).all() and hashlib.sha256(raw.tobytes()).hexdigest()==src['signal_sha256'];fs=1000000
 fir=firwin(401,20000,fs=fs,window=('kaiser',7.5));low=upfirdn(fir,raw,down=20)[:(len(raw)+19)//20];rectified=upfirdn(fir,abs(raw),down=20)[:len(low)]
 phase=estimate(sosfilt(butter(2,800,fs=50000,output='sos'),low)[::10],chunk_samples=10);ents=entries(phase,len(raw),fs)
 shape=sosfilt(butter(4,1000,fs=50000,output='sos'),low);prefix=np.r_[0.,np.cumsum(shape)]/50000;times=np.arange(len(prefix))/50000;hf=[]
 for band in [(10000,100000),(100000,400000)]:
  v=sosfilt(butter(3,band,fs=fs,btype='bandpass',output='sos'),raw);hf.append(np.r_[0.,np.cumsum(v*v)]/fs)
 blocks=[];hist=[];past={};previous={};last={}
 for e in ents:
  key=e['native'],e['seg'],e['epoch']
  if last.get(key)!=e['a']:previous.pop(key,None);past[key]=[]
  edges=e['edge'];dt=np.diff(edges/fs);lf=np.diff(np.interp(edges/fs,times,prefix))/dt;power=np.maximum(np.array([np.diff(q[edges])/dt for q in hf]),0);prob=power/np.maximum(power.sum(1,keepdims=True),1e-20);delta=np.zeros_like(prob) if key not in previous else prob-previous[key];previous[key]=prob
  blocks.append(np.vstack([lf,np.log10(np.maximum(power,1e-20)),delta]).astype('float32'));past.setdefault(key,[]).append(len(blocks)-1);hist.append(past[key][-9:].copy());last[key]=e['b']
 blocks=np.array(blocks);tab=pd.read_csv(PARENT/'cycles.csv');tab=tab[tab.record_id.eq(row['id'])];anchors=tab.anchor_seconds.to_numpy();selected,valid,native,support=select(ents,hist,phase,anchors,fs,float(row['onset_s']),float(row['end_s']),src['source_samples']/2000000)
 cnn=np.zeros((len(tab),9,5,64),'float32')
 for j,i in enumerate(selected):hh=hist[int(i)];cnn[j,-len(hh):]=blocks[hh]
 cnn[:,0,3:5]=0
 vmd=upfirdn(firwin(401,112500,fs=fs,window=('kaiser',7.5)),raw,down=4)[:(len(raw)+3)//4]
 pos=(anchors[:,None]-.02+np.arange(1000)[None,:]/50000)*50000
 assert pos.min()>=0 and (np.ceil(pos[:,-1])/50000<=anchors+1e-9).all()
 rect=np.interp(pos,np.arange(len(rectified)),rectified).astype('float32');signed=np.interp(pos,np.arange(len(low)),low).astype('float32')
 posv=(anchors[:,None]-.02+np.arange(5000)[None,:]/250000)*250000
 assert (np.ceil(posv[:,-1])/250000<=anchors+1e-9).all()
 fenet=np.interp(posv,np.arange(len(vmd)),vmd).astype('float32');t=resample_poly(signed,1,4,axis=-1).astype('float32');diff=np.gradient(t,axis=-1);temporal=np.stack([t,diff,np.gradient(diff,axis=-1)],1)[:,None];spectrum=abs(np.fft.rfft(t,axis=-1))[:,1:].astype('float32')
 np.savez_compressed(d/'inputs.npz',cnn=cnn,valid=valid,native=native,support=support,anchor=anchors,label=tab.label.to_numpy(),wave=rect,fenet_wave=fenet,temporal=temporal,spectrum=spectrum)
 r=dict(id=row['id'],status='complete',source_CH0_sha256=src['source_CH0_sha256'],signal1m_sha256=src['signal_sha256'],anchors=len(tab),kept=int(valid.sum()),excluded=int((~valid).sum()),native_fraction=float(native.mean()),seconds=time.time()-tick);save(d/'COMPLETE.json',r);print('RECORD1M',row['id'],r['kept'],round(r['seconds'],2),flush=True);return r

def prepare(cfg):
 if (OUT/'DATA_COMPLETE.json').exists():return
 rows=pd.read_csv(OUT/'manifest.csv').to_dict('records');receipts=[]
 with ProcessPoolExecutor(cfg['execution']['preparation_workers']) as pool:
  jobs=[pool.submit(one,r) for r in rows]
  for q in as_completed(jobs):receipts.append(q.result());save(OUT/'preparation_progress.json',dict(completed=len(receipts),total=len(rows)))
 parent=pd.read_csv(PARENT/'cycles.csv');frames=[];parts={k:[] for k in ['cnn_snn_1m','cnn_snn_2m_control','wave','fenet_wave','temporal','spectrum']};excluded=[]
 old2=np.load(PARENT/'native_full_rcr.npy',mmap_mode='r')
 for row in rows:
  z=np.load(OUT/'records'/row['id']/'inputs.npz');tt=parent[parent.record_id.eq(row['id'])].copy();np.testing.assert_array_equal(tt.anchor_seconds.to_numpy(),z['anchor']);np.testing.assert_array_equal(tt.label.to_numpy(),z['label']);keep=z['valid'];tt['previous_index']=tt.index;tt['source_group']=tt.record_id;tt['sample_id']=[f'{row["id"]}_{a:.8f}' for a in tt.anchor_seconds]
  for _,r in tt[~keep].iterrows():excluded.append(dict(record_id=row['id'],anchor_seconds=r.anchor_seconds,reason='1MHz9cycle support/guard; excluded from all five models'))
  frames.append(tt[keep]);parts['cnn_snn_1m'].append(z['cnn'][keep]);parts['cnn_snn_2m_control'].append(np.asarray(old2[tt.index.to_numpy()[keep]]))
  for k in ['wave','fenet_wave','temporal','spectrum']:parts[k].append(z[k][keep])
 t=pd.concat(frames,ignore_index=True);t.to_csv(OUT/'cycles.csv',index=False);assert t.record_id.nunique()==462 and t.condition_key.nunique()==156 and t.groupby('condition_key').label.nunique().min()==2
 for k,v in parts.items():
  arr=np.concatenate(v);assert np.isfinite(arr).all();np.save(OUT/(k+'.npy'),arr);del arr
 np.save(OUT/'reviewed.npy',np.arange(len(t)));splits=[dict(train=np.flatnonzero(t.fold.to_numpy()!=f).tolist(),test=np.flatnonzero(t.fold.to_numpy()==f).tolist()) for f in range(4)];save(OUT/'splits.json',splits);pd.DataFrame(excluded).to_csv(OUT/'cohort_exclusions.csv',index=False);pd.DataFrame(receipts).to_csv(OUT/'record_receipts.csv',index=False)
 save(OUT/'DATA_COMPLETE.json',dict(status='complete',records=462,conditions=156,parent_windows=len(parent),windows=len(t),excluded=len(parent)-len(t),normal=int(sum(t.label==0)),arc=int(sum(t.label==1)),source_rate=1000000,folds=4,same_cohort_all_models=True))
