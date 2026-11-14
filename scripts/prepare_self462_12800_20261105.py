"""Matched-cohort current-only12.8k frontend and same-lowband high-rate control."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,time
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfilt
import _bootstrap as boot
from arc_model.preprocessing.rcr_cycle_v2 import estimate,CycleRCRConfig
ROOT=boot.ROOT;OUT=ROOT/'outputs/self462_12800_20261105';PARENT=ROOT/'outputs/self462_feature_study_20261105'

def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))

def entries(clock,n,fs):
 tx=clock.estimate_time_s.to_numpy();ph=clock.phase_rad.to_numpy();valid=clock.phase_usable.to_numpy();segments=clock.reference_segment.to_numpy();epochs=clock.reference_epoch.to_numpy();out=[]
 for seg in np.unique(segments[valid]):
  ids=np.flatnonzero(valid&(segments==seg))
  for ix in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
   if len(ix)<2:continue
   t,p=tx[ix],ph[ix]
   if not np.all(np.diff(p)>0):continue
   turns=np.arange(np.ceil(p[0]/(2*np.pi)),np.floor(p[-1]/(2*np.pi))+1)*2*np.pi;bounds=np.interp(turns,p,t)
   for a,b in zip(bounds[:-1],bounds[1:]):
    if not 1/55<=b-a<=1/45:continue
    av=min(int(np.searchsorted(t,b,side='left')),len(t)-1);grid=np.interp(np.linspace(np.interp(a,t[:av+1],p[:av+1]),np.interp(b,t[:av+1],p[:av+1]),65),p[:av+1],t[:av+1]);edge=np.ceil(grid*fs).astype('int64');edge[0]=int(np.ceil(a*fs));edge[-1]=int(np.ceil(b*fs))
    if (np.diff(edge)>0).all() and edge[-1]<=n:out.append(dict(a=int(edge[0]),b=int(edge[-1]),av=float(t[av]),seg=int(seg),epoch=int(epochs[ix[av]]),native=True,edge=edge))
 for a in range(0,n-fs//50+1,fs//50):out.append(dict(a=a,b=a+fs//50,av=(a+fs//50)/fs,seg=-1,epoch=-1,native=False,edge=np.rint(np.linspace(a,a+fs//50,65)).astype('int64')))
 return out

def profiles(signal,fs,ents,edge_fs):
 shape=sosfilt(butter(4,1000,fs=fs,output='sos'),signal);prefix=[np.r_[0.,np.cumsum(shape)]/fs]
 for band in [(1000,3000),(3000,5000)]:
  x=sosfilt(butter(3,band,fs=fs,btype='bandpass',output='sos'),signal);prefix.append(np.r_[0.,np.cumsum(x*x)]/fs)
 times=np.arange(len(prefix[0]))/fs;out=[];hist=[];previous={};past={};last={}
 for e in ents:
  key=e['native'],e['seg'],e['epoch']
  if last.get(key)!=e['a']:previous.pop(key,None);past[key]=[]
  grid=e['edge']/edge_fs;dt=np.diff(grid);p=np.array([np.diff(np.interp(grid,times,q))/dt for q in prefix]);p[1:]=np.maximum(p[1:],0);prob=p[1:]/np.maximum(p[1:].sum(1,keepdims=True),1e-20);delta=np.zeros_like(prob) if key not in previous else prob-previous[key];previous[key]=prob
  out.append(np.vstack([p[0],np.log10(np.maximum(p[1:],1e-20)),delta]).astype('float32'));past.setdefault(key,[]).append(len(out)-1);hist.append(past[key][-9:].copy());last[key]=e['b']
 return np.array(out),hist

def select(ents,hist,clock,anchors,fs,on,off,duration,fixed=False):
 idx=[];valid=[];native=[];support=[]
 tx=clock.estimate_time_s.to_numpy();qphase=clock.phase_usable.to_numpy();seg=clock.reference_segment.to_numpy();epoch=clock.reference_epoch.to_numpy()
 ends=np.array([e['b']/fs for e in ents]);av=np.array([e['av'] for e in ents]);isnative=np.array([e['native'] for e in ents]);segs=np.array([e['seg'] for e in ents]);epochs=np.array([e['epoch'] for e in ents])
 for a in anchors:
  eligible=(av<=a+1e-10)&(ends<=a+1e-10)&(a-ends<=.04+1e-10);cur=[];qi=np.searchsorted(tx,a+1e-10,side='right')-1
  if not fixed and qi>=0 and qphase[qi]:cur=np.flatnonzero(eligible&isnative&(segs==seg[qi])&(epochs==epoch[qi]))
  if not len(cur):cur=np.flatnonzero(eligible&~isnative)
  assert len(cur)>0;i=int(cur[np.argmax(ends[cur])]);hh=hist[i]
  if a<on-.04:left,right=.04,on-.04
  elif on+.04<=a<=off-.04:left,right=on+.04,off-.04
  elif a>=off+.04:left,right=off+.04,duration-.04
  else:left,right=duration+1,0
  ok=len(hh)==9 and ents[hh[0]]['a']/fs>=left-1e-10 and ends[i]<=right+1e-10
  idx.append(i);valid.append(ok);native.append(ents[i]['native']);support.append([ents[hh[0]]['a']/fs,ends[i],av[i]])
 return np.array(idx),np.array(valid),np.array(native),np.array(support)

def one(job):
 row=job;dest=OUT/'records'/row['id'];dest.mkdir(parents=True,exist_ok=True)
 if (dest/'COMPLETE.json').exists():return json.loads((dest/'COMPLETE.json').read_text())
 tick=time.time();signals=OUT/'signals/records'/row['id'];src=json.loads((signals/'COMPLETE.json').read_text());old=json.loads((PARENT/'data/records'/row['id']/'COMPLETE.json').read_text());assert src['CH0_sha256']==old['CH0_sha256'] and src['header_sha256']==old['header_sha256']
 sig=np.load(signals/'signals.npz');low50=sig['low50'];low128=sig['low128'];orig=np.load(PARENT/'data/records'/row['id']/'rcr.npz');anchors=orig['anchor'];on=float(row['onset_s']);off=float(row['end_s']);duration=src['native_duration_s']
 nativeclock=estimate(sosfilt(butter(2,800,fs=50000,output='sos'),low50)[::10],chunk_samples=10)
 highentries=entries(nativeclock,src['native_samples'],2000000);control,highhist=profiles(low50,50000,highentries,2000000)
 assert control.shape==orig['base'].shape,(row['id'],control.shape,orig['base'].shape)
 chosen=orig['selected'];boundaryerror=max(abs(np.array([highentries[int(i)]['b']/2000000 for i in chosen])-orig['support'][:,1]))*2000000
 # The matching estimate must recover the same native windows, allowing last-bit boundary rounding.
 assert boundaryerror<=2.01,(row['id'],boundaryerror)
 lferror=float(abs(control[:,0]-orig['base'][:,0]).max());control[:,0]=orig['base'][:,0]
 np.testing.assert_array_equal(np.array([highhist[int(i)] for i in chosen]),orig['hist'])
 high=control[orig['hist']];high[:,0,3:5]=0
 cfg=CycleRCRConfig(reference_fs=12800);clock=estimate(sosfilt(butter(2,800,fs=12800,output='sos'),low128),config=cfg,chunk_samples=256)
 lowentries=entries(clock,len(low128),12800);lowprofiles,lowhist=profiles(low128,12800,lowentries,12800);validall=np.ones(len(anchors),bool);rows=[]
 np.savez_compressed(dest/'native_lowband_rcr.npz',input=high,anchor=anchors,label=orig['label'])
 for ref,fixed in [('low128_rcr',False),('low128_fixed',True)]:
  selected,valid,native,support=select(lowentries,lowhist,clock,anchors,12800,on,off,duration,fixed)
  x=np.zeros((len(anchors),9,5,64),'float32')
  for j,i in enumerate(selected):
   hh=lowhist[int(i)];x[j,-len(hh):]=lowprofiles[hh]
  x[:,0,3:5]=0;validall&=valid
  np.savez_compressed(dest/(ref+'.npz'),input=x,anchor=anchors,label=orig['label'],valid=valid,native=native,support=support)
  rows.append(dict(arm=ref,valid=int(valid.sum()),native_fraction=float(native.mean())))
 np.save(dest/'keep.npy',validall);clock.to_csv(dest/'low_reference_clock.csv',index=False)
 receipt=dict(id=row['id'],status='complete',anchors=len(anchors),retained=int(validall.sum()),excluded=int((~validall).sum()),native_boundary_max_error_samples=float(boundaryerror),native_LF_reconstruction_max_abs_error=lferror,low_fs=12800,low_phase_update_seconds=round(.002*12800)/12800,rows=rows,seconds=time.time()-tick)
 save(dest/'COMPLETE.json',receipt);print('LOWRATE',row['id'],receipt['retained'],round(receipt['seconds'],2),flush=True);return receipt

def prepare(cfg):
 if (OUT/'DATA_COMPLETE.json').exists():return
 rows=pd.read_csv(OUT/'manifest.csv').to_dict('records');receipts=[]
 with ProcessPoolExecutor(cfg['execution']['preparation_workers']) as pool:
  jobs=[pool.submit(one,r) for r in rows]
  for f in as_completed(jobs):
   receipts.append(f.result());save(OUT/'preparation_progress.json',dict(completed=len(receipts),total=len(rows)))
 parent=pd.read_csv(PARENT/'cycles.csv');parts={a:[] for a in cfg['arms']};tables=[];exclusions=[];order=[]
 for row in rows:
  rid=row['id'];directory=OUT/'records'/rid;old=np.load(PARENT/'data/records'/rid/'rcr.npz');keep=np.load(directory/'keep.npy');tt=parent[parent.record_id.eq(rid)].copy();assert len(tt)==len(keep);np.testing.assert_array_equal(tt.anchor_seconds.to_numpy(),old['anchor']);np.testing.assert_array_equal(tt.label.to_numpy(),old['label']);tt['parent_index']=tt.index
  for _,r in tt[~keep].iterrows():exclusions.append(dict(record_id=rid,anchor_seconds=r.anchor_seconds,reason='low-rate completed9cycle support or40ms guard; excluded from ALL arms'))
  tables.append(tt[keep]);order.extend(tt.index[keep]);full=old['base'][old['hist']].copy();full[:,0,3:5]=0;parts['native_full_rcr'].append(full[keep])
  for arm in ['native_lowband_rcr','low128_rcr','low128_fixed']:parts[arm].append(np.load(directory/(arm+'.npz'))['input'][keep])
 table=pd.concat(tables,ignore_index=True);table.to_csv(OUT/'cycles.csv',index=False);pd.DataFrame(exclusions).to_csv(OUT/'cohort_exclusions.csv',index=False)
 assert table.record_id.nunique()==462 and table.condition_key.nunique()==156
 assert table.groupby('condition_key').label.nunique().min()==2
 for arm in cfg['arms']:
  x=np.concatenate(parts[arm]);assert x.shape==(len(table),9,5,64) and np.isfinite(x).all();np.save(OUT/(arm+'.npy'),x)
 pd.DataFrame(receipts).to_csv(OUT/'record_receipts.csv',index=False)
 save(OUT/'DATA_COMPLETE.json',dict(records=462,parent_anchors=len(parent),common_anchors=len(table),excluded=len(parent)-len(table),normal=int(sum(table.label==0)),arc=int(sum(table.label==1)),conditions=156,arms=cfg['arms'],folds=4,cohort_matched_before_training=True))
