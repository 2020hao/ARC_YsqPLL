"""Read-only native CH0 feature extraction; run beside copied RCR/SOGI code on data host."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import os,json,csv,time,hashlib,argparse
from self462_recording import read_recording
import numpy as np
from scipy.signal import butter,sosfilt,firwin,upfirdn
import _bootstrap  # noqa: F401
from arc_model.preprocessing.rcr_cycle_v2 import estimate
from arc_model.preprocessing.pll import robust_pll

FEATURE_NAMES=['log_rms','crest_factor','skewness','kurtosis','relative_diff_rms','zero_fraction','half_energy_asymmetry','THD_2_to_10','harmonic3_ratio','harmonic5_ratio','log_mid_power','log_lowHF_power','log_highHF_power','highHF_fraction','mid_fraction','lowHF_burst','highHF_burst','lowHF_entropy','highHF_entropy','lowHF_peak_concentration','highHF_peak_concentration','lowHF_zero_neighborhood','highHF_zero_neighborhood','lowHF_half_asymmetry','highHF_half_asymmetry','lowHF_peak_sin','lowHF_peak_cos','highHF_peak_sin','highHF_peak_cos','lowHF_phase_concentration','highHF_phase_concentration','lowHF_distribution_MAD','highHF_distribution_MAD','lowHF_distribution_JS','highHF_distribution_JS','lowHF_cosine_distance','highHF_cosine_distance','lowHF_peak_drift','highHF_peak_drift','relative_rms_change']

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  while b:=f.read(8<<20):h.update(b)
 return h.hexdigest()

def candidates(tx,ph,valid,segments,epochs,n,fs):
 entries=[]
 for seg in np.unique(segments[valid]):
  ids=np.flatnonzero(valid&(segments==seg))
  # Split discontinuities even when caller segment tags are coarse.
  for ix in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
   if len(ix)<2:continue
   t=tx[ix];p=ph[ix]
   if not np.all(np.diff(p)>0):continue
   turns=np.arange(np.ceil(p[0]/(2*np.pi)),np.floor(p[-1]/(2*np.pi))+1)*2*np.pi
   bounds=np.interp(turns,p,t)
   for a,b in zip(bounds[:-1],bounds[1:]):
    if not 1/55<=b-a<=1/45:continue
    av=int(np.searchsorted(t,b,side='left'));av=min(av,len(t)-1)
    grid=np.interp(np.linspace(np.interp(a,t[:av+1],p[:av+1]),np.interp(b,t[:av+1],p[:av+1]),65),p[:av+1],t[:av+1])
    edges=np.ceil(grid*fs).astype('int64');edges[0]=int(np.ceil(a*fs));edges[-1]=int(np.ceil(b*fs))
    if (np.diff(edges)>0).all() and edges[-1]<=n:entries.append(dict(a=edges[0],b=edges[-1],av=float(t[av]),seg=int(seg),epoch=int(epochs[ix[av]]),native=True,edges=edges))
 for a in range(0,n-fs//50+1,fs//50):
  entries.append(dict(a=a,b=a+fs//50,av=(a+fs//50)/fs,seg=-1,epoch=-1,native=False,edges=np.rint(np.linspace(a,a+fs//50,65)).astype('int64')))
 return entries

def describe(shape,power,burst,low,a,b,previous=None):
 eps=1e-20
 xx=low[max(0,int(a*50000)):min(len(low),int(b*50000))];xx=xx[::max(1,len(xx)//2000)]
 rms=max(float(np.sqrt(np.mean(xx*xx))),1e-12);mu=xx.mean();sd=max(float(xx.std()),1e-12)
 n=len(xx);amp=np.abs(np.fft.rfft(xx*np.hanning(n)));freq=np.fft.rfftfreq(n,1/50000*max(1,int((b-a)*50000)//2000)) if n else np.array([])
 # Windows are <=22.3ms, so no downstride is taken; bins follow actual sample count.
 freq=np.fft.rfftfreq(n,1/50000)
 hs=np.array([np.sum(amp[abs(freq-50*h)<=25]**2) for h in range(1,11)])
 prob=power[1:]/np.maximum(power[1:].sum(1,keepdims=True),eps)
 total=power.mean(1);theta=(np.arange(64)+.5)*2*np.pi/64;peak=np.argmax(prob,axis=1);cent=(prob*np.exp(1j*theta)).sum(1)
 zero=(abs(np.angle(np.exp(1j*theta)))<=np.pi/10)|(abs(np.angle(np.exp(1j*(theta-np.pi))))<=np.pi/10)
 features=[np.log10(rms),np.max(abs(xx))/rms,np.mean(((xx-mu)/sd)**3),np.mean(((xx-mu)/sd)**4),np.sqrt(np.mean(np.diff(xx)**2))/rms,np.mean(abs(xx)<.05*rms),abs(np.mean(xx[:n//2]**2)-np.mean(xx[n//2:]**2))/(rms*rms),np.sqrt(hs[1:].sum()/max(hs[0],eps)),np.sqrt(hs[2]/max(hs[0],eps)),np.sqrt(hs[4]/max(hs[0],eps)),*np.log10(np.maximum(total,eps)),total[2]/max(total[1:].sum(),eps),total[0]/max(total.sum(),eps),*np.log10(np.maximum((burst/(np.maximum(power[1:],eps)**2)).mean(1),1)),*(-(prob*np.log(np.maximum(prob,eps))).sum(1)/np.log(64)),*prob.max(1),*prob[:,zero].sum(1),*abs(prob[:,:32].sum(1)-prob[:,32:].sum(1)),np.sin(theta[peak[0]]),np.cos(theta[peak[0]]),np.sin(theta[peak[1]]),np.cos(theta[peak[1]]),*abs(cent)]
 if previous is None:features +=[0.]*9
 else:
  old,oldrms=previous;delta=prob-old;mid=(prob+old)/2
  js=.5*(prob*np.log(np.maximum(prob,eps)/np.maximum(mid,eps))+old*np.log(np.maximum(old,eps)/np.maximum(mid,eps))).sum(1)
  cos=1-(prob*old).sum(1)/np.maximum(np.linalg.norm(prob,axis=1)*np.linalg.norm(old,axis=1),eps)
  drift=abs(np.angle(np.exp(1j*(theta[peak]-theta[np.argmax(old,axis=1)]))))/np.pi
  features +=[*np.mean(abs(delta),axis=1),*js,*cos,*drift,abs(np.log(rms/oldrms))]
 assert len(features)==len(FEATURE_NAMES),(len(features),len(FEATURE_NAMES))
 base=np.vstack([shape,np.log10(np.maximum(power[1:],eps)),np.zeros_like(prob)])
 delta=np.zeros_like(prob) if previous is None else prob-previous[0]
 base[3:]=delta
 # Four two-channel candidate maps; none uses labels.
 maps=np.stack([prob*64,abs(delta)*64,np.log10(np.maximum(burst/np.maximum(power[1:],eps)**2,1)),np.stack([power[1]/np.maximum(power[1:].sum(0),eps),power[2]/np.maximum(power[1:].sum(0),eps)])])
 return np.asarray(features,'float32'),base.astype('float32'),maps.astype('float32'),(prob,rms)

def one(job, *, recording=None):
 row,outdir=job;dest=Path(outdir)/row['id'];receipt=dest/'COMPLETE.json'
 if receipt.exists():return json.loads(receipt.read_text())
 dest.mkdir(parents=True,exist_ok=True);tick=time.time();fs=2000000
 path=Path(row['bin_114'])
 recording = read_recording(path) if recording is None else recording
 h, n, raw, rawsha = recording
 low=upfirdn(firwin(801,20000,fs=fs,window=('kaiser',7.5)),raw,down=40)[:(n+39)//40]
 shape=sosfilt(butter(4,1000,fs=50000,output='sos'),low);shprefix=np.r_[0.,np.cumsum(shape)]/50000; sht=np.arange(len(shprefix))/50000
 reference=sosfilt(butter(2,800,fs=50000,output='sos'),low)[::10]
 clock=estimate(reference,chunk_samples=10)
 ph,fr=robust_pll(reference);tx=np.arange(len(ph))/5000
 # Native full-rate filters are never computed from the 5kHz phase reference.
 prefixes=[]
 for band in [(1000,10000),(10000,100000),(100000,400000)]:
  y=sosfilt(butter(3,band,fs=fs,btype='bandpass',output='sos'),raw);power=y*y
  prefixes.append(np.r_[0.,np.cumsum(power)]/fs)
  if band[0]>=10000:prefixes.append(np.r_[0.,np.cumsum(power*power)]/fs)
 del y,power,raw
 # order MID,HF1,HF1fourth,HF2,HF2fourth
 ce={
 'fixed':candidates(np.array([]),np.array([]),np.array([],bool),np.array([],int),np.array([],int),n,fs),
 'sogi':candidates(tx,ph,np.isfinite(ph),np.zeros(len(ph),int),np.zeros(len(ph),int),n,fs),
 'rcr':candidates(clock.estimate_time_s.to_numpy(),clock.phase_rad.to_numpy(),clock.phase_usable.to_numpy(),clock.reference_segment.to_numpy(),clock.reference_epoch.to_numpy(),n,fs)}
 result={};on=float(row['onset_s']);off=float(row['end_s']);duration=n/fs
 anchors=np.arange(.24,duration-.04+1e-10,.02)
 for ref,entries in ce.items():
  fts=[];bases=[];maps=[];hist=[];prev={};past={};last={}
  for e in entries:
   key=(e['native'],e['seg'],e['epoch'])
   if last.get(key)!=e['a']:prev.pop(key,None);past[key]=[]
   edges=e['edges'];dt=np.diff(edges/fs)
   sh=np.diff(np.interp(edges/fs,sht,shprefix))/dt
   vv=np.array([np.diff(pre[edges])/dt for pre in prefixes]);power=vv[[0,1,3]];fourth=vv[[2,4]]
   feat,base,mp,pv=describe(sh,power,fourth,low,e['a']/fs,e['b']/fs,prev.get(key))
   prev[key]=pv;past.setdefault(key,[]).append(len(fts));last[key]=e['b']
   fts.append(feat);bases.append(base);maps.append(mp);hist.append(past[key][-9:].copy())
  fts=np.asarray(fts);bases=np.asarray(bases);maps=np.asarray(maps)
  sel=[];ys=[];ats=[];histories=[];kinds=[];supports=[]
  for anchor in anchors:
   if anchor<on-.04:left,right,y=.04,on-.04,0
   elif on+.04<=anchor<=off-.04:left,right,y=on+.04,off-.04,1
   elif anchor>=off+.04:left,right,y=off+.04,duration-.04,0
   else:continue
   eligible=[i for i,e in enumerate(entries) if e['av']<=anchor+1e-10 and e['b']/fs<=anchor+1e-10 and anchor-e['b']/fs<=.04+1e-10]
   use_native=False
   if ref=='rcr':
    qi=np.searchsorted(clock.estimate_time_s.to_numpy(),anchor+1e-10,side='right')-1
    if qi>=0 and bool(clock.phase_usable.iloc[qi]):
     native=[i for i in eligible if entries[i]['native'] and entries[i]['seg']==int(clock.reference_segment.iloc[qi]) and entries[i]['epoch']==int(clock.reference_epoch.iloc[qi])]
     if native:eligible=native;use_native=True
   elif ref=='sogi':
    native=[i for i in eligible if entries[i]['native']]
    if native:eligible=native;use_native=True
   if not use_native:eligible=[i for i in eligible if not entries[i]['native']]
   if not eligible:continue
   i=max(eligible,key=lambda i:entries[i]['b']);hh=hist[i]
   if len(hh)!=9 or entries[hh[0]]['a']/fs<left-1e-10 or entries[i]['b']/fs>right+1e-10:continue
   sel.append(i);ys.append(y);ats.append(round(anchor,8));histories.append(hh);kinds.append(int(entries[i]['native']));supports.append([entries[hh[0]]['a']/fs,entries[i]['b']/fs,entries[i]['av']])
  if not sel:raise ValueError('No retained anchors '+row['id']+' '+ref)
  result[ref]={'anchor':np.array(ats),'label':np.array(ys,'int8'),'features':fts[sel], 'base':bases,'maps':maps,'hist':np.array(histories,'int32'),'selected':np.array(sel,'int32'),'native':np.array(kinds,'int8'),'support':np.array(supports)}
 common=sorted(set.intersection(*(set(z['anchor']) for z in result.values())))
 assert len(common)>0
 for ref,z in result.items():
  keep=np.searchsorted(z['anchor'],common)
  for k in ['anchor','label','features','hist','selected','native','support']:z[k]=z[k][keep]
  np.savez_compressed(dest/(ref+'.npz'),**z)
 rec=dict(id=row['id'],source_file=str(path),source_size=path.stat().st_size,header_sha256=hashlib.sha256(h).hexdigest(),CH0_sha256=rawsha,anchors=len(common),normal=int(sum(result['fixed']['label']==0)),arc=int(sum(result['fixed']['label']==1)),duration_s=duration,rcr_native_fraction=float(result['rcr']['native'].mean()),seconds=time.time()-tick)
 receipt.write_text(json.dumps(rec));return rec

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--output',required=True);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rows=list(csv.DictReader(open(a.manifest,encoding='utf-8-sig')))
 (out/'feature_names.json').write_text(json.dumps(FEATURE_NAMES))
 receipts=[]
 with ProcessPoolExecutor(a.workers) as pool:
  jobs=[pool.submit(one,(r,str(out/'records'))) for r in rows]
  for f in as_completed(jobs):
   rec=f.result();receipts.append(rec);(out/'progress.json').write_text(json.dumps({'done':len(receipts),'total':len(rows),'last':rec}))
   print('EXTRACTED',len(receipts),rec['id'],round(rec['seconds'],2),flush=True)
 (out/'PREPARED.json').write_text(json.dumps({'records':receipts,'feature_names':FEATURE_NAMES,'status':'complete'}))
if __name__=='__main__':main()
