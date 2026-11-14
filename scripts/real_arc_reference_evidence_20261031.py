"""Real-current observational reference/feature/error evidence; no new fits.

The offline two-scale current-phase proxy is never a physical phase truth, online
input, label source or gate. All rules freeze before reading diagnostic outcomes.
"""
from concurrent.futures import ProcessPoolExecutor
import json,os,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.signal import fftconvolve,butter,sosfilt
import _bootstrap as boot
from validate_locked_dual_20261030 import DATASETS,MODELS,dataset,sha,save
from arc_model.preprocessing.signals import condition
from arc_model.preprocessing.rcr_reference import _implementation
from arc_model.preprocessing.rcr_frequency_tracking_v1 import estimate as selective_estimate

ROOT=boot.ROOT;OUT=ROOT/'outputs/real_arc_reference_evidence_20261031';CONFIG=ROOT/'configs/modules/real_arc_reference_evidence_20261031.yaml'
CP=ROOT/'outputs/continuous_pll_validation_20261027_v2';PRIOR=ROOT/'outputs/coordinate_evidence_20261031'
METHODS=('fixed50','SOGI','RCR','RCR_selective');CACHE={}


def independent_proxy(x,fs=5000):
    """Positive-sine complex demodulation, centered offline two-scale agreement."""
    x=np.asarray(x,float);t=np.arange(len(x))/fs;z=[];power=[]
    demod=2j*x*np.exp(-2j*np.pi*50*t)
    for n in [401,601]:
        w=np.hanning(n);w/=w.sum();z.append(fftconvolve(demod,w,mode='same'));power.append(np.maximum(fftconvolve(x*x,w,mode='same'),0.))
    phase=2*np.pi*50*t+np.unwrap(np.angle(z[1]));disagreement=abs(np.rad2deg(np.angle(z[0]*np.conj(z[1]))))
    fraction=np.minimum(abs(z[0])**2/(2*power[0]+1e-24),abs(z[1])**2/(2*power[1]+1e-24))
    rms=np.sqrt(np.maximum(power[1],0.));floor=max(float(np.sqrt(np.mean(x*x)))*1e-6,1e-12)
    valid=(disagreement<=5)&(fraction>=.1)&(rms>floor)&(t>=.06)&(t<=(len(x)-1)/fs-.06)
    return dict(time=t,phase=phase,valid=valid,scale_disagreement_deg=disagreement,fundamental_power_fraction=fraction)


def dense_at(t,phase,frequency,valid,query):
    j=np.floor(np.asarray(query)*5000+1e-8).astype(int);inside=(j>=0)&(j<len(t));j=np.clip(j,0,len(t)-1)
    theta=phase[j]+2*np.pi*frequency[j]*(np.asarray(query)-t[j]);ok=inside&valid[j]&np.isfinite(theta)
    return theta,ok,np.zeros(len(j),int),np.zeros(len(j),int)


def clock_at(clock,query):
    tt=clock.estimate_time_s.to_numpy();j=np.searchsorted(tt,np.asarray(query)+1e-12,side='right')-1;inside=j>=0;j=np.maximum(j,0)
    theta=clock.phase_rad.to_numpy()[j]+2*np.pi*clock.frequency_hz.to_numpy()[j]*(np.asarray(query)-tt[j])
    fund=clock.origin.to_numpy()[j]=='fundamental';valid=inside&clock.phase_valid.to_numpy()[j]&fund&np.isfinite(theta)
    return theta,valid,clock.reference_epoch.to_numpy()[j],clock.reference_segment.to_numpy()[j]


def probabilities(energy,phase,valid,epoch,segment,start):
    n=len(start);out=np.zeros((n,2,20));whole=np.zeros(n,bool);coverage=np.zeros((n,2));ids=np.full(n,-1,int)
    for k,a in enumerate(start):
        b=a+100
        if a<0 or b>len(energy):continue
        ok=valid[a:b]&np.isfinite(phase[a:b]);e=energy[a:b]
        total=e.sum(0);coverage[k]=np.divide(e[ok].sum(0),total,out=np.zeros(2),where=total>1e-24)
        if not ok.any():continue
        bins=np.floor(np.mod(phase[a:b][ok],2*np.pi)*20/(2*np.pi)).astype(int)%20
        for band in range(2):out[k,band]=np.bincount(bins,weights=e[ok,band],minlength=20)
        out[k]/=np.maximum(out[k].sum(-1,keepdims=True),1e-24)
        whole[k]=bool(ok.all() and len(np.unique(epoch[a:b]))==1 and len(np.unique(segment[a:b]))==1)
        if whole[k]:ids[k]=int(epoch[a])*100000+int(segment[a])
    return out,whole,coverage,ids


def dynamics(p,valid,previous,epochs):
    mad=np.full(len(p),np.nan);js=mad.copy()
    for i,j in enumerate(previous):
        if j<0 or not (valid[i] and valid[j]) or epochs[i]!=epochs[j]:continue
        u,v=p[i],p[j];mid=.5*(u+v)
        mad[i]=abs(u-v).mean();js[i]=(.5*np.sum(u*np.log((u+1e-12)/(mid+1e-12)),axis=1)+.5*np.sum(v*np.log((v+1e-12)/(mid+1e-12)),axis=1)).mean()
    return mad,js


def prepare():
    from check_baseline_release import verify
    verify()
    if (OUT/'protocol.json').exists():
        spec=json.loads((OUT/'protocol.json').read_text())
        for q in spec['source_identities']:assert sha(q['path'])==q['sha256'],q['path']
        return spec
    cfg=yaml.safe_load(CONFIG.read_text());OUT.mkdir(parents=True,exist_ok=False);sources={CONFIG,Path(__file__),ROOT/'scripts/run_experiment_suite.py',ROOT/'configs/datasets.yaml',
      ROOT/'src/arc_model/preprocessing/rcr_frequency_tracking_v1.py',ROOT/'src/arc_model/preprocessing/signals.py',ROOT/'src/arc_model/preprocessing/pll.py',
      ROOT/'src/arc_model/preprocessing/rcr_reference.py',ROOT/'src/arc_model/preprocessing/constants.py',ROOT/'scripts/validate_locked_dual_20261030.py'}
    sources.update((ROOT/'third_party/current_reference_recovery_v2').glob('*.py'));sources.add(ROOT/'third_party/current_reference_recovery_v2/pll_spec.json')
    jobs=[];counts=[]
    for ds in DATASETS:
        d=dataset(ds);keep=d['keep'];t=d['table'];counts.append(dict(dataset=ds,N=len(keep),groups=t.iloc[keep].condition_key.nunique()))
        sources.update([d['base']/'cycles.csv',d['rcr']/'cycles.csv',ROOT/d['config']['reviewed_indices'],ROOT/d['config']['split_file'],ROOT/'outputs/paper_baseline_completion_20261029/data'/ds/'record_receipts.json'])
        for rid in sorted(t.record_id.unique()):
            jobs.append(dict(dataset=ds,record_id=int(rid)));p=CP/'records'/ds/f'{rid:04}'
            sources.update([p/'COMPLETE.json',p/'reference5k.npy',p/'recovery_clock.csv'])
        for model in MODELS:
            for fold in range(5):
                p=PRIOR/'runs'/ds/model/f'f{fold}'
                sources.add(p/'COMPLETE.json')
                sources.update([p/(arm+'.npz') for arm in ['parent','fixed36','sogi36','rcr36']])
    spec=dict(status='frozen_before_real_diagnostic_outcomes',config=cfg,jobs=jobs,counts=counts,new_fits=0,
          source_identities=[dict(path=str(p),sha256=sha(p)) for p in sorted(sources)])
    save(OUT/'protocol.json',spec)
    for p in sources:
        if p.suffix in ['.py','.yaml']:
            q=OUT/'source_snapshot'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
    p=Path('/tmp/real_reference_previous_runner.py')
    if p.exists():(OUT/'source_snapshot/previous_run_experiment_suite.py').write_bytes(p.read_bytes())
    print('FROZEN 151real_records 2datasets 0fits',flush=True);return spec


def one(job):
    ds,rid=job['dataset'],job['record_id'];tick=time.perf_counter();src=CP/'records'/ds/f'{rid:04}';dest=OUT/'records'/ds/f'{rid:04}';dest.mkdir(parents=True,exist_ok=False)
    doc=json.loads((src/'COMPLETE.json').read_text());fs=int(doc['Fs']);rawpath=src/'raw_current.npy';digest=sha(rawpath)
    prior={int(q['record_id']):q for q in json.loads((ROOT/'outputs/paper_baseline_completion_20261029/data'/ds/'record_receipts.json').read_text())}[rid]
    assert digest==prior['raw_channel_cache_sha256'];st=Path(doc['source_file']).stat();assert st.st_size==prior['source_size'] and st.st_mtime_ns==prior['source_mtime_ns']
    raw=np.load(rawpath,mmap_mode='r');low,_,sphase50,sfreq=condition(raw,bands=(),sample_rate_hz=fs)
    ref=sosfilt(butter(2,800,fs=50000,output='sos'),low)[::10]
    np.testing.assert_array_equal(ref,np.load(src/'reference5k.npy'))
    t=np.arange(len(ref))/5000.;sphase=sphase50[::10];svalid=np.isfinite(sphase)
    proxy=independent_proxy(ref);old_estimate,specpath=_implementation();spec=json.loads(specpath.read_text());spec['reference_fs_hz']=5000
    rcr,events=old_estimate(t,ref,spec);legacy=pd.read_csv(src/'recovery_clock.csv',float_precision='round_trip')
    for col in ['phase_rad','frequency_hz','phase_valid','reference_epoch','reference_segment']:np.testing.assert_equal(rcr[col].to_numpy(),legacy[col].to_numpy())
    selective,sevents=selective_estimate(t,ref,spec,'selective');selective.to_csv(dest/'selective_clock.csv',index=False)
    t0=0.
    if ds=='zenodo5501527':
        with Path(doc['source_file']).open() as f:t0=float(f.readline().split()[0])
    d=dataset(ds);tab=d['table'];frame=tab[tab.record_id.eq(rid)].copy();globalids=frame.index.to_numpy();anchors=frame.anchor_seconds.to_numpy()-t0
    keep=np.isin(globalids,d['keep']);n=len(frame)
    # Fixed reference is a single0.2s50Hz fit, used only as a diagnostic comparator.
    z=ref[:1000];tx=t[:1000];B=np.column_stack([np.ones(len(z)),np.sin(2*np.pi*50*tx),np.cos(2*np.pi*50*tx)])
    coef=np.linalg.lstsq(B,z,rcond=None)[0];fixedphase=2*np.pi*50*t+np.arctan2(coef[2],coef[1]);fixedfreq=np.full(len(t),50.)
    curve={};anchor={}
    step=fs//5000;nc=len(raw)//step;celltime=(np.arange(nc)+.5)/5000.;energy=[]
    for band in [(10000,100000),(100000,400000)]:
        hf=sosfilt(butter(3,band,fs=fs,btype='bandpass',output='sos'),raw);prefix=np.r_[0.,np.cumsum(hf*hf)]/fs
        edges=np.arange(nc+1)*step;energy.append(np.diff(prefix[edges]))
    energy=np.maximum(np.stack(energy,axis=1),0.)
    for method in METHODS:
        if method=='fixed50':
            curve[method]=dense_at(t,fixedphase,fixedfreq,t>=.2,celltime);anchor[method]=dense_at(t,fixedphase,fixedfreq,t>=.2,anchors)
        elif method=='SOGI':curve[method]=dense_at(t,sphase,sfreq,svalid,celltime);anchor[method]=dense_at(t,sphase,sfreq,svalid,anchors)
        else:
            c=rcr if method=='RCR' else selective;curve[method]=clock_at(c,celltime);anchor[method]=clock_at(c,anchors)
    proxycellphase=np.interp(celltime,t,proxy['phase']);proxycellvalid=proxy['valid'][np.clip(np.floor(celltime*5000).astype(int),0,len(t)-1)]
    proxyanchor=np.interp(anchors,t,proxy['phase']);aq=np.clip(np.rint(anchors*5000).astype(int),0,len(t)-1);proxyvalid=proxy['valid'][aq]
    start=np.rint((anchors-.02)*5000).astype(int)
    pp,pok,pcov,pepoch=probabilities(energy,proxycellphase,proxycellvalid,np.zeros(nc,int),np.zeros(nc,int),start)
    previous=np.full(n,-1,int);previous8=np.full(n,-1,int)
    for i in range(n):
        if i>0 and abs(anchors[i]-anchors[i-1]-.02)<1e-7:previous[i]=i-1
        if i>=8 and abs(anchors[i]-anchors[i-8]-.16)<1e-7:previous8[i]=i-8
    pmad,pjs=dynamics(pp,pok,previous,pepoch);nmad,njs=dynamics(np.roll(pp,1,axis=-1),pok,previous,pepoch)
    invariant=float(max(np.nanmax(abs(pmad-nmad),initial=0),np.nanmax(abs(pjs-njs),initial=0)))
    assert invariant<1e-12
    # Raw-waveform disturbances are defined without phase/classifier outputs.
    windows=np.zeros((n,100));validwindow=(start>=0)&(start+100<=len(ref))
    for i,a in enumerate(start):
        if validwindow[i]:windows[i]=ref[a:a+100]
    rms=np.sqrt(np.mean(windows*windows,axis=1));normalized=windows/np.maximum(rms[:,None],1e-12)
    rmsjump=np.full(n,np.nan);shapejump=rmsjump.copy()
    for i,j in enumerate(previous):
        if j>=0:
            rmsjump[i]=abs(rms[i]-rms[j])/(rms[i]+rms[j]+1e-12);shapejump[i]=np.sqrt(np.mean((normalized[i]-normalized[j])**2))
    labels=frame.label.to_numpy();normal_disturb=np.zeros(n,bool)
    for i,j in enumerate(previous):
        if j>=0 and keep[i] and keep[j] and labels[i]==0 and labels[j]==0 and frame.interval_start.iloc[i]==frame.interval_start.iloc[j]:
            normal_disturb[i]=rmsjump[i]>=.2 or shapejump[i]>=.5
    #60ms proxy halfsupport plus40ms label uncertainty, strictly within state.
    left=frame.interval_start.to_numpy()-t0;right=frame.interval_end.to_numpy()-t0
    leftguard=np.where((ds=='continuous5m')&(abs(left)<1e-9),0.,left+.04)
    rightguard=np.where((ds=='continuous5m')&(abs(right-len(raw)/fs)<1e-6),right,right-.04)
    pure=(anchors-.06>=leftguard-1e-9)&(anchors+.06<=rightguard+1e-9)
    data=dict(index=globalids,sample_id=frame.sample_id.to_numpy(),record_id=np.full(n,rid),condition_key=frame.condition_key.to_numpy(),label=labels,keep=keep,
      anchor_local=anchors,anchor_seconds=frame.anchor_seconds.to_numpy(),post_startup=anchors>=.8,proxy_eligible=proxyvalid,proxy_label_pure=pure,
      proxy_scale_difference_deg=proxy['scale_disagreement_deg'][aq],proxy_fundamental_fraction=proxy['fundamental_power_fraction'][aq],
      RMS=rms,RMS_jump=rmsjump,shape_jump=shapejump,normal_disturbance_candidate=normal_disturb,proxy_MAD=pmad,proxy_JS=pjs)
    feature_arrays={'proxy_distribution':pp,'proxy_feature_valid':pok,'proxy_MAD':pmad,'proxy_JS':pjs,'proxy_energy_coverage':pcov}
    for method in METHODS:
        phase,valid,epoch,segment=anchor[method];error=np.rad2deg(np.angle(np.exp(1j*(phase-proxyanchor))));error[~(valid&proxyvalid)]=np.nan
        delta1=np.full(n,np.nan);delta8=delta1.copy()
        for i,j in enumerate(previous):
            if j>=0 and valid[i] and valid[j] and proxyvalid[i] and proxyvalid[j] and epoch[i]==epoch[j] and segment[i]==segment[j]:delta1[i]=abs(np.rad2deg(np.angle(np.exp(1j*np.deg2rad(error[i]-error[j])))))
        for i,j in enumerate(previous8):
            if j>=0 and valid[i] and valid[j] and proxyvalid[i] and proxyvalid[j] and epoch[i]==epoch[j] and segment[i]==segment[j]:delta8[i]=abs(np.rad2deg(np.angle(np.exp(1j*np.deg2rad(error[i]-error[j])))))
        bad=np.zeros(n,bool);severe=bad.copy()
        for i,j in enumerate(previous):
            if j>=0 and np.isfinite(error[i]+error[j]):bad[i]=abs(error[i])>10 and abs(error[j])>10;severe[i]=abs(error[i])>20 and abs(error[j])>20
        unstable=(delta1>5)|(delta8>10)
        p,ok,coverage,eid=probabilities(energy,*curve[method],start)
        mad,js=dynamics(p,ok,previous,eid)
        for name,value in [('fundamental_valid',valid),('phase_deviation_deg',error),('relative_lag1_deg',delta1),('relative_lag8_deg',delta8),
              ('absolute_bad10',bad),('absolute_bad20',severe),('relative_unstable',unstable),('relative_defined',np.isfinite(delta1)),
              ('feature_valid',ok),('mapped_energy_fraction',coverage.mean(1)),('MAD',mad),('JS',js),('MAD_gap',abs(mad-pmad)),('JS_gap',abs(js-pjs))]:data[method+'_'+name]=value
        feature_arrays[method+'_distribution']=p;feature_arrays[method+'_feature_valid']=ok;feature_arrays[method+'_energy_coverage']=coverage
    diag=pd.DataFrame(data);diag.to_csv(dest/'window_diagnostics.csv',index=False);np.savez_compressed(dest/'features.npz',**feature_arrays)
    np.savez_compressed(dest/'reference_curves.npz',time=t,current=ref,proxy_phase=proxy['phase'],proxy_valid=proxy['valid'],SOGI_phase=sphase,
         SOGI_frequency=sfreq,RCR_times=rcr.estimate_time_s.to_numpy(),RCR_phase=rcr.phase_rad.to_numpy(),RCR_frequency=rcr.frequency_hz.to_numpy(),
         RCR_valid=rcr.phase_valid.to_numpy(),RCR_origin=rcr.origin.to_numpy(dtype=str),cell_energy=energy)
    save(dest/'COMPLETE.json',dict(status='complete',job=job,windows=n,raw_array_sha256=digest,historical_raw_hash_verified=True,original_source_stat_verified=True,
        fresh_native_HF_integration=True,original_RCR_clock_exact=True,reference_current_exact=True,common_rotation_dynamic_error=invariant,
        t0=t0,source_file=doc['source_file'],native_fs=fs,seconds=time.perf_counter()-tick,files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
    print('REAL_DONE',ds,rid,round(time.perf_counter()-tick,2),flush=True)


def run_study(suite,execute=False):
    spec=prepare()
    if not execute:print('Ready:151native_real_records,0fits');return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only diagnostics require CUDA hidden')
    save(OUT/'EXECUTION_STARTED.json',dict(pid=os.getpid(),time=time.time(),hard_timeout_s=3600))
    try:
        with ProcessPoolExecutor(3) as pool:
            for _ in pool.map(one,spec['jobs']):pass
        for q in spec['source_identities']:assert sha(q['path'])==q['sha256'],q['path']
        save(OUT/'PREPARED_COMPLETE.json',dict(status='complete',records=151,new_fits=0,GPU_used=False))
    except Exception:
        save(OUT/'FAILED.json',dict(traceback=traceback.format_exc()));raise
