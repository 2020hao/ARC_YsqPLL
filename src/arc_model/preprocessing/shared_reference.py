"""Experimental one-clock SOGI/robust-harmonic fusion.

No independent PLL trajectories are averaged. Both discriminators observe one
shared theta/frequency state. This is a research candidate, not certified phase.
"""
from __future__ import annotations
import importlib.util,json
from pathlib import Path
import numpy as np
import pandas as pd

VARIANTS=('sogi','naive','hybrid','no_quality','no_template','no_irls')

def _observer():
    root=Path(__file__).resolve().parents[3]/'third_party/current_reference_recovery_v2'
    sp=importlib.util.spec_from_file_location('_arc_shared_observation_math',root/'dual_reference_pll.py')
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    return m,json.loads((root/'pll_spec.json').read_text())

def shared_reference(x,fs=5000,variant='hybrid'):
    """Return dense phase/rate and a 20-ms diagnostic clock on one input path.

    x must already have the registered causal low-reference preprocessing. The
    function does not filter high-frequency detector inputs or accept labels.
    """
    if variant not in VARIANTS:raise ValueError('Unknown fusion variant')
    x=np.asarray(x,float)
    if x.ndim!=1 or len(x)<int(.06*fs) or not np.isfinite(x).all():raise ValueError('Finite reference of at least60ms required')
    observer,spec=_observer()
    if variant=='no_irls':spec['irls_iterations']=0
    spec['reference_fs_hz']=fs
    init=int(.04*fs);t=np.arange(len(x))/fs;design=np.column_stack([np.ones(init)]+[fn(2*np.pi*50*h*t[:init]) for h in [1,3,5] for fn in [np.sin,np.cos]])
    coef=np.linalg.lstsq(design,x[:init],rcond=None)[0];dc=float(coef[0]);amp=max(float(np.hypot(coef[1],coef[2])),1e-8);theta=2*np.pi*50*init/fs+np.arctan2(coef[2],coef[1]);a,b=amp*np.sin(theta),-amp*np.cos(theta);prev=float(x[init-1]-dc);integ=0.;dt=1/fs;wn=2*np.pi*3;kp=2*.707*wn;ki=wn**2
    phase=np.full(len(x),np.nan);rate=np.full(len(x),np.nan);phase[init]=theta;rate[init]=50.;dense_valid=np.zeros(len(x),bool);dense_epoch=np.zeros(len(x),int)
    err2=0.;power=amp**2/2;clipavg=0.;smooth=1-np.exp(-dt/.02);step=round(.02*fs);obs_phase=np.nan;obs_time=0.;obs_freq=50.;mix=0.;template=None;template_count=0;epoch=0;last_valid=t[init];gap_armed=False;rows=[];last_obs_valid=False
    for n in range(init+1,len(x)):
        u=float(x[n]-dc);bound=4*max(np.hypot(a,b),.1*amp);clipped=abs(u)>bound;u=np.clip(u,-bound,bound);omega=2*np.pi*50+integ;w=.5*dt*omega;r1=(1-np.sqrt(2)*w)*a-w*b+np.sqrt(2)*w*(u+prev);r2=w*a+b;a=(r1-w*r2)/(1+np.sqrt(2)*w+w*w);b=r2+w*a
        pred=theta+dt*2*np.pi*rate[n-1];es=np.arctan2(a*np.cos(pred)+b*np.sin(pred),a*np.sin(pred)-b*np.cos(pred));es=float(np.clip(es,-.35,.35));err2=(1-smooth)*err2+smooth*es**2;power=(1-smooth)*power+smooth*(float(x[n])-dc)**2;clipavg=(1-smooth)*clipavg+smooth*clipped
        qs=float(np.exp(-err2/np.deg2rad(10)**2)*max(0.,1-clipavg));signal_good=power>(.05*amp)**2 and power>1e-16;source='NONE';qr=0.;er=np.nan;resid=np.nan;repeat=np.nan;fundamental_fraction=np.nan;tc=np.nan
        if n%step==step-1:
            if n+1>=round(.06*fs) and variant!='sogi':
                frequency=float(np.clip(50+integ/(2*np.pi),45,55));nwin=max(round(.06*fs),int(np.ceil(3*fs/frequency))+1);tx=t[max(0,n+1-nwin):n+1];y=x[max(0,n+1-nwin):n+1];fit=observer.fit_harmonics(tx,y,pred,frequency,t[n],spec);repeat=observer.periodicity(tx,y,frequency,t[n],spec);resid=fit['residual_ratio'];fundamental_fraction=fit['fundamental_fraction'];basic=observer.basic_good(fit,repeat,spec);fg=basic and observer.fundamental_good(fit,spec)
                if fg:
                    source='FUNDAMENTAL';er=float(np.angle(fit['z'][0]));template_count+=1
                    if template is None and t[n]>=.2 and template_count>=3:
                        h=np.asarray(spec['harmonics']);template=fit['z']*np.exp(-1j*h*er)
                else:template_count=0
                if source=='NONE' and template is not None and variant!='no_template':
                    delta,tc,margin=observer.template_shift(fit['z'],template,spec,False)
                    if basic and tc>=.80 and margin>=.06:source='TEMPLATE';er=delta
                good=source!='NONE'
                if good:qr=float(np.clip(1-resid/.35,0,1)*np.clip(repeat,0,1))
                if variant=='naive':
                    er=float(np.angle(fit['z'][0]));source='FUNDAMENTAL_FIXED_MIX';good=np.isfinite(er) and fit['signal_rms']>1e-12;mix=.5 if good else 0.
                elif variant=='sogi':mix=0.
                elif variant=='no_quality':
                    er=float(np.angle(fit['z'][0]));source='FUNDAMENTAL_UNGATED';good=fit['signal_rms']>1e-12;mix=.5*(1-.75*qs) if good else 0.
                else:mix=.5*qr*(1-.75*qs) if good else 0.
                last_obs_valid=bool(good)
                if good:
                    obs_phase=pred+float(np.clip(er,-np.deg2rad(35),np.deg2rad(35)));obs_time=t[n];obs_freq=frequency
                    if variant!='sogi' and basic:dc+=.1*qr*(fit['dc']-dc)
            primary_valid=signal_good and qs>=.5
            valid=bool(primary_valid or (last_obs_valid and qr>=.2))
            if variant=='sogi':valid=bool(primary_valid)
            event=''
            if not valid and t[n]-last_valid>=.2:gap_armed=True
            if valid:
                if gap_armed:epoch+=1;gap_armed=False;event='reference_epoch_after_gap'
                last_valid=t[n]
            rows.append(dict(estimate_time_s=float(t[n]),phase_rad=float(pred),frequency_hz=float(rate[n-1]),phase_valid=valid,reference_epoch=epoch,primary_quality=qs,observer_quality=qr,mixing_weight=mix,observation_source=source,robust_residual_ratio=resid,repeat_correlation=repeat,fundamental_fraction=fundamental_fraction,template_correlation=tc,observer_innovation_deg=float(np.rad2deg(er)),dc_estimate=dc,event=event))
        if np.isfinite(obs_phase) and t[n]-obs_time<=.020001:
            er_live=float(np.angle(np.exp(1j*(obs_phase+2*np.pi*obs_freq*(t[n]-obs_time)-pred))));error=(1-mix)*es+mix*np.clip(er_live,-.35,.35)
        else:error=es
        integ=float(np.clip(integ+ki*dt*error,-2*np.pi*5,2*np.pi*5));rr=float(np.clip(2*np.pi*50+integ+kp*error,2*np.pi*40,2*np.pi*60));theta+=rr*dt;phase[n]=theta;rate[n]=rr/(2*np.pi);prev=u
        dense_valid[n]=signal_good and (qs>=.5 or(variant!='sogi' and last_obs_valid and qr>=.2));dense_epoch[n]=epoch
        if n%step==step-1:rows[-1].update(phase_rad=float(theta),frequency_hz=float(rr/(2*np.pi)))
    clock=pd.DataFrame(rows)
    if len(clock):
        # Lifecycle IDs reflect uninterrupted quality-valid output, never labels.
        val=clock.phase_valid.to_numpy();starts=val&~np.r_[False,val[:-1]];sg=np.cumsum(starts)-1;clock['reference_segment']=np.where(val,sg,-1);clock['origin']='fundamental'
    return phase,rate,clock,dict(valid=dense_valid,epoch=dense_epoch)
