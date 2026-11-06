"""Causal RCR candidate with bounded memory and millisecond endpoint updates.

Only past reference-current samples enter a robust harmonic endpoint fit.
Observed, template-supported and predicted phases remain distinct. The output
clock is immutable: reacquisition never rewrites previously emitted phases.
"""
from collections import deque
from dataclasses import dataclass
import time
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RealtimeRCRConfig:
    reference_fs: int = 5000
    update_seconds: float = .001
    memory_cycles: int = 4
    observation_cycles: float = 2.
    startup_seconds: float = .04
    frequency_min: float = 47.
    frequency_max: float = 53.
    harmonic_count: int = 9
    huber_c: float = 1.5
    irls_steps: int = 2
    max_residual_ratio: float = .45
    min_inlier_fraction: float = .65
    min_fundamental_fraction: float = .08
    min_fundamental_snr: float = 3.
    phase_time_constant_seconds: float = .004
    frequency_time_constant_seconds: float = .08
    maximum_innovation_deg: float = 20.
    maximum_step_correction_deg: float = 5.
    template_min_correlation: float = .94
    template_min_margin: float = .06
    holdover_seconds: float = .04
    reset_after_seconds: float = .08
    reacquire_consensus_seconds: float = .005


def wrap(x):
    return np.angle(np.exp(1j*x))


def harmonic_fit(t, y, endpoint, frequency, cfg, refine_frequency=False):
    tau = np.asarray(t)-endpoint
    harmonics = np.arange(1,cfg.harmonic_count+1)
    args = 2*np.pi*frequency*tau[:,None]*harmonics[None,:]
    basis = np.column_stack((np.ones(len(y)),np.sin(args),np.cos(args)))
    recent = np.exp(tau/.03)
    weight = recent.copy()
    eps = 1e-12
    for _ in range(cfg.irls_steps+1):
        root = np.sqrt(weight)
        coef = np.linalg.lstsq(basis*root[:,None],y*root,rcond=None)[0]
        residual = y-basis@coef
        scale = max(.01*float(np.std(y)),1.4826*float(np.median(abs(residual-np.median(residual)))),eps)
        robust = np.minimum(1.,cfg.huber_c*scale/np.maximum(abs(residual),eps))
        weight = recent*robust
    z = coef[1:1+cfg.harmonic_count]+1j*coef[1+cfg.harmonic_count:]
    rr = np.sqrt(np.sum(weight*residual**2)/max(weight.sum(),eps))
    amplitude = max(float(np.linalg.norm(z)/np.sqrt(2)),eps)
    refined = float(frequency)
    if refine_frequency:
        derivative = (2*np.pi*tau[:,None]*harmonics[None,:]*(
            np.cos(args)*z.real[None,:]-np.sin(args)*z.imag[None,:])).sum(1)
        augmented = np.column_stack((basis,derivative))
        norms = np.maximum(np.sqrt((augmented**2*weight[:,None]).sum(0)),1e-10)
        root = np.sqrt(weight)
        correction = np.linalg.lstsq((augmented/norms)*root[:,None],residual*root,rcond=None)[0]/norms
        refined = float(np.clip(frequency+np.clip(correction[-1],-.25,.25),cfg.frequency_min,cfg.frequency_max))
    return dict(z=z,phase=float(np.angle(z[0])),frequency=refined,
                residual_ratio=float(rr/amplitude),inlier_fraction=float((robust>=.5).mean()),
                fundamental_fraction=float(abs(z[0])**2/max(np.sum(abs(z)**2),eps)),
                fundamental_snr=float(abs(z[0])/max(rr,eps)),signal_rms=amplitude)


def good_fundamental(fit,cfg):
    return bool(fit['residual_ratio']<=cfg.max_residual_ratio and
                fit['inlier_fraction']>=cfg.min_inlier_fraction and
                fit['fundamental_fraction']>=cfg.min_fundamental_fraction and
                fit['fundamental_snr']>=cfg.min_fundamental_snr and fit['signal_rms']>1e-8)


class RealtimeRCR:
    def __init__(self,config=None,retain_log=False):
        self.config=config or RealtimeRCRConfig()
        cfg=self.config
        self.sample_index=0;self.values=deque(maxlen=int(np.ceil(cfg.memory_cycles/cfg.frequency_min*cfg.reference_fs)))
        self.times=deque(maxlen=self.values.maxlen)
        self.theta=None;self.frequency=50.;self.last_time=None;self.last_observed=None
        self.last_supported=None;self.segment=-1;self.epoch=0;self.previous_usable=False
        self.templates=deque(maxlen=cfg.memory_cycles);self.last_template_time=-np.inf
        # Online callers consume update() returns. Full diagnostics are opt-in
        # for offline replay, so deployment memory does not grow with uptime.
        self.was_lost=False;self.rows=[] if retain_log else None
        self.last_bootstrap_attempt=-np.inf
        self.pending=deque(maxlen=12)

    def update(self,samples):
        """Consume a finite chunk; results are independent of caller chunk size."""
        cfg=self.config;step=max(1,round(cfg.update_seconds*cfg.reference_fs));new=[]
        for value in np.asarray(samples,dtype=float):
            if not np.isfinite(value):raise ValueError('Finite reference-current samples required')
            current_time=self.sample_index/cfg.reference_fs
            self.values.append(float(value));self.times.append(current_time);self.sample_index+=1
            if self.sample_index%step:continue
            tick=time.perf_counter();row=self._step(current_time)
            row['processing_seconds']=time.perf_counter()-tick
            if self.rows is not None:self.rows.append(row)
            new.append(row)
        return new

    def _step(self,now):
        cfg=self.config;t=np.asarray(self.times);y=np.asarray(self.values)
        row=dict(estimate_time_s=now,latest_input_time_s=now,phase_rad=np.nan,
            frequency_hz=self.frequency,phase_valid=False,observed_valid=False,
            phase_usable=False,prediction_only=False,mode='STARTUP',origin='fundamental',
            reference_segment=-1,reference_epoch=self.epoch,observation_source='NONE',
            age_since_support_s=np.inf,innovation_deg=np.nan,fundamental_fraction=np.nan,
            residual_ratio=np.nan,inlier_fraction=np.nan,memory_samples=len(y))
        if self.theta is None:
            if len(y)<round(cfg.startup_seconds*cfg.reference_fs):return row
            if np.std(y)<1e-8 or now-self.last_bootstrap_attempt<.005-1e-12:return row
            self.last_bootstrap_attempt=now
            # One bounded startup frequency fit, using only the initial past window.
            candidates=[harmonic_fit(t,y,now,f,cfg) for f in np.arange(47.,53.01,.5)]
            losses=[q['residual_ratio'] for q in candidates];fit=candidates[int(np.argmin(losses))]
            if not good_fundamental(fit,cfg):return row
            frequency=np.arange(47.,53.01,.5)[int(np.argmin(losses))]
            fit=harmonic_fit(t,y,now,frequency,cfg,True)
            self.frequency=fit['frequency'];fit=harmonic_fit(t,y,now,self.frequency,cfg)
            if not good_fundamental(fit,cfg):return row
            self.theta=fit['phase']+2*np.pi*round((2*np.pi*50*now-fit['phase'])/(2*np.pi))
            self.last_time=now;self.last_observed=now;self.last_supported=now
            self.segment+=1;self.previous_usable=True
            self.templates.append(fit['z']*np.exp(-1j*np.arange(1,cfg.harmonic_count+1)*fit['phase']))
            self.last_template_time=now
            row.update(phase_rad=self.theta,frequency_hz=self.frequency,phase_valid=True,
                observed_valid=True,phase_usable=True,mode='ACQUIRED',reference_segment=self.segment,
                observation_source='FUNDAMENTAL',age_since_support_s=0.,
                fundamental_fraction=fit['fundamental_fraction'],residual_ratio=fit['residual_ratio'],
                inlier_fraction=fit['inlier_fraction'])
            return row
        dt=now-self.last_time;prediction=self.theta+2*np.pi*self.frequency*dt
        window=max(.036,cfg.observation_cycles/self.frequency)
        take=t>=now-window-1e-12;tx,yy=t[take],y[take]
        fit=harmonic_fit(tx,yy,now,self.frequency,cfg,True)
        observed=good_fundamental(fit,cfg);source='NONE';delta=0.;supported=False
        if observed:
            # Refit phase at the observed frequency; phase is timestamped at NOW.
            target_frequency=fit['frequency']
            measured=harmonic_fit(tx,yy,now,target_frequency,cfg)
            delta=float(wrap(measured['phase']-prediction));fit=measured;source='FUNDAMENTAL';supported=True
        elif self.templates and fit['residual_ratio']<=cfg.max_residual_ratio and fit['inlier_fraction']>=cfg.min_inlier_fraction:
            template=np.median(np.stack([q.real for q in self.templates]),axis=0)+1j*np.median(np.stack([q.imag for q in self.templates]),axis=0)
            h=np.arange(2,cfg.harmonic_count+1);a=fit['z'][1:];b=template[1:]
            product=a*np.conj(b);denom=max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-12)
            grid=prediction+np.deg2rad(np.linspace(-25,25,101))
            correlation=np.real(np.exp(-1j*grid[:,None]*h[None,:])@product)/denom
            best=int(np.argmax(correlation));far=abs(grid-grid[best])>=np.deg2rad(10)
            margin=float(correlation[best]-np.max(correlation[far])) if far.any() else 0.
            if correlation[best]>=cfg.template_min_correlation and margin>=cfg.template_min_margin:
                delta=float(wrap(grid[best]-prediction));source='TEMPLATE_MEMORY';supported=True
        age=now-self.last_supported;epoch_reset=False
        if supported:
            # Abrupt incompatible phase after a real unavailable period starts a new epoch.
            if age>cfg.reset_after_seconds and observed:
                self.epoch+=1;prediction=fit['phase']+2*np.pi*round((prediction-fit['phase'])/(2*np.pi));delta=0.;epoch_reset=True
            if abs(np.rad2deg(delta))>cfg.maximum_innovation_deg and age<=cfg.reset_after_seconds:
                if observed and fit['residual_ratio']<.2 and fit['fundamental_fraction']>.2:
                    self.pending.append((now,fit['phase']))
                    first_time,first_phase=self.pending[0]
                    consistent=abs(np.rad2deg(wrap(fit['phase']-first_phase-2*np.pi*target_frequency*(now-first_time))))<5.
                    if now-first_time>=cfg.reacquire_consensus_seconds-1e-12 and consistent:
                        self.epoch+=1;prediction=fit['phase']+2*np.pi*round((prediction-fit['phase'])/(2*np.pi))
                        delta=0.;epoch_reset=True;source='REACQUIRED_FUNDAMENTAL';self.pending.clear()
                    else:supported=False;observed=False;source='INNOVATION_REJECTED'
                else:
                    self.pending.clear();supported=False;observed=False;source='INNOVATION_REJECTED'
            else:self.pending.clear()
        if epoch_reset:
            self.templates.clear();self.last_template_time=-np.inf
        if supported:
            quality=np.clip(1-fit['residual_ratio']/cfg.max_residual_ratio,.1,1.)
            alpha=(1-np.exp(-dt/cfg.phase_time_constant_seconds))*quality
            correction=np.clip(alpha*delta,-np.deg2rad(cfg.maximum_step_correction_deg),np.deg2rad(cfg.maximum_step_correction_deg))
            self.theta=prediction+correction;self.last_supported=now
            if observed:
                gain=1-np.exp(-dt/cfg.frequency_time_constant_seconds)
                self.frequency=float(np.clip(self.frequency+gain*(target_frequency-self.frequency),cfg.frequency_min,cfg.frequency_max))
                self.last_observed=now
                if now-self.last_template_time>=1/self.frequency and fit['residual_ratio']<.2:
                    self.templates.append(fit['z']*np.exp(-1j*np.arange(1,cfg.harmonic_count+1)*fit['phase']))
                    self.last_template_time=now
        else:self.theta=prediction
        usable=supported or now-self.last_supported<=cfg.holdover_seconds+1e-12
        if usable and (not self.previous_usable or epoch_reset):self.segment+=1
        self.previous_usable=usable;self.last_time=now
        row.update(phase_rad=self.theta,frequency_hz=self.frequency,phase_valid=bool(usable),
            observed_valid=bool(observed and supported),phase_usable=bool(usable),prediction_only=not supported,
            mode=source if supported else ('BOUNDED_HOLDOVER' if usable else 'UNAVAILABLE'),
            reference_segment=self.segment if usable else -1,reference_epoch=self.epoch,
            observation_source=source,age_since_support_s=now-self.last_supported,
            innovation_deg=float(np.rad2deg(delta)) if supported else np.nan,
            fundamental_fraction=fit['fundamental_fraction'],residual_ratio=fit['residual_ratio'],
            inlier_fraction=fit['inlier_fraction'])
        return row


def estimate(reference,config=None,chunk_samples=37):
    tracker=RealtimeRCR(config,retain_log=True)
    for start in range(0,len(reference),chunk_samples):tracker.update(reference[start:start+chunk_samples])
    return pd.DataFrame(tracker.rows)
