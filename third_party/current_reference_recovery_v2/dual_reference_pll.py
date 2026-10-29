"""Current-only dual-discriminator PLL research prototype. Not two PLL phases spliced.
Robust DC+harmonic regression -> fundamental phase, or signed frozen morphology phase.
Bounded shared alpha-beta clock; uncertainty/alias ambiguity can produce no valid phase.
No C2, raw HF, arc labels or known synthetic phase are accepted by the estimator.
"""
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def wrap(x):
    return np.angle(np.exp(1j*x))


def fit_harmonics(t, y, theta, frequency, end, spec):
    h=np.asarray(spec['harmonics'])
    angle=theta+2*np.pi*frequency*(t-end)
    args=angle[:,None]*h[None,:]
    B=np.column_stack((np.ones(len(y)),np.sin(args),np.cos(args)))
    coef=np.linalg.lstsq(B,y,rcond=None)[0]
    eps=spec['numeric_floor']
    scale_floor=max(eps,spec['regression_scale_floor_fraction']*float(np.std(y)))
    weight=np.ones(len(y),dtype=float)  # OLS limit for the explicit no-IRLS ablation.
    for _ in range(spec['irls_iterations']):
        residual=y-B@coef
        scale=max(scale_floor,1.4826*float(np.median(np.abs(residual-np.median(residual)))))
        weight=np.minimum(1.,spec['huber_c']*scale/np.maximum(np.abs(residual),eps))
        coef=np.linalg.lstsq(B*np.sqrt(weight[:,None]),y*np.sqrt(weight),rcond=None)[0]
    residual=y-B@coef
    z=coef[1:1+len(h)]+1j*coef[1+len(h):]
    harmonic_rms=float(np.linalg.norm(z)/np.sqrt(2))
    rr=float(np.sqrt(np.sum(weight*residual**2)/np.sum(weight)))
    return dict(z=z,dc=float(coef[0]),residual_rms=rr,signal_rms=harmonic_rms,
                residual_ratio=rr/max(harmonic_rms,eps),
                inlier_fraction=float(np.mean(weight>=.5)),
                fundamental_fraction=float(abs(z[0])**2/max(np.sum(abs(z)**2),eps**2)),
                fundamental_snr=float(abs(z[0])/max(rr,eps)))


def periodicity(t,y,frequency,end,spec):
    # Signed cycles, per-cycle DC/amplitude removed; interpolation ONLY in reference branch.
    period=1/frequency
    if t[0]>end-3*period+1/spec['reference_fs_hz']:
        return float('nan')
    phase=np.arange(40)/40
    cycles=[]
    for k in (1,2,3):
        tx=end-k*period+phase*period
        cy=np.interp(tx,t,y);cy=cy-np.median(cy)
        norm=np.linalg.norm(cy)
        if norm<spec['numeric_floor']:return 0.
        cycles.append(cy/norm)
    return float(np.median([np.dot(cycles[a],cycles[b]) for a,b in ((0,1),(0,2),(1,2))]))


def template_shift(z,template,spec,use_all=False):
    # z=a+i*b, a*sin(h*theta)+b*cos(h*theta). Positive shift rotates z by exp(i*h*delta).
    h=np.asarray(spec['harmonics'])
    mask=np.ones(len(h),bool) if use_all else h>=2
    za=z[mask];zb=template[mask];hh=h[mask]
    denom=float(np.linalg.norm(za)*np.linalg.norm(zb))
    if denom<spec['numeric_floor']**2:return 0.,0.,0.
    grid=np.linspace(-np.pi,np.pi,spec['template_grid_points'],endpoint=False)
    product=za*np.conj(zb)
    corr=np.real(np.exp(-1j*np.outer(grid,hh))@product)/denom
    best=int(np.argmax(corr));delta=float(grid[best])
    step=2*np.pi/len(grid)
    # Quadratic local sub-bin interpolation, bounded to half a grid interval.
    ym,y0,yp=corr[(best-1)%len(grid)],corr[best],corr[(best+1)%len(grid)]
    curvature=ym-2*y0+yp
    if curvature<0:
        delta+=float(np.clip(.5*(ym-yp)/curvature,-.5,.5))*step
    peak=float(np.real(np.sum(product*np.exp(-1j*hh*delta)))/denom)
    far=np.abs(wrap(grid-delta))>=np.deg2rad(spec['template_other_peak_exclusion_deg'])
    margin=peak-float(np.max(corr[far]))
    return float(wrap(delta)),peak,margin


def fundamental_good(fit,spec):
    return (fit['fundamental_fraction']>=spec['minimum_fundamental_power_fraction'] and
            fit['fundamental_snr']>=spec['minimum_fundamental_to_residual_amplitude'])


def basic_good(fit,repeat,spec):
    return bool(np.isfinite(repeat) and repeat>=spec['minimum_repeat_correlation'] and
                fit['signal_rms']>spec['numeric_floor'] and
                fit['residual_ratio']<=spec['maximum_robust_residual_ratio'] and
                fit['inlier_fraction']>=spec['minimum_inlier_fraction'])


def bootstrap(t,y,end,spec,allow_template):
    lo,hi=spec['frequency_bounds_hz']
    # Cheap rejection before a bounded frequency fit; no running-amplitude or C2 oracle.
    rough=periodicity(t,y,50.,end,spec)
    if not np.isfinite(rough) or rough<spec['minimum_repeat_correlation']-.15:return None
    def objective(f):
        fit=fit_harmonics(t,y,0.,f,end,spec)
        return fit['residual_ratio']
    grid=np.arange(lo,hi+1e-9,spec['bootstrap_grid_step_hz'])
    losses=np.array([objective(f) for f in grid]);i=int(np.argmin(losses))
    a,b=grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]
    result=minimize_scalar(objective,bounds=(a,b),method='bounded',options={'xatol':1e-5})
    freq=float(result.x)
    fit=fit_harmonics(t,y,0.,freq,end,spec)
    repeat=periodicity(t,y,freq,end,spec)
    if not basic_good(fit,repeat,spec):return None
    h=np.asarray(spec['harmonics'])
    if fundamental_good(fit,spec):
        phase=float(np.angle(fit['z'][0]));origin='fundamental'
    elif allow_template:
        _,peak,margin=template_shift(fit['z'],fit['z'],spec,use_all=True)
        if margin<spec['template_peak_margin_minimum']:return None
        grid=np.linspace(0,2*np.pi,spec['template_grid_points'],endpoint=False)
        wave=np.sin(np.outer(grid,h))@fit['z'].real+np.cos(np.outer(grid,h))@fit['z'].imag
        phase=-float(grid[int(np.argmax(wave))]);origin='morphology'
    else:return None
    template=fit['z']*np.exp(-1j*h*phase)
    return dict(frequency=freq,theta=phase,template=template,origin=origin,
                dc=fit['dc'],amplitude=float(abs(fit['z'][0])),signal_rms=fit['signal_rms'],
                repeat=repeat,end=end)


def estimate(t,current,spec,allow_template=True):
    t=np.asarray(t,float);x=np.asarray(current,float)
    assert len(t)==len(x) and np.isfinite(x).all()
    fs=spec['reference_fs_hz'];step=round(spec['block_s']*fs)
    assert np.allclose(np.diff(t),1/fs,atol=1e-10)
    win=round(spec['observation_window_s']*fs)
    boot=round(spec['bootstrap_window_s']*fs)
    initialized=False;rows=[];cal=None;last_good_time=None
    theta=freq=last_end=0.;template=None;origin=None;good_run=0;previous_valid=False;segment=-1
    for k in range(step,len(x)+1,step):
        end=float(t[k-1]);start=float(t[k-step]);block=k//step-1
        row=dict(block=block,t_start_s=start,t_stop_s=float(start+spec['block_s']),estimate_time_s=end,
                 phase_rad=np.nan,frequency_hz=np.nan,phase_valid=False,mode='UNINITIALIZED',
                 observation_source='NONE',reference_segment=-1,origin='unavailable',
                 innovation_deg=np.nan,repeat_correlation=np.nan,robust_residual_ratio=np.nan,
                 fundamental_fraction=np.nan,template_correlation=np.nan,template_peak_margin=np.nan,
                 gain_alpha=np.nan,corrected=False,prediction_only=False,dc_estimate=np.nan)
        if not initialized:
            if k>=boot:
                cal=bootstrap(t[k-boot:k],x[k-boot:k],end,spec,allow_template)
                if cal is not None:
                    initialized=True;theta=cal['theta'];freq=cal['frequency'];template=cal['template']
                    origin=cal['origin'];last_end=end;good_run=1
                    row.update(phase_rad=theta,frequency_hz=freq,mode='ACQUIRE',origin=origin,
                               repeat_correlation=cal['repeat'],dc_estimate=cal['dc'])
            rows.append(row);continue
        dt=end-last_end;pred=theta+2*np.pi*freq*dt
        # Three actual periods, not only nominal60ms when f<50; bounded by available data.
        nwin=max(win,int(math.ceil(3*fs/freq))+1)
        tx=t[max(0,k-nwin):k];y=x[max(0,k-nwin):k]
        fit=fit_harmonics(tx,y,pred,freq,end,spec)
        repeat=periodicity(tx,y,freq,end,spec)
        delta,tc,margin=template_shift(fit['z'],template,spec,use_all=(origin=='morphology'))
        reliable=basic_good(fit,repeat,spec)
        source='NONE';innovation=0.
        if reliable and origin=='fundamental' and fundamental_good(fit,spec):
            source='FUNDAMENTAL';innovation=float(np.angle(fit['z'][0]))
        elif reliable and allow_template and tc>=spec['template_correlation_minimum'] and margin>=spec['template_peak_margin_minimum']:
            source='TEMPLATE';innovation=delta
        # One shared clock; bounded internal phase innovation and integrator/frequency state.
        corrected=source!='NONE'
        q=float(np.clip(1-fit['residual_ratio']/spec['maximum_robust_residual_ratio'],0,1))*max(0.,repeat if np.isfinite(repeat) else 0.)
        alpha=spec['clock_alpha_min']+(spec['clock_alpha_max']-spec['clock_alpha_min'])*q
        correction=np.clip(innovation,-np.deg2rad(spec['maximum_correction_innovation_deg']),np.deg2rad(spec['maximum_correction_innovation_deg']))
        theta=pred+alpha*correction if corrected else pred
        if corrected:
            freq=float(np.clip(freq+spec['clock_beta']*correction/(2*np.pi*dt),*spec['frequency_bounds_hz']))
        stable=corrected and abs(np.rad2deg(innovation))<=spec['maximum_valid_innovation_deg']
        good_run=good_run+1 if stable else 0
        valid=stable and good_run>=spec['consecutive_good_windows']
        if valid and not previous_valid:segment+=1
        if valid:last_good_time=end
        age=end-last_good_time if last_good_time is not None else np.inf
        mode=source if valid else ('ACQUIRE' if stable else ('HOLDOVER' if age<=spec['holdover_prediction_s']+1e-12 else 'UNAVAILABLE'))
        row.update(phase_rad=theta,frequency_hz=freq,phase_valid=valid,mode=mode,
                   observation_source=source,reference_segment=segment if valid else -1,origin=origin,
                   innovation_deg=np.rad2deg(innovation) if corrected else np.nan,
                   repeat_correlation=repeat,robust_residual_ratio=fit['residual_ratio'],
                   fundamental_fraction=fit['fundamental_fraction'],template_correlation=tc,template_peak_margin=margin,
                   gain_alpha=alpha,corrected=corrected,prediction_only=not corrected,dc_estimate=fit['dc'])
        rows.append(row);last_end=end;previous_valid=valid
    frame=pd.DataFrame(rows)
    if cal is not None:
        cal={**cal,'template_real':cal['template'].real.tolist(),'template_imag':cal['template'].imag.tolist()}
        cal.pop('template')
    return frame,cal


def complete_cycles(frame,raw_fs,spec):
    # Only interpolation of the REFERENCE clock. Raw waveform samples are never interpolated.
    rows=[]
    valid=frame[frame.phase_valid]
    for segment,d in valid.groupby('reference_segment',sort=False):
        if len(d)<2:continue
        time=d.estimate_time_s.to_numpy();theta=d.phase_rad.to_numpy()
        assert np.all(np.diff(time)>0) and np.all(np.diff(theta)>0)
        turns=np.arange(math.ceil(theta[0]/(2*np.pi)),math.floor(theta[-1]/(2*np.pi))+1)
        bounds=np.interp(turns*2*np.pi,theta,time)
        for turn,left,right in zip(turns[:-1],bounds[:-1],bounds[1:]):
            period=float(right-left)
            if not spec['minimum_period_s']<=period<=spec['maximum_period_s']:continue
            a,b=int(np.ceil(left*raw_fs-1e-6)),int(np.ceil(right*raw_fs-1e-6))
            rows.append(dict(reference_segment=int(segment),turn=int(turn),start_s=float(left),stop_s=float(right),
                             start_sample=a,stop_sample_exclusive=b,native_sample_count=b-a,period_s=period,
                             origin=str(d.origin.iloc[0]),span_start_s=float(time[0]),span_stop_s=float(time[-1])))
    return pd.DataFrame(rows,columns=['reference_segment','turn','start_s','stop_s','start_sample','stop_sample_exclusive',
        'native_sample_count','period_s','origin','span_start_s','span_stop_s'])
