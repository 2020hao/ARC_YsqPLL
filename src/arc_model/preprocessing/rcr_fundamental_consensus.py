"""RCR-CFT1 research candidate, derived from the user-provided RCR-v2 lifecycle.
Original vendor files remain immutable. One shared clock, fundamental-only past-cycle
consensus; no class labels, SOGI outputs, HF signal, or ground truth enter estimation.
"""
import math
import numpy as np
import pandas as pd
from .rcr_reference import _implementation
_implementation()  # provenance-checked original module load
from dual_reference_pll import wrap,fit_harmonics,periodicity,template_shift,fundamental_good,basic_good,bootstrap
from reference_recovery_v2 import candidate_agrees,bridge_value,complete_cycles

def coherent_fundamental(t,x,pred,freq,end,offset,spec):
    times=[];angles=[];snr=[];quality=[]
    for j in (3,2,1):
        left=end-j/freq;right=end-(j-1)/freq;mask=(t>left-1e-12)&(t<=right+1e-12);tx=t[mask];y=x[mask]
        if len(y)<2*len(spec['harmonics'])+4:return None,'insufficient_cycle_samples'
        fit=fit_harmonics(tx,y,pred,freq,end,spec)
        if (not fundamental_good(fit,spec) or fit['residual_ratio']>spec['maximum_robust_residual_ratio'] or fit['inlier_fraction']<spec['minimum_inlier_fraction']):return None,'cycle_fundamental_evidence_failed'
        times.append(float(np.mean(tx)));angles.append(float(wrap(np.angle(fit['z'][0])-offset)));snr.append(fit['fundamental_snr']);quality.append(float(np.clip(1-fit['residual_ratio']/spec['maximum_robust_residual_ratio'],0,1)))
    tau=np.asarray(times)-end;delta=np.unwrap(angles);B=np.column_stack((np.ones(3),tau));coef=np.linalg.lstsq(B,delta,rcond=None)[0];residual=float(np.max(abs(np.rad2deg(delta-B@coef))));fobs=freq+coef[1]/(2*np.pi)
    if residual>spec['dynamic_max_line_residual_deg']:return None,'phase_incoherent'
    if not spec['frequency_bounds_hz'][0]<=fobs<=spec['frequency_bounds_hz'][1]:return None,'frequency_outside'
    differences=np.diff(delta);span=float(abs(np.rad2deg(delta[-1]-delta[0])));trend=bool(np.all(differences>0) or np.all(differences<0));trend=trend and span>=max(2.,3*residual) and min(abs(differences))/max(float(max(abs(differences))),1e-12)>=.5
    constant=float(np.angle(np.mean(np.exp(1j*np.asarray(angles)))));q=min(quality)*max(0.,1-residual/spec['dynamic_max_line_residual_deg'])
    return dict(innovation=float(wrap(coef[0])),constant_innovation=constant,frequency=float(fobs),line_residual_deg=residual,trend_supported=trend,span_deg=span,min_snr=float(min(snr)),quality=float(q),earliest_sample_s=float(t[0]),latest_sample_s=float(t[-1])),'coherent'

def estimate(t,current,spec,variant="full",allow_template=True):
    if variant not in ("full","no_rescue","no_endpoint","disabled"):raise ValueError(variant)
    t=np.asarray(t,float);x=np.asarray(current,float);fs=spec['reference_fs_hz']
    assert len(t)==len(x) and np.isfinite(x).all() and np.allclose(np.diff(t),1/fs,atol=1e-10)
    step=round(spec['block_s']*fs);boot=round(spec['bootstrap_window_s']*fs)
    rows=[];events=[];cal=None;candidate=None;count=0;initialized=False
    theta=freq=last_end=0.;template=None;origin=None;offset=None;bridge=[]
    last_observed=None;last_valid=None;good_run=0;previous_valid=False;segment=-1;epoch=0
    reset_candidate=None;reset_count=0
    for k in range(step,len(x)+1,step):
        end=float(t[k-1]);start=float(t[k-step])
        row=dict(block=k//step-1,t_start_s=start,t_stop_s=start+spec['block_s'],estimate_time_s=end,
            phase_rad=np.nan,frequency_hz=np.nan,phase_valid=False,mode='UNINITIALIZED',observation_source='NONE',
            reference_segment=-1,reference_epoch=epoch,origin='unavailable',innovation_deg=np.nan,
            repeat_correlation=np.nan,robust_residual_ratio=np.nan,fundamental_fraction=np.nan,
            template_correlation=np.nan,template_peak_margin=np.nan,gain_alpha=np.nan,corrected=False,
            prediction_only=False,dc_estimate=np.nan,event='',bridge_offset_deg=np.nan,
            dynamic_status='not_requested',dynamic_frequency_hz=np.nan,dynamic_line_residual_deg=np.nan,
            comparable_to_initial=(epoch==0),latest_input_time_s=end)
        if not initialized:
            if k>=boot:
                attempt=bootstrap(t[k-boot:k],x[k-boot:k],end,spec,allow_template)
                if attempt is None:candidate=None;count=0
                else:
                    count=count+1 if candidate_agrees(candidate,attempt,spec) else 1;candidate=attempt
                    need=spec['fundamental_bootstrap_confirmations'] if attempt['origin']=='fundamental' else spec['morphology_bootstrap_confirmations']
                    row['mode']='BOOTSTRAP_PROBATION'
                    if count>=need:
                        cal=attempt.copy();initialized=True;theta=cal['theta'];freq=cal['frequency']
                        template=cal['template'];origin=cal['origin'];offset=0. if origin=='fundamental' else None
                        last_end=end;last_observed=end;good_run=1
                        row.update(phase_rad=theta,frequency_hz=freq,mode='ACQUIRE',origin=origin,event='confirmed_initialization')
                        events.append(dict(time_s=end,event='confirmed_initialization',epoch=0,origin=origin,confirmations=count))
            rows.append(row);continue
        dt=end-last_end;pred=theta+2*np.pi*freq*dt
        nwin=max(round(spec['observation_window_s']*fs),int(math.ceil(3*fs/freq))+1)
        tx=t[max(0,k-nwin):k];y=x[max(0,k-nwin):k]
        fit=fit_harmonics(tx,y,pred,freq,end,spec);repeat=periodicity(tx,y,freq,end,spec)
        delta,tc,margin=template_shift(fit['z'],template,spec,origin=='morphology')
        reliable=basic_good(fit,repeat,spec);fg=reliable and fundamental_good(fit,spec)
        tg=allow_template and reliable and tc>=spec['template_correlation_minimum'] and margin>=spec['template_peak_margin_minimum']
        # A HIGHER-HARMONIC anchor allows amplitude of a newly returned fundamental to change.
        # Both observations use the frozen template's original coordinate; never truth/current-clock alone.
        if origin=='morphology' and offset is None:
            hd,hc,hm=template_shift(fit['z'],template,spec,False)
            if fg and hc>=spec['bridge_shape_correlation'] and hm>=spec['template_peak_margin_minimum']:
                bridge.append(float(wrap(np.angle(fit['z'][0])-hd)))
                bridge=bridge[-spec['bridge_consecutive_windows']:]
                val=bridge_value(bridge,spec)
                if val is not None:
                    offset=val;row['event']='frozen_origin_bridge'
                    events.append(dict(time_s=end,event='frozen_origin_bridge',epoch=epoch,offset_deg=float(np.rad2deg(offset)),
                        observations=len(bridge),offsets_deg=np.rad2deg(bridge).tolist()))
            else:bridge=[]
        source='NONE';innovation=0.
        if fg and offset is not None:source='FUNDAMENTAL';innovation=float(wrap(np.angle(fit['z'][0])-offset))
        elif tg:source='TEMPLATE';innovation=delta
        corrected=source!='NONE';dyn=None;dynamic_ok=True;dyn_status='disabled'
        original_source=source;consensus=None;used_rescue=False;used_endpoint=False
        if variant!='disabled' and origin=='fundamental' and offset is not None and source in ('NONE','FUNDAMENTAL'):
            consensus,dyn_status=coherent_fundamental(tx,y,pred,freq,end,offset,spec)
            if consensus is not None:
                rescue=(source=='NONE' and variant!='no_rescue')
                endpoint=(variant!='no_endpoint' and consensus['trend_supported'])
                if endpoint and source=='FUNDAMENTAL' and abs(np.rad2deg(wrap(consensus['innovation']-innovation)))>10.:
                    endpoint=False;dyn_status='endpoint_disagrees_with_window'
                if rescue or (source=='FUNDAMENTAL' and endpoint):
                    proposal=consensus['innovation'] if endpoint else consensus['constant_innovation']
                    if abs(np.rad2deg(proposal))<=spec['maximum_correction_innovation_deg']:
                        innovation=proposal;corrected=True;used_rescue=rescue;used_endpoint=endpoint
                        source='FUNDAMENTAL_CONSENSUS';dyn=consensus if endpoint else None
        # Reacquisition can end permanent template trapping, but cannot invent a missing origin bridge.
        if corrected:last_observed=end;reset_candidate=None;reset_count=0
        elif end-last_observed>=spec['recovery_no_observation_s']-1e-12 and k>=boot:
            attempt=bootstrap(t[k-boot:k],x[k-boot:k],end,spec,False)
            if attempt is None:reset_candidate=None;reset_count=0
            else:
                reset_count=reset_count+1 if candidate_agrees(reset_candidate,attempt,spec) else 1;reset_candidate=attempt
                if reset_count>=spec['fundamental_bootstrap_confirmations']:
                    epoch+=1;theta=attempt['theta']+2*np.pi*round((pred-attempt['theta'])/(2*np.pi))
                    freq=attempt['frequency'];template=attempt['template'];origin='fundamental';offset=0.;bridge=[]
                    good_run=1;previous_valid=False;last_valid=None;last_observed=end;last_end=end
                    row.update(phase_rad=theta,frequency_hz=freq,mode='ACQUIRE',origin=origin,event='origin_reset_no_bridge',
                        reference_epoch=epoch,comparable_to_initial=False,dc_estimate=attempt['dc'])
                    events.append(dict(time_s=end,event='origin_reset_no_bridge',epoch=epoch,origin=origin,
                        theta=theta,frequency=freq,reason='No usable observation for >=0.20s; 3 confirmed independent-of-clock fundamental fits. Global cycle count unknown.'))
                    reset_candidate=None;reset_count=0;rows.append(row);continue
        q=float(np.clip(1-fit['residual_ratio']/spec['maximum_robust_residual_ratio'],0,1))*max(0.,repeat if np.isfinite(repeat) else 0.)
        if used_rescue or used_endpoint:q=consensus['quality']
        alpha=spec['clock_alpha_min']+(spec['clock_alpha_max']-spec['clock_alpha_min'])*q
        correction=float(np.clip(innovation,-np.deg2rad(spec['maximum_correction_innovation_deg']),np.deg2rad(spec['maximum_correction_innovation_deg'])))
        theta=pred+alpha*correction if corrected else pred
        if corrected:
            target=freq+spec['clock_beta']*correction/(2*np.pi*dt)
            if dyn is not None:target+=spec['dynamic_frequency_gain']*(dyn['frequency']-freq)
            freq=float(np.clip(target,*spec['frequency_bounds_hz']))
        stable=corrected and dynamic_ok and abs(np.rad2deg(innovation))<=spec['maximum_valid_innovation_deg']
        good_run=good_run+1 if stable else 0;valid=bool(stable and good_run>=spec['consecutive_good_windows'])
        if valid and not previous_valid:segment+=1
        if valid:last_valid=end
        age=end-last_valid if last_valid is not None else np.inf
        mode=source if valid else ('ACQUIRE' if stable else ('HOLDOVER' if age<=spec['holdover_prediction_s']+1e-12 else 'UNAVAILABLE'))
        row.update(phase_rad=theta,frequency_hz=freq,phase_valid=valid,mode=mode,observation_source=source,
            reference_segment=segment if valid else -1,reference_epoch=epoch,origin=origin,
            innovation_deg=float(np.rad2deg(innovation)) if corrected else np.nan,repeat_correlation=repeat,
            robust_residual_ratio=fit['residual_ratio'],fundamental_fraction=fit['fundamental_fraction'],
            template_correlation=tc,template_peak_margin=margin,gain_alpha=alpha,corrected=corrected,
            prediction_only=not corrected,dc_estimate=fit['dc'],bridge_offset_deg=float(np.rad2deg(offset)) if offset is not None else np.nan,
            dynamic_status=dyn_status,dynamic_frequency_hz=dyn['frequency'] if dyn else np.nan,
            dynamic_line_residual_deg=dyn['line_residual_deg'] if dyn else np.nan,comparable_to_initial=(epoch==0))
        row.update(original_observation_source=original_source,consensus_status=dyn_status,
                   consensus_rescue=used_rescue,consensus_endpoint=used_endpoint,
                   consensus_line_residual_deg=consensus['line_residual_deg'] if consensus else np.nan,
                   consensus_min_snr=consensus['min_snr'] if consensus else np.nan,
                   consensus_frequency_hz=consensus['frequency'] if consensus else np.nan,
                   consensus_phase_span_deg=consensus['span_deg'] if consensus else np.nan)
        rows.append(row);last_end=end;previous_valid=valid
    if cal is not None:
        cal={**cal,'template_real':cal['template'].real.tolist(),'template_imag':cal['template'].imag.tolist()};cal.pop('template')
    return pd.DataFrame(rows),dict(initialization=cal,events=events)

