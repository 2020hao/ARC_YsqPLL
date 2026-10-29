"""RSPA1: reference-state-conditioned phase dynamics with a separate invariant stream."""
from __future__ import annotations
import numpy as np
from arc_model.methods.cycle_dynamics import cycle_distributions
ARMS=('full','no_state','no_phase_dynamic','no_P4','no_quality','phasefree_dynamic','fixed20_dynamic')
def invariant_features(energy:np.ndarray,period_seconds:np.ndarray)->np.ndarray:
 e=np.maximum(np.asarray(energy,float),0);period=np.asarray(period_seconds,float)
 if e.ndim!=3 or e.shape[1:]!=(2,20) or len(period)!=len(e) or (period<=0).any():raise ValueError('Expected completed two-band cycles')
 power=e.sum(-1)/period[:,None];p=cycle_distributions(e);entropy=-(p*np.log(np.maximum(p,1e-30))).sum(-1)/np.log(20)
 return np.c_[np.log(power+1e-12),entropy].astype(np.float32)
def quality_score(repeat,residual,innovation_deg):
 return np.nan_to_num(np.clip(repeat,0,1)*np.clip(1-np.asarray(residual)/.35,0,1)*np.clip(1-abs(np.asarray(innovation_deg))/35.,0,1),nan=0.).astype(np.float32)
def design_matrix(dynamic,invariant,native,quality,keep,mu,sd,p_mu,p_sd,arm):
 if arm not in ARMS:raise ValueError(arm)
 d=np.asarray(dynamic);k=np.asarray(keep,int);n=len(d);x=np.zeros((n,28),np.float32);values=((d[:,k]-mu)/sd).astype(np.float32)
 q=np.ones(n,np.float32) if arm=='no_state' else np.asarray(native,dtype=np.float32)*(1. if arm=='no_quality' else np.asarray(quality,dtype=np.float32))
 for j,col in enumerate(k):
  if col<32:
   values[:,j]*=d[:,32+col//8]
   if col%4!=3:values[:,j]*=q
 x[:,:len(k)]=values;x[:,24:]=((invariant-p_mu)/p_sd).astype(np.float32)
 if arm=='no_phase_dynamic':
  for j,col in enumerate(k):
   if col<32 and col%4!=3:x[:,j]=0
 if arm=='no_P4':x[:,24:]=0
 if not np.isfinite(x).all():raise ValueError('Nonfinite adapter input')
 return x
