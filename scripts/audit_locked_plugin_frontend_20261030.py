"""Connect native raw frontend APIs to the model-native experiment caches."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import _bootstrap
import validate_locked_plugin_20261030 as v

rows=[]
for ds in v.DS:
    rid=0
    src=v.RAW/ds/f'{rid:04}'
    meta=json.loads((src/'COMPLETE.json').read_text())
    raw=np.load(src/'raw_current.npy',mmap_mode='r')
    fs=int(meta['fs'])
    origin=0.
    if ds=='zenodo5501527':
        with Path(meta['source_file']).open() as f:origin=float(f.readline().split()[0])
    table=pd.read_csv(v.PREP/ds/'recovery/cycles.csv')
    idx=int(table[(table.record_id==rid)&table.kind.eq('pll')].index[0])
    q=table.loc[idx]
    clock=pd.read_csv(src/'recovery_clock.csv')
    z=clock[clock.phase_valid&clock.reference_segment.eq(q.segment)]
    tt=z.estimate_time_s.to_numpy();pp=z.phase_rad.to_numpy()
    a=round((q.start_seconds-origin)*fs)/fs
    b=round((q.end_seconds-origin)*fs)/fs
    grid=np.interp(np.linspace(np.interp(a,tt,pp),np.interp(b,tt,pp),1001),pp,tt)+origin
    # Whole native-rate recordings remain causal because every selected FIR
    # sample precedes the reviewed decision anchor; truncate and re-run to check.
    prefix=raw[:min(len(raw),int(np.floor((q.anchor_seconds-origin)*fs))+1)]
    for model in v.PACKAGES:
        frontend=v.package(model,'frontend').aligned_wave
        got=frontend(raw,fs,grid,origin_seconds=origin,anchor_seconds=q.anchor_seconds)
        prior=frontend(prefix,fs,grid,origin_seconds=origin,anchor_seconds=q.anchor_seconds)
        if model=='ArcFENet':
            expected=np.load(v.PREP/ds/'recovery/native.npy',mmap_mode='r')[idx]
            err=float(abs(got-expected).max());causal=float(abs(got-prior).max())
        elif model=='HYDRA':
            expected=np.load(v.OUT/'data'/ds/'aligned_rectified50k.npy',mmap_mode='r')[idx]
            err=float(abs(got-expected).max());causal=float(abs(got-prior).max())
        else:
            expected={'temporal':np.load(v.PREP/ds/'recovery/temporal.npy',mmap_mode='r')[idx],
                      'spectrum':np.load(v.PREP/ds/'recovery/spectrum.npy',mmap_mode='r')[idx]}
            err=max(float(abs(got[k]-expected[k]).max()) for k in expected)
            causal=max(float(abs(got[k]-prior[k]).max()) for k in expected)
        assert err<1e-4,(ds,model,err)
        assert causal==0.,(ds,model,causal)
        rows.append(dict(dataset=ds,record_id=rid,anchor_index=idx,model=model,native_fs=fs,
                         cache_api_max_error=err,prefix_causality_max_error=causal))
        print('PLUGIN_FRONTEND_API',rows[-1],flush=True)
v.save(v.OUT/'FRONTEND_API_AUDIT.json',dict(status='pass',scope='One native1/2MHz valid RCR cycle per dataset and all3 independent frontend APIs; full151 HYDRA fixed20 raw rebuild checked separately',checks=rows))
