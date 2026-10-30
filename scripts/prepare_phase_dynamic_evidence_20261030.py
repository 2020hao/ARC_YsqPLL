"""Four predefined position descriptors and controls from native-HF integrals.

No labels, model scores or train/test outcomes choose features or bin locations.
The input integrals were computed on native1/2MHz current, not 5kHz references.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import yaml
import _bootstrap as boot

ROOT=boot.ROOT
OUT=ROOT/'outputs/phase_dynamic_evidence_20261030'
CONFIG=ROOT/'configs/modules/phase_dynamic_evidence_20261030.yaml'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def distributions(energy):
    e=np.asarray(energy,dtype=np.float64)
    if e.ndim!=3 or e.shape[1:]!=(2,20) or not np.isfinite(e).all() or np.min(e)<0:
        raise ValueError('Finite nonnegative native-band energy [N,2,20] required')
    total=e.sum(-1,keepdims=True)
    return np.divide(e,total,out=np.zeros_like(e),where=total>0)


def phase4(p):
    return np.stack([p[..., [19,0]].sum(-1),p[..., [9,10]].sum(-1)],axis=-1).reshape(len(p),4).astype('float32')


def phasefree4(p):
    return np.stack([p.max(-1),(p*p).sum(-1)],axis=-1).reshape(len(p),4).astype('float32')


def cycle_shifts(table,fs):
    values=[]
    for r in table.itertuples():
        key=f'phase_dynamic_evidence_20261030|{r.record_id}|{r.kind}|{r.segment}|{r.epoch}|{round(r.start_seconds*fs)}|{round(r.end_seconds*fs)}'
        values.append(int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'little')%20)
    return np.asarray(values,dtype=np.int64)


def prepare():
    if (OUT/'FEATURE_AUDIT.json').exists():
        audit=json.loads((OUT/'FEATURE_AUDIT.json').read_text())
        for r in audit['identities']:assert sha(r['path'])==r['sha256'],r['path']
        print('FEATURES_ALREADY_VERIFIED');return
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load(CONFIG.read_text());datasets=yaml.safe_load((ROOT/'configs/datasets.yaml').read_text())
    identities=[dict(path=str(CONFIG),sha256=sha(CONFIG)),dict(path=str(Path(__file__).resolve()),sha256=sha(__file__))]
    diagnostics=[]
    for ds in cfg['datasets']:
        c=datasets[ds];d=OUT/'features'/ds;d.mkdir(parents=True,exist_ok=False)
        rcr=ROOT/c['prepared_roots']['current_self_sync'];fixed=ROOT/c['prepared_roots']['fixed20']
        t=pd.read_csv(rcr/'cycles.csv');u=pd.read_csv(fixed/'cycles.csv')
        for col in ['sample_id','record_id','source_group','condition_key','label','anchor_index']:
            assert np.array_equal(t[col],u[col]),col
        keep=np.load(ROOT/c['reviewed_indices']);assert len(keep)==c['reviewed_windows']
        pr=distributions(np.load(rcr/'energy.npy'));pf=distributions(np.load(fixed/'energy.npy'))
        assert len(pr)==len(t)==len(pf)
        shift=cycle_shifts(t,c['native_sample_rate_hz']);valid=t.kind.eq('pll').to_numpy()
        shifted=np.stack([np.roll(v,int(s),axis=-1) for v,s in zip(pr,shift)])
        shifted[~valid]=pr[~valid]
        arrays=dict(rcr_phase4=phase4(pr),fixed_phase4=phase4(pf),rcr_shift4=phase4(shifted),rcr_phasefree4=phasefree4(pr))
        assert np.allclose(phasefree4(shifted),phasefree4(pr),rtol=1e-6,atol=1e-7)
        assert np.allclose(np.sort(shifted,axis=-1),np.sort(pr,axis=-1),rtol=0,atol=0)
        for key,a in arrays.items():
            assert a.shape==(len(t),4) and np.isfinite(a).all() and a.min()>=0 and a.max()<=1+1e-6
            np.save(d/(key+'.npy'),a);identities.append(dict(path=str(d/(key+'.npy')),sha256=sha(d/(key+'.npy'))))
        np.save(d/'circular_shifts.npy',shift)
        for p in [rcr/'energy.npy',fixed/'energy.npy',rcr/'cycles.csv',fixed/'cycles.csv',ROOT/c['split_file'],ROOT/c['reviewed_indices'],d/'circular_shifts.npy']:
            identities.append(dict(path=str(p),sha256=sha(p)))
        fs=c['native_sample_rate_hz']
        for _,g in t.groupby(['record_id','kind','segment','epoch','start_seconds','end_seconds'],dropna=False):
            assert len(np.unique(shift[g.index]))==1
        diagnostics.append(dict(dataset=ds,total_rows=len(t),reviewed=len(keep),native_sample_rate_hz=fs,
            valid_RCR=int(valid[keep].sum()),nominal_fallback=int((~valid[keep]).sum()),
            position_features='band0_0deg,band0_180deg,band1_0deg,band1_180deg; +/-18deg reference neighborhoods',
            phasefree_features='band0_peak_fraction,band0_concentration,band1_peak_fraction,band1_concentration',
            valid_shift_nonzero=int((shift[keep[valid[keep]]]!=0).sum()),
            future_information=False,label_driven_selection=False))
    audit=dict(status='pass',configuration_count=12,reused_configuration_count=4,new_configuration_count=8,
               new_outer_arm_bundles=240,new_linear_head_fits=420,new_backbone_fits=0,
               current_reference_not_voltage_or_independent_event_truth=True,
               zero_energy_policy='all four fractions zero',shift_only_valid_RCR=True,
               diagnoses=diagnostics,identities=identities)
    (OUT/'FEATURE_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print('FEATURES_READY',len(identities))


if __name__=='__main__':prepare()
