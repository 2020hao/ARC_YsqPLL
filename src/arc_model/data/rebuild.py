"""Rebuild the main-cohort caches from source-verified 2 MHz current files."""
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from arc_model.data.raw import one
from arc_model.features.physical import descriptors
from arc_model.features.descriptors import extend
from arc_model.models.hydra.features import transform_windows
from arc_model.representations.mbwvd import mbwvd
from arc_model.utils.io import save_json


def check_record(dataset, raw_root, record_id):
    record=dataset.ledger[record_id]
    wave,features,native=one(record,Path(raw_root))
    ix=np.flatnonzero(dataset.record_ids==record_id)
    errors={key:float(np.max(np.abs(value-dataset.array(key)[ix])))
            for key,value in [('wave_50000',wave),('features',features),('wave_250000',native)]}
    errors['mbwvd']=float(np.max(np.abs(mbwvd(native[0,0])[0]-dataset.array('mbwvd')[ix[0],0])))
    return dict(record_id=record_id, source_sha256=record['sha256'], windows=len(ix),
                max_abs_errors=errors, passed=max(errors.values())<1e-6)


def rebuild(records, raw_root, output):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if any(r['cohort']!='main' for r in records):
        raise ValueError('Only real-appliance records may be rebuilt')
    output.mkdir(parents=True,exist_ok=True)
    waves,features,natives,ids=[],[],[],[]
    for record in records:
        a,b,c=one(record,Path(raw_root))
        waves.append(a);features.append(b);natives.append(c)
        ids.extend([record['record_id']]*len(a))
        print(f"SOURCE {record['record_id']} windows={len(a)}",flush=True)
    wave,physical,native=map(np.concatenate,(waves,features,natives))
    for name,a in [('record_ids',np.array(ids)),('wave_50000',wave),('features',physical),('wave_250000',native)]:
        np.save(output/(name+'.npy'),a)
    low=resample_poly(wave,1,4,axis=-1).astype('float32')
    d1=np.gradient(low,axis=-1);d2=np.gradient(d1,axis=-1)
    np.save(output/'fusion_temporal.npy',np.stack([low,d1,d2],axis=2)[:,:,None])
    np.save(output/'fusion_spectrum.npy',np.abs(np.fft.rfft(low,axis=-1))[...,1:].astype('float32'))
    raw_features=np.array([[descriptors(w,np.zeros((2,len(w)))) for w in pair] for pair in low])
    low_features=extend(low,raw_features);low_features[:,:,4:22]=0
    np.save(output/'features_low.npy',low_features)
    for seed in range(3):
        np.save(output/f'hydra_features_seed{seed}.npy',
                np.stack([transform_windows(wave[:,variant],seed) for variant in range(2)],axis=1))
    tf=np.lib.format.open_memmap(output/'mbwvd.npy',mode='w+',dtype='float32',shape=(len(ids),2,100,100))
    for i in range(len(ids)):
        for variant in range(2):
            tf[i,variant]=mbwvd(native[i,variant])[0]
        if i%100==0:
            print(f'MBWVD {i}/{len(ids)}',flush=True)
    tf.flush()
    save_json(output/'rebuild_provenance.json',dict(records=len(records),windows=len(ids),
              sources={r['record_id']:r['sha256'] for r in records},scope='real_appliances_only'))
