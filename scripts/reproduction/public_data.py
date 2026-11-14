"""Rebuild the paper's public phase representation and fixed-window baseline inputs."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, firwin, resample_poly, sosfilt, upfirdn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _bootstrap as boot  # noqa: E402
from arc_model.preprocessing.causal_window_selection import parallel_candidates, select_available  # noqa: E402
from arc_model.preprocessing.rcr_cycle_v2 import CycleRCRConfig, estimate  # noqa: E402

FILES_BY_FOLD = (
    ('C3_foodprocessor_carbonizedpath.dat', 'C3_kettle_carbonizedpath.dat'),
    ('C3_computer_carbonizedpath.dat', 'C3_drill550_carbonizedpath.dat',
     'C3_vacuumcleaner1200W_carbonizedpath.dat'),
    ('C3_drill710_carbonizedpath.dat', 'C3_heater_carbonizedpath.dat'),
    ('C3_dolcegusto_carbonizedpath.dat', 'C3_pump_carbonizedpath.dat',
     'C3_vacuumcleaner1250_carbonizedpath.dat'),
    ('C3_halogen_carbonizedpath.dat', 'C3_jigsaw135_carnonizedpath.dat',
     'C3_printer_carbonizedpath.dat'),
)
RATE = 1_000_000


def cycles_from_dense(phase, valid, epochs):
    """Frozen 2 ms clock integration and physical-cycle boundaries."""
    good = []
    segment = 0
    i = 0
    while i < len(phase):
        if not valid[i] or not np.isfinite(phase[i]):
            i += 1
            continue
        j = i + 1
        while j < len(phase) and valid[j] and epochs[j] == epochs[i] and np.isfinite(phase[j]):
            j += 1
        if j - i >= 2:
            turns = np.arange(int(np.ceil(phase[i]/(2*np.pi))), int(np.floor(phase[j-1]/(2*np.pi)))+1)
            bounds = np.interp(turns*2*np.pi, phase[i:j], np.arange(i, j)/(1/CycleRCRConfig().update_seconds))
            for a,b in zip(bounds[:-1],bounds[1:]):
                if 1/55 <= b-a <= 1/45:
                    good.append((int(np.ceil(a*RATE)), int(np.ceil(b*RATE)), segment, int(epochs[i])))
        segment += 1
        i = j
    return pd.DataFrame(good, columns=['start_sample','stop_sample_exclusive','reference_segment','reference_epoch'])


def history(ledger):
    indices = np.zeros((len(ledger),9), dtype=np.int64)
    masks = np.zeros_like(indices, dtype=np.float32)
    for _, group in ledger.groupby(['kind','segment','epoch']):
        recent = []
        for i in group.sort_values('end_seconds').index:
            row = ledger.loc[i]
            if recent and abs(row.start_seconds-ledger.loc[recent[-1],'end_seconds']) > 3e-6:
                recent = []
            recent = (recent+[i])[-9:]
            indices[i,-len(recent):] = recent
            masks[i,-len(recent):] = 1
    return indices, masks


def phase_windows(raw, origin, anchors):
    low = upfirdn(firwin(401,20_000,fs=RATE,window=('kaiser',7.5)),raw,down=20)[:(len(raw)+19)//20]
    reference = sosfilt(butter(2,800,fs=50_000,output='sos'),low)[::10]
    clock = estimate(reference, chunk_samples=10)
    times = clock.estimate_time_s.to_numpy()
    phase = clock.phase_rad.to_numpy()
    valid = clock.phase_usable.to_numpy()
    entries = cycles_from_dense(phase, valid, clock.reference_epoch.to_numpy())
    shift = int(round(times[0]*RATE))
    entries[['start_sample','stop_sample_exclusive']] += shift
    candidates = parallel_candidates(entries,len(raw),RATE)
    rows = []
    for c in candidates:
        a,b = c['start_sample']/RATE,c['stop_sample_exclusive']/RATE
        available = times[min(np.searchsorted(times,b),len(times)-1)] if c['kind']=='pll' else b
        rows.append(dict(start_seconds=a+origin,end_seconds=b+origin,available_seconds=available+origin,
                         kind=c['kind'],segment=c['segment'],epoch=c['reference_epoch']))
    clock_idx = np.maximum(np.searchsorted(times,anchors-origin+1e-10,side='right')-1,0)
    chosen = select_available(rows,anchors,valid[clock_idx],clock.reference_segment.to_numpy()[clock_idx],
                              clock.reference_epoch.to_numpy()[clock_idx])
    ledger = pd.DataFrame(rows)
    lp = sosfilt(butter(4,1000,fs=50_000,output='sos'),low)
    shape = np.r_[0.,np.cumsum(lp)]/50_000
    shape_times = np.arange(len(shape))/50_000
    prefixes = []
    for band in ((10_000,100_000),(100_000,400_000)):
        z = sosfilt(butter(3,band,fs=RATE,btype='bandpass',output='sos'),raw)
        prefixes.append(np.r_[0.,np.cumsum(z*z)]/RATE)
    profiles = np.empty((len(ledger),3,64),dtype=np.float32)
    for i,row in ledger.iterrows():
        a,b = row.start_seconds-origin,row.end_seconds-origin
        grid = np.linspace(a,b,65)
        if row.kind=='pll':
            mask=valid & (clock.reference_segment.to_numpy()==row.segment) & (times<=row.available_seconds-origin+1e-10)
            tx, px = times[mask], phase[mask]
            grid = np.interp(np.linspace(np.interp(a,tx,px),np.interp(b,tx,px),65),px,tx)
        edges = np.clip(np.ceil(grid*RATE).astype(np.int64),0,len(raw))
        edges[0],edges[-1] = int(round(a*RATE)),int(round(b*RATE))
        durations = np.diff(edges/RATE)
        profiles[i,0] = np.diff(np.interp(edges/RATE,shape_times,shape))/durations
        for band,prefix in enumerate(prefixes,1):
            profiles[i,band] = np.log10(np.maximum(np.diff(prefix[edges])/durations,1e-20))
    indices,masks = history(ledger)
    x = profiles[indices[chosen]]
    mask = masks[chosen]
    probability = np.exp(np.clip(x[:,:,1:3]-x[:,:,1:3].max(-1,keepdims=True),-100,0)*np.log(10))
    probability /= np.maximum(probability.sum(-1,keepdims=True),1e-20)
    delta = np.zeros_like(probability)
    delta[:,1:] = (probability[:,1:]-probability[:,:-1])*(mask[:,1:]*mask[:,:-1])[:,:,None,None]
    result = np.zeros(x.shape[:2]+(8,64),np.float32)
    result[:,:,:5] = np.concatenate((x,delta),axis=2)
    return result,mask,ledger.iloc[chosen].reset_index(drop=True)


def baseline_windows(raw, origin, table):
    fir = firwin(401,20_000,fs=RATE,window=('kaiser',7.5))
    signed = upfirdn(fir,raw,down=20)[:(len(raw)+19)//20]
    rectified = upfirdn(fir,abs(raw),down=20)
    native = upfirdn(firwin(401,112_500,fs=RATE,window=('kaiser',7.5)),raw,down=4)[:(len(raw)+3)//4]
    starts = np.round((table.start_seconds-origin)*RATE)/RATE
    ends = np.round((table.end_seconds-origin)*RATE)/RATE
    grid = np.stack([np.linspace(a,b,1001) for a,b in zip(starts,ends)])
    signed_windows = np.interp(grid[:,:-1]*50_000,np.arange(len(signed)),signed).astype(np.float32)
    rect_grid = np.stack([np.linspace(a,b,1000,endpoint=False) for a,b in zip(table.start_seconds,table.end_seconds)])
    rect_windows = np.interp((rect_grid-origin)*50_000,np.arange(len(rectified)),rectified).astype(np.float32)
    positions = np.stack([np.interp(np.arange(5000)/5000,np.arange(1001)/1000,g) for g in grid])
    fenet_wave = np.interp(positions*250_000,np.arange(len(native)),native).astype(np.float32)
    fusion = resample_poly(signed_windows,1,4,axis=-1).astype(np.float32)
    diff = np.gradient(fusion,axis=-1)
    temporal = np.stack([fusion,diff,np.gradient(diff,axis=-1)],1)[:,None]
    spectrum = abs(np.fft.rfft(fusion,axis=-1))[:,1:].astype(np.float32)
    return rect_windows,fenet_wave,temporal,spectrum


def prepare(recordings, protocol, out):
    if (out/'PREPARED').exists():
        raise FileExistsError(f'Public inputs already prepared in {out}.')
    table = pd.read_csv(protocol, float_precision='round_trip')
    required = {'record_id','source_group','sample_id','label','condition_key','anchor_seconds','start_seconds','end_seconds','interval_start','interval_end','retain'}
    if not required.issubset(table.columns):
        raise ValueError(f'Missing protocol fields: {required-set(table.columns)}')
    if len(table)!=1364 or table.sample_id.nunique()!=1364 or table.source_group.nunique()!=13:
        raise ValueError('Expected the frozen 1,364-candidate public annotation manifest.')
    lookup = {name:f for f,names in enumerate(FILES_BY_FOLD) for name in names}
    if set(table.source_group) != set(lookup):
        raise ValueError('Public annotation filenames must match the 13 frozen C3 sources.')
    if table.groupby('source_group').record_id.nunique().max()!=1 or table.record_id.nunique()!=13:
        raise ValueError('Expected exactly one recording ID per public file.')
    table['fold'] = table.source_group.map(lookup)
    keep = np.flatnonzero(table.retain.to_numpy()==1)
    assert len(keep)==1357 and table.iloc[keep].label.value_counts().to_dict()=={0:794,1:563}
    out.mkdir(parents=True,exist_ok=True)
    data = {'realtime_raw':np.empty((len(table),9,8,64),np.float32),
            'realtime_mask':np.empty((len(table),9),np.float32),'wave':np.empty((len(table),1000),np.float32),
            'fenet_wave':np.empty((len(table),5000),np.float32),'temporal':np.empty((len(table),1,3,250),np.float32),
            'spectrum':np.empty((len(table),125),np.float32)}
    frame = table.copy()
    for filename, t in table.groupby('source_group',sort=False):
        path = recordings/filename
        if path.parent != recordings or not filename.startswith('C3_'):
            raise ValueError(filename)
        samples = np.loadtxt(path,usecols=(0,1))
        if not np.allclose(np.diff(samples[:,0]),1/RATE,atol=2e-9,rtol=0):
            raise ValueError(f'Invalid time grid: {filename}')
        origin = samples[0,0]
        raw = samples[:,1].copy()
        assert np.isfinite(raw).all()
        x,mask,selected = phase_windows(raw,origin,t.anchor_seconds.to_numpy())
        data['realtime_raw'][t.index],data['realtime_mask'][t.index] = x,mask
        for n,v in zip(('wave','fenet_wave','temporal','spectrum'),baseline_windows(raw,origin,t)):
            data[n][t.index] = v
        for k in ('start_seconds','end_seconds','available_seconds','kind','segment','epoch'):
            frame.loc[t.index,k] = selected[k].to_numpy()
        print(filename, len(t), flush=True)
    assert (frame.available_seconds <= frame.anchor_seconds+1e-9).all()
    guard = ((frame.start_seconds >= frame.interval_start + .04 - 1e-9)
             & (frame.end_seconds <= frame.interval_end - .04 + 1e-9))
    if not guard.iloc[keep].all():
        raise ValueError('Retained window crosses the reviewed 40 ms label guard.')
    for f,n in enumerate([219,303,215,311,309]):
        test = keep[table.fold.to_numpy()[keep]==f]
        train = np.setdiff1d(keep,test)
        assert len(test)==n and set(table.iloc[test].record_id).isdisjoint(table.iloc[train].record_id)
        np.savez(out/f'roles_f{f}.npz',outer_train=train,test=test)
    for key,value in data.items():
        np.save(out/f'{key}.npy',value)
    np.save(out/'keep.npy',keep)
    frame.to_csv(out/'cycles.csv',index=False)
    (out/'PREPARED').touch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings',type=Path,required=True)
    parser.add_argument('--annotations',type=Path,required=True)
    args = parser.parse_args()
    prepare(args.recordings,args.annotations,boot.ROOT/'outputs/map_input_calibration_20261105/data/zenodo5501527')


if __name__=='__main__':
    main()
