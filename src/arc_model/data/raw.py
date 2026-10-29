"""Rebuild one source-verified current recording into paired model inputs."""
import hashlib
import numpy as np
import h5py
from scipy.signal import firwin,upfirdn
from arc_model.preprocessing.signals import condition
from arc_model.preprocessing.windows import windows
from arc_model.features.descriptors import extend

def one(record,data_root):
    source=data_root/record["file"]
    assert hashlib.sha256(source.read_bytes()).hexdigest()==record["sha256"],"Source file changed"
    with h5py.File(source) as h:raw=np.asarray(h["current"]).reshape(-1)
    low,env,phase,_=condition(raw)
    x0,x1,p0,p1,meta=windows(low,env,phase)
    wave=np.stack([x0,x1],axis=1);physical=np.stack([p0,p1],axis=1)
    assert len(wave)==record["n_windows"]
    fs=250000;dec=8
    sig=upfirdn(firwin(801,.45*fs,fs=2e6,window=("kaiser",7.5)),raw,down=dec)[:(len(raw)+dec-1)//dec]
    native=np.empty((len(meta),2,5000),dtype="float32")
    grid=np.arange(len(low));valid=np.isfinite(phase);xx=np.arange(len(sig))
    for j,m in enumerate(meta):
        anchor=int(m["fixed_start_sample"]);start=int(round(anchor*fs/50000))
        turn=np.ceil(phase[anchor]/(2*np.pi))*2*np.pi
        pos=np.interp(turn+2*np.pi*np.arange(5000)/5000,phase[valid],grid[valid])
        native[j,0]=sig[start:start+5000]
        native[j,1]=np.interp(pos*fs/50000,xx,sig)
    return wave,extend(wave,physical),native
