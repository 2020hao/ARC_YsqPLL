"""Causal 1--10kHz energy and tiny fixed competing-kernel occupancy.

Input is the existing causal 50kHz anti-aliased current, not the5kHz PLL signal.
Two groups have two5tap zero-mean/L1-normalized kernels each, dilations1/4.
Each group exports the fraction of samples won by kernel0 in each phase bin.
All20 fixed coefficients are counted separately from learned model parameters.
"""
import numpy as np
from scipy.signal import butter,sosfilt,lfilter


def kernels():
    w=np.random.RandomState(1729).normal(size=(2,2,5))
    w-=w.mean(-1,keepdims=True)
    return w/np.abs(w).sum(-1,keepdims=True)


class MidbandFilter:
    def __init__(self,energy=True,competition=True):
        self.energy=energy;self.competition=competition
        self.sos=butter(3,(1000,10000),fs=50000,btype='bandpass',output='sos')
        self.sos_state=np.zeros((len(self.sos),2));self.weights=kernels();self.taps=[];self.states=[]
        for group,dilation in enumerate((1,4)):
            b=np.zeros((2,4*dilation+1));b[:,::dilation]=self.weights[group]
            self.taps.append(b);self.states.append(np.zeros((2,b.shape[1]-1)))

    def update(self,low):
        low=np.asarray(low,dtype=np.float64)
        mid,self.sos_state=sosfilt(self.sos,low,zi=self.sos_state)
        out=np.zeros((3,len(low)),dtype=np.float64)
        if self.energy:out[0]=mid*mid
        if self.competition:
            for g in range(2):
                responses=[]
                for k in range(2):
                    v,self.states[g][k]=lfilter(self.taps[g][k],[1.],mid,zi=self.states[g][k]);responses.append(v)
                out[g+1]=(responses[0]>=responses[1]).astype(np.float64)
        return out


def profile(prefix,low_start,edges,native_fs,energy=True,competition=True):
    """Interpolate prefix integrals at the exact existing native bin edges."""
    times=(low_start+np.arange(prefix.shape[1]))/50000
    edge_times=np.asarray(edges,dtype=np.float64)/native_fs;dt=np.diff(edge_times)
    if (dt<=0).any() or edge_times[0]<times[0]-1e-12 or edge_times[-1]>times[-1]+1e-12:
        raise ValueError('Unavailable or nonpositive phase-bin support')
    value=np.stack([np.diff(np.interp(edge_times,times,row))/dt for row in prefix])
    value[0]=np.log10(np.maximum(value[0],1e-20)) if energy else -20.
    value[1:]=np.clip(value[1:],0,1) if competition else 0.
    return value.astype('float32')


def assemble(base5,extra3,mask):
    """Keep the original5 numeric channels exact, append MID,deltaMID,counts2."""
    x=np.asarray(base5,dtype='float32');extra=np.asarray(extra3,dtype='float32');m=np.asarray(mask)
    value=extra[:,:,0:1];p=10.**np.clip(value-value.max(-1,keepdims=True),-100,0)
    p/=np.maximum(p.sum(-1,keepdims=True),1e-20)
    change=np.zeros_like(p);change[:,1:]=(p[:,1:]-p[:,:-1])*(m[:,1:]*m[:,:-1])[:,:,None,None]
    return np.concatenate([x[:,:,:5],value,change,extra[:,:,1:]],axis=2).astype('float32')
