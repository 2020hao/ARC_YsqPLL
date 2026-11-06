"""Bounded native-rate streaming frontend and direct per-cycle classifier.

Fixed2ms acquisition chunks. No labels, future samples or baseline scores.
Both native and nominal histories exist for availability fallback, but only one
selected input and one unified classifier produce a prediction.
"""
from collections import deque
import time
import numpy as np
from scipy.signal import butter,firwin,sosfilt,upfirdn
from arc_model.preprocessing.rcr_cycle_v2 import CycleRCR


class CycleStream:
    def __init__(self,fs,origin_seconds=0.):
        if fs not in (1000000,2000000):raise ValueError('Supported acquisition rates:1/2MHz')
        self.fs=fs;self.origin_seconds=float(origin_seconds);self.chunk_samples=fs//500;self.down=fs//50000
        self.fir=firwin(int(fs*.0004)+1,20000,fs=fs,window=('kaiser',7.5))
        self.fir_history=np.zeros(len(self.fir)-1)
        self.lp=butter(2,800,fs=50000,output='sos');self.lp_state=np.zeros((len(self.lp),2))
        self.shape_filter=butter(4,1000,fs=50000,output='sos');self.shape_state=np.zeros((len(self.shape_filter),2))
        self.bands=[butter(3,b,fs=fs,btype='bandpass',output='sos') for b in ((10000,100000),(100000,400000))]
        self.band_states=[np.zeros((len(s),2)) for s in self.bands]
        self.raw_count=0;self.low_count=0;self.cumulative=np.zeros(3)
        self.prefixes=deque(maxlen=30);self.clock=deque(maxlen=80);self.tracker=CycleRCR()
        self.last_clock=None;self.last_bound=None;self.latest_native=None
        self.histories={'native':deque(maxlen=9),'fallback':deque(maxlen=9)}
        self.last_end={'native':None,'fallback':None};self.history_signature={'native':None,'fallback':None}

    def _profile(self,a,b,kind,segment):
        a,b=int(a),int(b)
        # Match the acquisition time coordinate used to prepare training data.
        # Public DAT starts at a nonzero timestamp; round-tripping that origin
        # before ceil can change an exactly integral bin edge by one sample.
        at=(a/self.fs+self.origin_seconds)-self.origin_seconds
        bt=(b/self.fs+self.origin_seconds)-self.origin_seconds
        if kind=='native':
            q=[v for v in self.clock if v['phase_usable'] and v['reference_segment']==segment]
            tx=np.asarray([v['estimate_time_s'] for v in q]);px=np.asarray([v['phase_rad'] for v in q])
            grid=np.interp(np.linspace(np.interp(at,tx,px),np.interp(bt,tx,px),65),px,tx)
        else:grid=np.linspace(at,bt,65)
        edges=np.ceil(grid*self.fs).astype('int64');edges[0]=a;edges[-1]=b
        if (np.diff(edges)<=0).any():raise ValueError('Non-positive phase bin')
        raw_start=self.prefixes[0]['raw_start'];low_start=self.prefixes[0]['low_start']
        arrays=[np.concatenate([p['values'][i][:-1] for p in self.prefixes]+[self.prefixes[-1]['values'][i][-1:]]) for i in range(3)]
        assert edges[0]>=raw_start and edges[-1]<=self.raw_count
        dt=np.diff(edges/self.fs);result=np.empty((3,64),dtype='float32')
        shape_times=(low_start+np.arange(len(arrays[0])))/50000
        result[0]=np.diff(np.interp(edges/self.fs,shape_times,arrays[0]))/dt
        for channel in (1,2):result[channel]=np.log10(np.maximum(np.diff(arrays[channel][edges-raw_start])/dt,1e-20))
        return result

    def _append(self,kind,a,b,segment,epoch,available,status):
        signature=(segment,epoch) if kind=='native' else (-1,-1)
        if self.history_signature[kind]!=signature or self.last_end[kind]!=a:self.histories[kind].clear()
        profile=self._profile(a,b,kind,segment);self.histories[kind].append(profile)
        self.history_signature[kind]=signature;self.last_end[kind]=b
        x=np.zeros((9,3,64),dtype='float32');mask=np.zeros(9,dtype='float32');n=len(self.histories[kind]);x[-n:]=np.stack(self.histories[kind]);mask[-n:]=1
        prob=np.exp(np.clip(x[:,1:3]-x[:,1:3].max(-1,keepdims=True),-100,0)*np.log(10));prob/=np.maximum(prob.sum(-1,keepdims=True),1e-20)
        change=np.zeros_like(prob);change[1:]=(prob[1:]-prob[:-1])*(mask[1:]*mask[:-1])[:,None,None]
        return dict(start_sample=a,stop_sample_exclusive=b,available_seconds=available,phase_status=status,
                    coordinate=kind,segment=segment,epoch=epoch,profile=profile,
                    input=np.concatenate([x,change],axis=1).astype('float32'),mask=mask)

    def update(self,raw):
        raw=np.asarray(raw,dtype=float)
        if len(raw)!=self.chunk_samples or not np.isfinite(raw).all():raise ValueError('One finite2ms native chunk required')
        tick=time.perf_counter();raw_start=self.raw_count;low_start=self.low_count
        joined=np.r_[self.fir_history,raw];offset=len(self.fir_history)//self.down
        low=upfirdn(self.fir,joined,down=self.down)[offset:offset+len(raw)//self.down]
        self.fir_history=joined[-len(self.fir_history):].copy()
        lp,self.lp_state=sosfilt(self.lp,low,zi=self.lp_state)
        shape,self.shape_state=sosfilt(self.shape_filter,low,zi=self.shape_state)
        # Carry the unscaled accumulator into cumsum. Adding the old total
        # after a local cumsum changes floating-point addition order, which
        # measurably shifts weak HF powers after a strong transient.
        shape_sum=np.cumsum(np.r_[self.cumulative[0],shape])
        values=[shape_sum/50000];totals=[shape_sum[-1]]
        for i,sos in enumerate(self.bands):
            y,self.band_states[i]=sosfilt(sos,raw,zi=self.band_states[i])
            power_sum=np.cumsum(np.r_[self.cumulative[i+1],y*y])
            values.append(power_sum/self.fs);totals.append(power_sum[-1])
        self.cumulative=np.asarray(totals);self.raw_count+=len(raw);self.low_count+=len(low)
        self.prefixes.append(dict(raw_start=raw_start,low_start=low_start,values=values))
        rows=self.tracker.update(lp[::10]);assert len(rows)==1
        row=rows[0];self.clock.append(row);events=[];previous=self.last_clock
        if row['phase_usable'] and previous is not None and previous['phase_usable'] and row['reference_segment']==previous['reference_segment']:
            lo=int(np.floor(previous['phase_rad']/(2*np.pi)))+1;hi=int(np.floor(row['phase_rad']/(2*np.pi)))
            for turn in range(lo,hi+1):
                boundary=float(np.interp(turn*2*np.pi,[previous['phase_rad'],row['phase_rad']],[previous['estimate_time_s'],row['estimate_time_s']]))
                sample=int(np.ceil(boundary*self.fs))
                if self.last_bound is not None and self.last_bound[1]==row['reference_segment']:
                    a=self.last_bound[0]
                    if 1/55<=(sample-a)/self.fs<=1/45:
                        event=self._append('native',a,sample,row['reference_segment'],row['reference_epoch'],self.raw_count/self.fs,row['mode'])
                        self.latest_native=event;events.append(event)
                self.last_bound=(sample,row['reference_segment'])
        elif not row['phase_usable']:self.last_bound=None
        self.last_clock=row
        # Always advance nominal history to preserve its past support during lock.
        if self.raw_count%(self.fs//50)==0:
            fixed=self._append('fallback',self.raw_count-self.fs//50,self.raw_count,-1,-1,self.raw_count/self.fs,'NO_OBSERVED_PHASE_NOMINAL_COORDINATE')
            if not row['phase_usable'] or self.latest_native is None or self.latest_native['segment']!=row['reference_segment']:
                events.append(fixed)
        return events,dict(raw_time_seconds=self.raw_count/self.fs,phase_row=row,
                           frontend_seconds=time.perf_counter()-tick,buffered_raw_seconds=sum(len(p['values'][1])-1 for p in self.prefixes)/self.fs)


class DirectCyclePredictor:
    def __init__(self,network,mean,std,device='cpu'):
        import torch
        self.torch=torch;self.network=network.to(device).eval();self.device=device
        self.mean=np.asarray(mean,dtype='float32')[None,:,None];self.std=np.asarray(std,dtype='float32')[None,:,None]

    def predict(self,event):
        torch=self.torch
        x=torch.as_tensor((event['input']-self.mean)/self.std,dtype=torch.float32,device=self.device).unsqueeze(0)
        mask=torch.as_tensor(event['mask'],device=self.device).unsqueeze(0)
        with torch.inference_mode():value=float(self.network(x,mask).sigmoid().item())
        return dict(arc_probability=value,decision=int(value>=.5),coordinate=event['coordinate'],phase_status=event['phase_status'])
