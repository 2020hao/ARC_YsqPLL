"""Causal streaming extension of the frozen CycleStream; one shared classifier."""
from collections import deque
import time
import numpy as np
from scipy.signal import sosfilt,upfirdn
from arc_model.implementations.phase_event_snn.streaming import CycleStream
from .midband import MidbandFilter,profile,assemble


class MidbandCycleStream(CycleStream):
    def __init__(self,fs,origin_seconds=0.,arm='both'):
        super().__init__(fs,origin_seconds)
        self.energy=arm in ('mid','both');self.competition=arm in ('competition','both')
        self.midband=MidbandFilter(self.energy,self.competition)
        self.extra_cumulative=np.zeros(3);self.extra_prefixes=deque(maxlen=30)

    def _profile(self,a,b,kind,segment):
        original=super()._profile(a,b,kind,segment)
        at=(a/self.fs+self.origin_seconds)-self.origin_seconds
        bt=(b/self.fs+self.origin_seconds)-self.origin_seconds
        if kind=='native':
            q=[v for v in self.clock if v['phase_usable'] and v['reference_segment']==segment]
            tx=np.array([v['estimate_time_s'] for v in q]);px=np.array([v['phase_rad'] for v in q])
            grid=np.interp(np.linspace(np.interp(at,tx,px),np.interp(bt,tx,px),65),px,tx)
        else:grid=np.linspace(at,bt,65)
        edges=np.ceil(grid*self.fs).astype('int64');edges[0]=a;edges[-1]=b
        prefix=np.stack([np.concatenate([v['values'][c][:-1] for v in self.extra_prefixes]+[self.extra_prefixes[-1]['values'][c][-1:]]) for c in range(3)])
        extra=profile(prefix,self.extra_prefixes[0]['low_start'],edges,self.fs,self.energy,self.competition)
        return np.concatenate([original,extra],axis=0)

    def _append(self,kind,a,b,segment,epoch,available,status):
        signature=(segment,epoch) if kind=='native' else (-1,-1)
        if self.history_signature[kind]!=signature or self.last_end[kind]!=a:self.histories[kind].clear()
        p=self._profile(a,b,kind,segment);self.histories[kind].append(p);self.history_signature[kind]=signature;self.last_end[kind]=b
        x=np.zeros((9,6,64),dtype='float32');mask=np.zeros(9,dtype='float32');n=len(self.histories[kind]);x[-n:]=np.stack(self.histories[kind]);mask[-n:]=1
        prob=np.exp(np.clip(x[:,1:3]-x[:,1:3].max(-1,keepdims=True),-100,0)*np.log(10));prob/=np.maximum(prob.sum(-1,keepdims=True),1e-20)
        change=np.zeros_like(prob);change[1:]=(prob[1:]-prob[:-1])*(mask[1:]*mask[:-1])[:,None,None]
        original=np.concatenate([x[:,:3],change],axis=1)
        features=assemble(original[None],x[:,3:6][None],mask[None])[0]
        return dict(start_sample=a,stop_sample_exclusive=b,available_seconds=available,phase_status=status,coordinate=kind,segment=segment,epoch=epoch,profile=p,input=features,mask=mask)

    def update(self,raw):
        raw=np.asarray(raw,dtype=float)
        if len(raw)!=self.chunk_samples or not np.isfinite(raw).all():raise ValueError('One finite2ms native chunk required')
        tick=time.perf_counter();raw_start=self.raw_count;low_start=self.low_count
        joined=np.r_[self.fir_history,raw];offset=len(self.fir_history)//self.down
        low=upfirdn(self.fir,joined,down=self.down)[offset:offset+len(raw)//self.down]
        self.fir_history=joined[-len(self.fir_history):].copy()
        extra=self.midband.update(low)
        values_extra=np.stack([np.cumsum(np.r_[self.extra_cumulative[c],extra[c]]) for c in range(3)])
        self.extra_cumulative=values_extra[:,-1].copy()
        self.extra_prefixes.append(dict(low_start=low_start,values=values_extra/50000))
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
