import numpy as np
from scipy.signal import sosfilt,butter
from arc_model.implementations.cycle_phase_snn.streaming import CycleStream
from arc_model.preprocessing.signals import condition
from arc_model.preprocessing.rcr_cycle_v2 import estimate


def test_native_stream_matches_causal_reference_and_bin_integrals():
    for fs in (1000000,2000000):
        t=np.arange(int(.18*fs))/fs
        raw=np.sin(2*np.pi*50*t+.3)+.08*np.sin(2*np.pi*150*t)+.02*(1+.5*np.sin(2*np.pi*50*t))*np.sin(2*np.pi*33000*t)
        origin=-.123456789 if fs==1000000 else 0.
        stream=CycleStream(fs,origin_seconds=origin);events=[];clocks=[]
        for a in range(0,len(raw),stream.chunk_samples):
            e,d=stream.update(raw[a:a+stream.chunk_samples]);events.extend(e);clocks.append(d['phase_row'])
            assert d['buffered_raw_seconds']<=.0600001
        low,_,_,_=condition(raw,bands=(),sample_rate_hz=fs)
        reference=sosfilt(butter(2,800,fs=50000,output='sos'),low)[::10]
        offline=estimate(reference,chunk_samples=10)
        np.testing.assert_allclose([q['phase_rad'] for q in clocks],offline.phase_rad,rtol=0,atol=1e-10,equal_nan=True)
        bands=[sosfilt(butter(3,b,fs=fs,btype='bandpass',output='sos'),raw) for b in ((10000,100000),(100000,400000))]
        shape=sosfilt(butter(4,1000,fs=50000,output='sos'),low);sp=np.r_[0.,np.cumsum(shape)]/50000
        assert sum(e['coordinate']=='native' for e in events)>=4
        for event in events:
            assert event['input'].dtype==np.float32
            a,b=event['start_sample'],event['stop_sample_exclusive']
            at=(a/fs+origin)-origin;bt=(b/fs+origin)-origin
            q=offline[offline.estimate_time_s<=event['available_seconds']]
            if event['coordinate']=='native':
                q=q[q.phase_usable & (q.reference_segment==event['segment'])]
                tx=q.estimate_time_s.to_numpy();px=q.phase_rad.to_numpy()
                grid=np.interp(np.linspace(np.interp(at,tx,px),np.interp(bt,tx,px),65),px,tx)
            else:grid=np.linspace(at,bt,65)
            edges=np.ceil(grid*fs).astype(int);edges[0]=a;edges[-1]=b;dt=np.diff(edges/fs)
            expected=np.empty((3,64))
            expected[0]=np.diff(np.interp(edges/fs,np.arange(len(sp))/50000,sp))/dt
            for k,band in enumerate(bands,1):
                prefix=np.r_[0.,np.cumsum(band*band)]/fs
                expected[k]=np.log10(np.maximum(np.diff(prefix[edges])/dt,1e-20))
            np.testing.assert_allclose(event['profile'],expected,rtol=0,atol=2e-5)
