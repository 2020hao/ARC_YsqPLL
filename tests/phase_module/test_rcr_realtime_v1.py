import numpy as np
from arc_model.preprocessing.rcr_realtime_v1 import estimate,RealtimeRCR


def test_prefix_causality_and_chunk_independence():
    t=np.arange(2500)/5000;x=np.sin(2*np.pi*50*t+.4)+.15*np.sin(2*np.pi*150*t)
    a=estimate(x,chunk_samples=1);b=estimate(x,chunk_samples=137)
    np.testing.assert_allclose(a.phase_rad,b.phase_rad,rtol=0,atol=0,equal_nan=True)
    changed=x.copy();changed[1500:]+=2*np.sin(2*np.pi*73*t[1500:])
    c=estimate(changed,chunk_samples=91)
    np.testing.assert_allclose(a.phase_rad.iloc[:300],c.phase_rad.iloc[:300],rtol=0,atol=0,equal_nan=True)
    assert (a.latest_input_time_s<=a.estimate_time_s).all()
    assert a.memory_samples.max()<=426
    usable=a.phase_usable.to_numpy();assert (np.diff(a.phase_rad.to_numpy()[usable])>0).all()
    assert a[a.phase_usable].estimate_time_s.iloc[0]<.06


def test_no_signal_is_not_reported_as_observed_phase():
    result=estimate(np.zeros(2000))
    assert not result.observed_valid.any() and not result.phase_usable.any()


def test_online_diagnostics_do_not_accumulate():
    tracker=RealtimeRCR()
    rows=tracker.update(np.sin(2*np.pi*50*np.arange(2500)/5000))
    assert tracker.rows is None and len(rows)==500
    assert len(tracker.values)<=426 and len(tracker.templates)<=4
