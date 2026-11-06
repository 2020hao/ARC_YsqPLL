import json
import numpy as np
import pandas as pd
from arc_model.preprocessing.rcr_reference import _implementation
from arc_model.preprocessing.rcr_frequency_tracking_v1 import estimate


def test_disabled_exact_and_selective_weak_source_inactive():
    original,specpath=_implementation();spec=json.loads(specpath.read_text());spec['reference_fs_hz']=5000
    t=np.arange(5000)/5000.;x=.02*np.sin(2*np.pi*50*t)+.7*np.sin(2*np.pi*100*t)+.5*np.sin(2*np.pi*150*t+.2)
    a,ev=original(t,x,spec);b,bv=estimate(t,x,spec,'disabled')
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    assert json.dumps(ev,sort_keys=True,default=str)==json.dumps(bv,sort_keys=True,default=str)
    c,_=estimate(t,x,spec,'selective')
    assert not c.get('frequency_gain_accelerated',pd.Series(False,index=c.index)).fillna(False).any()
    assert c.phase_valid.any()
    assert (c.loc[c.phase_valid,'origin']=='morphology').all()
    for column in ['phase_rad','frequency_hz','phase_valid','reference_epoch','reference_segment']:
        np.testing.assert_equal(a[column].to_numpy(),c[column].to_numpy())
