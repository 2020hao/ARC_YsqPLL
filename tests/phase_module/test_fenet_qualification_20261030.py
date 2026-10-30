"""Contracts for the separately versioned FENet numerical frontend."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pytest
from scipy.signal import firwin,freqz

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('fs,taps',[(1000000,401),(2000000,801)])
def test_filter_preserves_original_sensitive_band_and_anti_aliases(fs,taps):
    h=firwin(taps,118500,fs=fs,window=('kaiser',7.5))
    f=np.linspace(0,fs/2,100001);_,response=freqz(h,worN=f,fs=fs)
    db=20*np.log10(np.maximum(abs(response),1e-15))
    assert abs(db[f<=112000]).max()<.1
    assert db[f>=125000].max()<-60
    assert (taps-1)/2/fs==pytest.approx(.0002)


def test_qualification_does_not_mutate_or_import_research_enhancement():
    source=(ROOT/'scripts/qualify_fenet_20261030.py').read_text()
    assert 'baseline_arc_fenet.network import Model' in source
    assert 'dynamic_correction' not in source and 'reference_recovery' not in source
    assert "torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-5)" in source
    assert 'range(20)' in source and 'order.split(128)' in source


def test_frozen_release_keeps_original_numerical_budget():
    import yaml
    cfg=yaml.safe_load((ROOT/'configs/models.yaml').read_text())['ArcFENet_Baseline']
    assert cfg['model_training_param']['epochs']==20
    source=(ROOT/'src/arc_model/implementations/baseline_arc_fenet/mbwvd.py').read_text()
    assert 'max_iter=500' in source
    seal=json.loads((ROOT/'releases/baselines_v1_20261030/SEALED.json').read_text())
    assert seal['active_model_config']=='configs/models.yaml'
