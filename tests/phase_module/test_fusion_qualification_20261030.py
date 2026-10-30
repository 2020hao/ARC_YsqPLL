"""Scientific contracts for the separate Fusion qualification study."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('fusion_qualification',ROOT/'scripts/qualify_fusion_current_20261030.py')
# The script's local bootstrap is resolved exactly as for its CLI.
import sys
sys.path.insert(0,str(ROOT/'scripts'))
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


def test_group_class_weights_ignore_group_size():
 y=np.array([0,1,0,0,0,1,1]);g=np.array(['a','a','b','b','b','b','b'])
 w=study.group_weights(y,g)
 assert np.isclose(w.sum(),1.)
 for group in np.unique(g):
  for label in [0,1]:assert np.isclose(w[(g==group)&(y==label)].sum(),.25)


def test_calibration_is_monotone_and_handles_large_logits():
 margin=np.array([-100,-3,-1,1,3,100.]);y=np.array([0,0,0,1,1,1]);g=np.array(['a']*6)
 fit=study.fit_calibration(margin,y,g)
 assert fit['slope']>0 and np.isfinite(fit['objective'])
 assert (np.diff(fit['slope']*margin+fit['intercept'])>0).all()


def test_gpu_zero_is_rejected_before_cuda_allocation(monkeypatch):
 monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
 with pytest.raises(ValueError,match='allowed'):study.setup_gpu()
