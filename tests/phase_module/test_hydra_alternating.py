import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('validate_hydra_alternating',ROOT/'scripts/validate_hydra_alternating.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_alternation_uses_original_grid_before_missing_windows():
    t=pd.DataFrame(dict(record_id=[0,0,0,1,1],start_seconds=[.02,.04,.08,1.2,1.22]))
    ordinal,train=module.alternating_membership(t,{0:0.,1:1.2})
    np.testing.assert_array_equal(ordinal,[1,2,4,0,1])
    np.testing.assert_array_equal(train,[False,True,True,True,False])
