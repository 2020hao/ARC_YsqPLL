"""Analytic controls for cluster-level paired statistics, not per-cycle pseudo-N."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('locked_route_statistics', SCRIPTS/'summarize_locked_routes_20261030.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_thirteen_equal_positive_clusters_have_exact_sign_probability():
    result = module.paired_interval(np.full(13, .02))
    assert result['groups'] == 13
    assert abs(result['p'] - 2/8192) < 1e-15
    for name in ('delta_pp', 'lo_pp', 'hi_pp'):
        assert abs(result[name] - 2) < 1e-12


def test_identical_predictions_have_zero_effect_and_unit_p():
    result = module.paired_interval(np.zeros(13))
    assert result['p'] == 1
    assert result['delta_pp'] == result['lo_pp'] == result['hi_pp'] == 0
