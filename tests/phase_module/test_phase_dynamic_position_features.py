import importlib.util
from pathlib import Path
import sys
import numpy as np

SCRIPTS=Path(__file__).resolve().parents[2]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('phase_position_features',SCRIPTS/'prepare_phase_dynamic_evidence_20261030.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_uniform_and_empty_energies_have_analytic_fractions():
    e=np.ones((2,2,20));e[1]=0
    p=module.distributions(e)
    np.testing.assert_allclose(module.phase4(p)[0],.1)
    np.testing.assert_array_equal(module.phase4(p)[1],0)
    np.testing.assert_allclose(module.phasefree4(p)[0],[.05,.05,.05,.05])


def test_position_control_preserves_global_concentration():
    e=np.zeros((1,2,20));e[0,0,0]=2;e[0,1,10]=4
    p=module.distributions(e);s=np.roll(p,5,axis=-1)
    np.testing.assert_array_equal(module.phase4(p),[[1,0,0,1]])
    assert not np.array_equal(module.phase4(p),module.phase4(s))
    np.testing.assert_array_equal(module.phasefree4(p),module.phasefree4(s))


def test_ratio_is_independent_of_current_units():
    e=np.random.default_rng(21).uniform(.1,2,(3,2,20))
    np.testing.assert_allclose(module.phase4(module.distributions(e)),
                               module.phase4(module.distributions(e*1e-30)),rtol=1e-6)
