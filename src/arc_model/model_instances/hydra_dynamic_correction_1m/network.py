"""Fixed instance; local dependency closure preserves checkpoint keys and RNG."""
from ._source_dual_hydra_model import Model as _Original

FIXED_KWARGS = {'k': 8, 'g': 8, 'kernel_length': 5, 'stride': 1, 'transform_seed': 0}

class Model(_Original):
    def __init__(self, state=None, **kwargs):
        for key, value in kwargs.items():
            if key not in FIXED_KWARGS or value != FIXED_KWARGS[key]:
                raise ValueError("This instance fixes its architecture; use a different model ID")
        super().__init__(state=state, **FIXED_KWARGS)
