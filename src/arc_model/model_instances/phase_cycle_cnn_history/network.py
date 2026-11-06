"""Fixed instance; local dependency closure preserves checkpoint keys and RNG."""
from ._source_sp_cycle_net_network import SPCycleNet as _Original

FIXED_KWARGS = {'use_history': True}

class Model(_Original):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key not in FIXED_KWARGS or value != FIXED_KWARGS[key]:
                raise ValueError("This instance fixes its architecture; use a different model ID")
        super().__init__(**FIXED_KWARGS)
