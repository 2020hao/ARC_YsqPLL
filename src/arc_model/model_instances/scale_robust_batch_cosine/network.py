"""Fixed instance; local dependency closure preserves checkpoint keys and RNG."""
from ._source_map_input_calibration_scale_robust import ScaleRobustPhaseMap as _Original

FIXED_KWARGS = {'normalization': 'batch', 'head': 'cosine'}

class Model(_Original):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key not in FIXED_KWARGS or value != FIXED_KWARGS[key]:
                raise ValueError("This instance fixes its architecture; use a different model ID")
        super().__init__(**FIXED_KWARGS)
