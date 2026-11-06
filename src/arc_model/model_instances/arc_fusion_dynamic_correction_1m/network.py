"""Fixed instance; local dependency closure preserves checkpoint keys and RNG."""
from ._source_dual_arc_fusion_model import Model as _Original

FIXED_KWARGS = {'extra': 0, 'd_model': 64, 'n_heads': 4, 'feedforward_dim': 128, 'dropout': 0.1, 'n_encoder_layers': 2, 'spectral_pool_stride': 2}

class Model(_Original):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key not in FIXED_KWARGS or value != FIXED_KWARGS[key]:
                raise ValueError("This instance fixes its architecture; use a different model ID")
        super().__init__(**FIXED_KWARGS)
