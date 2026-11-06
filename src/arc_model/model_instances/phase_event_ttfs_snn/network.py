"""Fixed instance; local dependency closure preserves checkpoint keys and RNG."""
from ._source_phase_event_snn_network import PhaseEventSNN as _Original

FIXED_KWARGS = {'encoding': 'ttfs', 'spiking': True}

class Model(_Original):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key not in FIXED_KWARGS or value != FIXED_KWARGS[key]:
                raise ValueError("This instance fixes its architecture; use a different model ID")
        super().__init__(**FIXED_KWARGS)
