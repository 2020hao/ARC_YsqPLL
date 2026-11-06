"""Performance-study adapter; preserve the earlier CPU-validation candidate."""
from ._source_map_input_calibration_detail_network import DetailPhaseMap


ARMS = ("base", "burst", "allocation", "both", "capacity")


class DetailPerformanceMap(DetailPhaseMap):
    def __init__(self, arm="both"):
        if arm not in ARMS:
            raise ValueError("Unknown detail ablation")
        super().__init__("both" if arm == "capacity" else arm)
        if arm == "capacity":
            # Nine-channel first layer with exactly the original information.
            # Duplicate the two original HF power channels; no new descriptor.
            self.active = (0, 1, 2, 3, 4, 1, 2, 1, 2)
            self.arm = arm
