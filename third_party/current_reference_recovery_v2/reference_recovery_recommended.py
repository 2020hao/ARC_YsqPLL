"""Selected research candidate: lifecycle repair ONLY.
The dynamic extrapolation variant failed a synthetic morphology counterexample.
This wrapper deliberately cannot enable it. Not a hardware/deployment validation.
"""
from reference_recovery_v2 import estimate as _estimate
from reference_recovery_v2 import complete_cycles


def estimate(time_s, current, spec):
    """Return (all native-time slot diagnostics, initialization/events).
    Keep reference_epoch/reference_segment. Do not bridge invalid slots or reset epochs.
    phase_valid describes this estimator's evidence checks, not verified physical truth.
    """
    return _estimate(time_s,current,spec,dynamic=False)
