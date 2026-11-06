"""Convex score calibration using only supplied training-fold OOF scores.

No model or dataset is loaded here. The runner owns disjoint inner-group OOF
generation and must never supply outer-test labels. Weights give every group
equal mass and each of its two classes equal mass, then normalize globally to
sum1. The regularizer is consequently independent of group sample counts or
the number of OOF scores: mean weighted BCE + .01*(bias²+(scale-1)²).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

REGULARIZATION = 0.01
BIAS_BOUNDS = (-4.0, 4.0)
SCALE_BOUNDS = (0.25, 4.0)


def _one_dimensional(values, name, *, numeric=False):
    values = np.asarray(values, dtype=np.float64 if numeric else None)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if numeric and not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite values")
    return values


def condition_class_weights(y, group):
    """Return sum1 weights with mass1/G per group and1/(2G) per class cell.

    Every calibration group must contain both classes. A missing class is a
    split/label-support problem and is reported explicitly instead of silently
    changing the declared condition/class-balanced objective.
    """
    y = _one_dimensional(y, "y", numeric=True)
    group = _one_dimensional(group, "group")
    if len(y) != len(group):
        raise ValueError("y and group must align row-for-row")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must contain only binary labels0/1")
    if group.dtype.kind in "fc" and not np.isfinite(group).all():
        raise ValueError("group identifiers must not be missing/nonfinite")
    if group.dtype.kind == "O" and any(
        value is None or (isinstance(value, float) and not np.isfinite(value))
        for value in group
    ):
        raise ValueError("group identifiers must not be missing/nonfinite")
    try:
        names, inverse = np.unique(group, return_inverse=True)
    except TypeError as error:
        raise ValueError("group identifiers must be mutually comparable") from error
    weights = np.zeros(len(y), dtype=np.float64)
    for index, name in enumerate(names):
        for label in (0, 1):
            ids = (inverse == index) & (y == label)
            count = int(ids.sum())
            if not count:
                raise ValueError(f"Calibration group {name!r} lacks class{label}")
            weights[ids] = 1.0 / (len(names) * 2 * count)
    # This final sum1 normalization fixes lambda relative to the global mean
    # objective, even when group/class counts are unequal.
    weights /= weights.sum()
    return weights


def _training_arrays(logits, y, group):
    logits = _one_dimensional(logits, "logits", numeric=True)
    y = _one_dimensional(y, "y", numeric=True)
    group = _one_dimensional(group, "group")
    if logits.shape != y.shape or y.shape != group.shape:
        raise ValueError("logits, y and group must align row-for-row")
    if np.abs(logits).max() > np.finfo(np.float64).max / SCALE_BOUNDS[1] - 4:
        raise ValueError("logits are too large for the bounded affine transform")
    weights = condition_class_weights(y, group)
    return logits, y, group, weights


def _objective_gradient(parameters, logits, y, weights, affine):
    scale, bias = (float(parameters[0]), float(parameters[1])) if affine else (1.0, float(parameters[0]))
    z = scale * logits + bias
    bce = np.logaddexp(0.0, z) - y * z
    penalty = REGULARIZATION * (bias * bias + (scale - 1) ** 2)
    objective = float(np.dot(weights, bce) + penalty)
    error = weights * (expit(z) - y)
    bias_gradient = float(error.sum() + 2 * REGULARIZATION * bias)
    if affine:
        gradient = np.array([
            np.dot(error, logits) + 2 * REGULARIZATION * (scale - 1),
            bias_gradient,
        ], dtype=np.float64)
    else:
        gradient = np.array([bias_gradient], dtype=np.float64)
    return objective, gradient


def _projected_gradient(parameters, gradient, bounds, tolerance=1e-8):
    projected = gradient.copy()
    for index, (lower, upper) in enumerate(bounds):
        if parameters[index] <= lower + tolerance and gradient[index] >= 0:
            projected[index] = 0.0
        if parameters[index] >= upper - tolerance and gradient[index] <= 0:
            projected[index] = 0.0
    return projected


def _fit(logits, y, group, *, affine):
    logits, y, group, weights = _training_arrays(logits, y, group)
    initial = np.array([1.0, 0.0] if affine else [0.0], dtype=np.float64)
    bounds = [SCALE_BOUNDS, BIAS_BOUNDS] if affine else [BIAS_BOUNDS]
    result = minimize(
        _objective_gradient,
        initial,
        args=(logits, y, weights, affine),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "maxls": 50, "gtol": 1e-10, "ftol": 1e-14},
    )
    objective, gradient = _objective_gradient(result.x, logits, y, weights, affine)
    identity_objective, _ = _objective_gradient(initial, logits, y, weights, affine)
    projected = _projected_gradient(result.x, gradient, bounds)
    violation = max(
        max(lower - value, value - upper, 0.0)
        for value, (lower, upper) in zip(result.x, bounds)
    )
    finite = bool(np.isfinite(objective) and np.isfinite(gradient).all() and np.isfinite(result.x).all())
    projected_norm = float(np.abs(projected).max())
    certificate_passed = (
        finite
        and violation <= 1e-9
        and projected_norm <= 1e-6
        and objective <= identity_objective + 1e-10
    )
    scale, bias = (float(result.x[0]), float(result.x[1])) if affine else (1.0, float(result.x[0]))
    z = scale * logits + bias
    nll = float(np.dot(weights, np.logaddexp(0.0, z) - y * z))
    names, inverse = np.unique(group, return_inverse=True)
    group_masses = np.bincount(inverse, weights=weights)
    return {
        "mode": "affine" if affine else "bias",
        "scale": scale,
        "bias": bias,
        "lambda": REGULARIZATION,
        "objective": objective,
        "weighted_BCE": nll,
        "regularization_penalty": objective - nll,
        "identity_objective": identity_objective,
        "objective_improvement": identity_objective - objective,
        "gradient_order": ["scale", "bias"] if affine else ["bias"],
        "gradient": gradient.tolist(),
        "projected_gradient": projected.tolist(),
        "success": bool(result.success and certificate_passed),
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "iterations": int(result.nit),
        "n": int(len(logits)),
        "groups": int(len(names)),
        "weight_normalization": "global_sum1_condition_and_class_equal",
        "certificate": {
            "passed": bool(certificate_passed),
            "finite": finite,
            "bound_violation": float(violation),
            "projected_gradient_inf_norm": projected_norm,
            "KKT_tolerance": 1e-6,
            "weight_sum": float(weights.sum()),
            "group_mass_min": float(group_masses.min()),
            "group_mass_max": float(group_masses.max()),
            "objective_not_above_identity": bool(objective <= identity_objective + 1e-10),
            "strong_convexity_regularizer_lower_bound": 2 * REGULARIZATION,
        },
    }


def fit_bias(logits, y, group):
    """Fit one bounded intercept using supplied training-inner-OOF scores."""
    return _fit(logits, y, group, affine=False)


def fit_affine(logits, y, group):
    """Fit positive bounded scale and intercept on training-inner-OOF scores."""
    return _fit(logits, y, group, affine=True)


def _transform_parameters(calibration):
    scale = float(calibration["scale"])
    bias = float(calibration["bias"])
    if not np.isfinite(scale) or not np.isfinite(bias):
        raise ValueError("Calibration parameters must be finite")
    if not SCALE_BOUNDS[0] <= scale <= SCALE_BOUNDS[1]:
        raise ValueError("Calibration scale must lie in[0.25,4]")
    if not BIAS_BOUNDS[0] <= bias <= BIAS_BOUNDS[1]:
        raise ValueError("Calibration bias must lie in[-4,4]")
    return scale, bias


def predict_logits(logits, calibration):
    """Apply fitted parameters to scores without accepting labels or groups."""
    logits = _one_dimensional(logits, "logits", numeric=True)
    scale, bias = _transform_parameters(calibration)
    result = scale * logits + bias
    if not np.isfinite(result).all():
        raise ValueError("Calibrated scores overflowed")
    return result


def predict_proba(logits, calibration):
    """Return a label-free sigmoid of the one calibrated classifier score."""
    return expit(predict_logits(logits, calibration))
