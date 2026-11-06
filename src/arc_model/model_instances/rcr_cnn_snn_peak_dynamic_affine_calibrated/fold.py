"""Fold score calibration into the existing final Linear without new weights."""

import copy

import numpy as np

from .calibration import _transform_parameters


def fold_to_readout(net, scale, bias, *, inplace=False):
    """Return a copied model by default; ``inplace=True`` explicitly mutates it.

    For the existing one-output ``net.readout`` Linear, apply W'=scale*W and
    b'=scale*b+bias. No extra classification layer or parameter is introduced.
    An unbiased Linear supports a zero calibration offset only; a nonzero
    offset is rejected instead of silently adding a learnable parameter.
    """
    import torch
    from torch import nn

    scale, bias = _transform_parameters({"scale": scale, "bias": bias})
    if not isinstance(inplace, bool):
        raise TypeError("inplace must be an explicit boolean")
    readout = getattr(net, "readout", None)
    if not isinstance(readout, nn.Linear) or readout.out_features != 1:
        raise ValueError("net.readout must be a final single-output nn.Linear")
    if readout.bias is None and bias != 0.0:
        raise ValueError("A nonzero offset cannot be folded into an unbiased readout")
    if not np.isfinite(float(scale)) or not np.isfinite(float(bias)):
        raise ValueError("Calibration parameters must be finite")
    result = net if inplace else copy.deepcopy(net)
    with torch.no_grad():
        result.readout.weight.mul_(scale)
        if result.readout.bias is not None:
            result.readout.bias.mul_(scale).add_(bias)
    return result
