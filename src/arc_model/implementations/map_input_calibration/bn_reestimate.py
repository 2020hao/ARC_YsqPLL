"""Training-only, layerwise population BatchNorm statistics in inference order.

The batch factory yields model arguments, never labels. Earlier BN layers use
their newly estimated inference statistics when estimating later layers.
All samples/positions carry equal weight; a small final batch is not an EMA step
with the same weight as a full batch. Affine and other learned weights are fixed.
"""
import torch
from torch import nn


class _Captured(Exception):
    pass


class ChannelMoments:
    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None
        self.batches = 0

    def add(self, x):
        value = x.detach().double()
        dims = (0,) + tuple(range(2, value.ndim))
        variance, mean = torch.var_mean(value, dim=dims, unbiased=False)
        count = value.numel() // value.shape[1]
        if self.count == 0:
            self.mean, self.m2 = mean, variance * count
        else:
            delta = mean - self.mean
            total = self.count + count
            self.m2 += variance * count + delta.square() * (self.count * count / total)
            self.mean += delta * (count / total)
        self.count += count
        self.batches += 1

    @property
    def variance(self):
        if not self.count:
            raise ValueError('No training activations')
        return (self.m2 / self.count).clamp_min(0)


@torch.inference_mode()
def reestimate_bn(model, batch_factory):
    """Mutate only BN buffers; return activation population metadata by layer."""
    model.eval()
    layers = {n: m for n, m in model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)}
    order = []
    hooks = [m.register_forward_pre_hook(lambda module, args, name=n: order.append(name)) for n, m in layers.items()]
    try:
        model(*next(iter(batch_factory())))
    finally:
        for hook in hooks:
            hook.remove()
    if len(order) != len(set(order)) or set(order) != set(layers):
        raise ValueError('Expected each BN exactly once per forward')
    result = []
    for name in order:
        layer = layers[name]
        if not layer.track_running_stats:
            raise ValueError('BN without running statistics is unsupported')
        moments = ChannelMoments()
        def capture(module, args):
            moments.add(args[0])
            raise _Captured()
        hook = layer.register_forward_pre_hook(capture)
        try:
            for args in batch_factory():
                try:
                    model(*args)
                except _Captured:
                    pass
        finally:
            hook.remove()
        layer.running_mean.copy_(moments.mean)
        layer.running_var.copy_(moments.variance)
        layer.num_batches_tracked.fill_(moments.batches)
        result.append(dict(layer=name, count_per_channel=moments.count, batches=moments.batches,
                           variance='population_M2_over_N', accumulator='float64_parallel_Welford'))
    return result
