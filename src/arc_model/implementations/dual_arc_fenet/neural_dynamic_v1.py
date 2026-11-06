"""Independent train-only residual encoders on top of this model's frozen B1.

Feature order is lag-major [lag(1,2,4,8), band(2), statistic(4)], then
four availability flags. The existing linear correction, calibration and gate
are preserved. No imports from another model implementation package.
"""
import numpy as np
import torch
from torch import nn
from scipy.special import expit
from .phase_dynamic_v2 import replay as linear_replay


class LinearResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(36, 1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, x, valid):
        return 2*torch.tanh(self.output(x).flatten()/2)


class DenseResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(36, 3), nn.SiLU())
        self.output = nn.Linear(3, 1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, x, valid):
        return 2*torch.tanh(self.output(self.hidden(x)).flatten()/2)


class LagSharedResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.band_log_scale = nn.Parameter(torch.zeros(2))
        self.encoder = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4), nn.SiLU())
        self.output = nn.Linear(36, 1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, x, valid):
        # Eight tokens: [two bands, four historical lags]; no global averaging.
        tokens = x[:, :32].reshape(-1, 4, 2, 4).permute(0, 2, 1, 3)
        tokens = tokens*self.band_log_scale.clamp(-2, 2).exp()[None, :, None, None]
        h = self.encoder(tokens)*valid[:, None, :, None]
        h = torch.cat([h.flatten(1), x[:, 32:36]], dim=1)
        return 2*torch.tanh(self.output(h).flatten()/2)


ARCHITECTURES = {'linear': LinearResidual, 'dense': DenseResidual, 'lag_shared': LagSharedResidual}
PARAMETERS = {'linear': 37, 'dense': 115, 'lag_shared': 115}


def head_state(net, feature_state, fit_indices, architecture, seed, losses):
    state = dict(mu=np.asarray(feature_state['mu']).copy(), sd=np.asarray(feature_state['sd']).copy(),
                 fit_indices=np.asarray(fit_indices).copy(), architecture=np.asarray(architecture),
                 seed=np.asarray(seed), loss=np.asarray(losses),
                 parameters=np.asarray(sum(p.numel() for p in net.parameters())))
    state.update({'weight::'+key: value.detach().cpu().numpy().copy() for key, value in net.state_dict().items()})
    return state


def restore(state, device='cpu'):
    net = ARCHITECTURES[str(state['architecture'])]().to(device)
    net.load_state_dict({key.removeprefix('weight::'): torch.as_tensor(value, device=device)
                        for key, value in state.items() if key.startswith('weight::')})
    return net.eval()


def predict(state, features, device='cpu'):
    net = restore(state, device)
    z = ((features-state['mu'])/state['sd']).astype('float32')
    valid = features[:, 32:36].astype('float32')
    result = []
    with torch.inference_mode():
        for start in range(0, len(z), 8192):
            result.append(net(torch.as_tensor(z[start:start+8192], device=device),
                              torch.as_tensor(valid[start:start+8192], device=device)).cpu().numpy())
    return np.concatenate(result)


def fit(features, labels, offset, indices, feature_state, architecture, seed, *,
        steps=160, learning_rate=.002, weight_decay=.01, residual_l2=.01, device='cuda'):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    net = ARCHITECTURES[architecture]().to(device)
    assert sum(p.numel() for p in net.parameters()) == PARAMETERS[architecture]
    z = ((features[indices]-feature_state['mu'])/feature_state['sd']).astype('float32')
    x = torch.as_tensor(z, device=device)
    valid = torch.as_tensor(features[indices, 32:36], dtype=torch.float32, device=device)
    y = torch.as_tensor(labels[indices], dtype=torch.float32, device=device)
    base = torch.as_tensor(offset[indices], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    losses = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        delta = net(x, valid)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(base+delta, y)+residual_l2*delta.square().mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        opt.step()
        if step in (0, steps-1): losses.append(float(loss.detach()))
    return head_state(net, feature_state, indices, architecture, seed, losses)


def base_terms(linear_state, features, raw, slope, intercept, strength, quantile, raw_off=False):
    delta, distance = linear_replay(linear_state, features)
    normal_threshold = float(linear_state['gate_threshold'])
    q = expit(-(distance-normal_threshold)/(.15*normal_threshold+1e-8))
    if raw_off or strength == 0:
        return np.asarray(raw).copy(), np.zeros_like(q), delta
    score = raw+strength*q*(slope*raw+intercept-raw+delta)
    return score, strength*q, delta


def corrected(base_margin, scale, delta, beta):
    # Required exact identity: turning the new module off recovers strong B1,
    # not the weaker naked backbone.
    if float(beta) == 0:
        return np.asarray(base_margin).copy()
    return np.asarray(base_margin)+float(beta)*np.asarray(scale)*np.asarray(delta)

