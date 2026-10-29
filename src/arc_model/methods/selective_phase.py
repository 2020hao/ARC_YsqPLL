"""Causal harmonic observations and position-selective cycle statistics.

Quality values are model-based proxies, not calibrated probabilities. This
module never accepts labels or fits classifier/preprocessing parameters.
"""
from __future__ import annotations
import hashlib
import numpy as np
from scipy.signal import sosfreqz
from .cycle_dynamics import dynamic_features

HARMONICS = (3, 5, 7, 9)
LAGS = (1, 2, 4, 8)


def harmonic_observation(y, phase, *, correction_sos=None, frequency=50.):
    """Huber harmonic fit on the supplied past-only samples."""
    y, phase = np.asarray(y, float), np.asarray(phase, float)
    valid = np.isfinite(y) & np.isfinite(phase)
    y, phase = y[valid], phase[valid]
    empty = dict(innovation=0., variance=np.deg2rad(30.)**2, valid=False,
                 amplitude=np.zeros(4), residual_phase=np.zeros(4), masks=np.zeros(4),
                 residual_ratio=1., signal_rms=0.)
    if len(y) < 40 or np.std(y) < 1e-12:
        return empty
    basis = np.column_stack([np.ones(len(y))] +
                            [fn(h * phase) for h in range(1, 10) for fn in (np.sin, np.cos)])
    weights = np.ones(len(y))
    beta = np.linalg.lstsq(basis, y, rcond=None)[0]
    for _ in range(4):
        residual = y - basis @ beta
        scale = max(1e-12, .01*np.std(y), 1.4826*np.median(np.abs(residual-np.median(residual))))
        weights = np.minimum(1., 1.5*scale/np.maximum(np.abs(residual), 1e-12))
        beta = np.linalg.lstsq(basis*np.sqrt(weights[:, None]), y*np.sqrt(weights), rcond=None)[0]
    residual = y - basis @ beta
    bread = np.linalg.pinv(basis.T @ (weights[:, None]*basis), rcond=1e-10)
    meat = basis.T @ (((weights*residual)**2)[:, None] * basis)
    covariance = bread @ meat @ bread * len(y)/max(1, len(y)-19)
    z = beta[1::2] + 1j*beta[2::2]
    amplitude = np.abs(z)
    floor = max(1e-10, np.sqrt(np.mean(y*y))*1e-6)
    def angle_gradient(h):
        g = np.zeros(19)
        a, b = beta[2*h-1:2*h+1]
        g[2*h-1:2*h+1] = np.array([-b, a])/max(a*a+b*b, floor*floor)
        return g
    g1 = angle_gradient(1)
    var1 = max(float(g1 @ covariance @ g1), np.deg2rad(.25)**2)
    fundamental_valid = amplitude[0] > floor and np.sqrt(var1) <= np.deg2rad(30)
    amp, residual_phase, masks = [], [], []
    response = np.ones(9, dtype=complex)
    if correction_sos is not None:
        _, response = sosfreqz(correction_sos, worN=np.arange(1, 10)*frequency, fs=50000)
    for h in HARMONICS:
        gh = angle_gradient(h) - h*g1
        phase_variance = max(float(gh @ covariance @ gh), 0.)
        visible = fundamental_valid and amplitude[h-1] > floor and phase_variance <= np.deg2rad(30)**2
        psi = np.angle(z[h-1]) - h*np.angle(z[0]) - (np.angle(response[h-1])-h*np.angle(response[0]))
        residual_phase.append(float(np.angle(np.exp(1j*psi))))
        ratio = amplitude[h-1]/max(amplitude[0], floor)*abs(response[0])/max(abs(response[h-1]), 1e-8)
        amp.append(float(np.log1p(min(ratio, 100.))))
        masks.append(float(visible))
    return dict(innovation=float(np.angle(z[0])), variance=var1, valid=bool(fundamental_valid),
                amplitude=np.array(amp), residual_phase=np.array(residual_phase), masks=np.array(masks),
                residual_ratio=float(np.sqrt(np.mean(residual**2))/max(np.std(y), floor)),
                signal_rms=float(np.sqrt(np.mean(y*y))))


def pair_sigma(a, b):
    if not a['valid'] or not b['valid']:
        return np.deg2rad(18.)
    delta = np.angle(np.exp(1j*(a['innovation']-b['innovation'])))
    return float(min(np.deg2rad(18.), np.sqrt(a['variance']+b['variance']+delta*delta)))


def circular_soft_bins(p, sigma):
    """Positive, mass-preserving fractional rotations under five-node quadrature."""
    nodes, weights = np.polynomial.hermite.hermgauss(5)
    result = np.zeros_like(p)
    for node, weight in zip(nodes, weights/np.sqrt(np.pi)):
        shift = np.sqrt(2.)*sigma*node/(2*np.pi)*p.shape[-1]
        lower = int(np.floor(shift)); fraction = shift-lower
        result += weight*((1-fraction)*np.roll(p, lower, axis=-1) + fraction*np.roll(p, lower+1, axis=-1))
    return result


def _stats(u, v):
    midpoint = .5*(u+v)
    return np.array([u@v/(np.linalg.norm(u)*np.linalg.norm(v)+1e-12),
                     np.mean(np.abs(u-v)),
                     .5*np.sum(u*np.log((u+1e-12)/(midpoint+1e-12)) + v*np.log((v+1e-12)/(midpoint+1e-12)))])


def selective_features(distributions, records, observations):
    original = dynamic_features(distributions, records)
    output = {name: original.copy() for name in ['C0', 'C1_all32', 'C2_select24', 'C3_fixed_soft', 'C4_pair_soft']}
    lookup = {(str(r['record_id']), int(r['cycle'])): i for i, r in enumerate(records)}
    rho_table = np.zeros((len(records), 4))
    for i, row in enumerate(records):
        for l, lag in enumerate(LAGS):
            j = lookup.get((str(row['record_id']), int(row['cycle'])-lag))
            if j is None:
                continue
            sigma = pair_sigma(observations[i], observations[j])
            rho = np.exp(-.5*(sigma/np.deg2rad(5.))**2); rho_table[i, l] = rho
            for b in range(2):
                k = l*8+b*4
                for name in ['C1_all32', 'C2_select24']:
                    output[name][i, k] = 1-rho*(1-original[i, k])
                    output[name][i, k+1:k+3] *= rho
                output['C1_all32'][i, k+3] *= rho
                for name, width in [('C3_fixed_soft', np.deg2rad(5.)), ('C4_pair_soft', sigma)]:
                    u = circular_soft_bins(distributions[i, b], width/np.sqrt(2.))
                    v = circular_soft_bins(distributions[j, b], width/np.sqrt(2.))
                    output[name][i, k:k+3] = _stats(u, v)
    return output, rho_table


def harmonic_slots(observation, record, cycle, *, randomized=False, remove=None):
    """Four amplitude, eight sin/cos, four visibility slots, constant dimension."""
    amp, angles, masks = (observation[k].copy() for k in ['amplitude', 'residual_phase', 'masks'])
    if randomized:
        for j, h in enumerate(HARMONICS):
            digest = hashlib.sha256(f'SH1|{record}|{cycle}|{h}'.encode()).digest()
            angles[j] += 2*np.pi*int.from_bytes(digest[:8], 'little')/2**64
    if remove is not None:
        j = HARMONICS.index(remove); amp[j] = 0.; masks[j] = 0.
    sc = np.stack([np.sin(angles), np.cos(angles)], axis=1)*masks[:, None]
    return np.r_[amp, sc.ravel(), masks]
