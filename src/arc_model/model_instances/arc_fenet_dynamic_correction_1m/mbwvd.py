import numpy as np
from scipy.signal import hilbert, resample


def vmd_high(x, fs=250000, alpha=2000, tol=1e-7, max_iter=500):
    """ADMM VMD (Dragomiretskiy/Zosso formulation), K=3, tau=0.

    Mirrored extension, positive-frequency modes, Gauss-Seidel updates;
    centres initialised exactly at 50/22000/101000 Hz per the arc paper.
    Alpha/tolerance/iteration cap are disclosed defaults absent from that paper.
    """
    n = len(x)
    scale = max(float(np.sqrt(np.mean(x * x))), 1e-12)
    x = x / scale
    f = np.concatenate([x[: n // 2][::-1], x, x[-n // 2 :][::-1]])
    spectrum = np.fft.rfft(f)
    freq = np.fft.rfftfreq(len(f))
    modes = np.zeros((3, len(spectrum)), dtype=complex)
    centers = np.array([50, 22000, 101000]) / fs
    for it in range(max_iter):
        old = modes.copy()
        total = modes.sum(axis=0)
        for k in range(3):
            residual = spectrum - (total - modes[k])
            new = residual / (1 + alpha * (freq - centers[k]) ** 2)
            total += new - modes[k]
            modes[k] = new
            power = np.abs(new) ** 2
            centers[k] = np.sum(freq * power) / max(power.sum(), 1e-30)
        change = np.sum(np.abs(modes - old) ** 2) / len(f)
        if change < tol:
            break
    # Exclude the LOWEST final-frequency mode, preserving two high modes.
    keep = np.argsort(centers)[1:]
    high = np.fft.irfft(modes[keep].sum(axis=0), n=len(f))[n // 2 : n // 2 + n] * scale
    return high, centers * fs, it + 1, change < tol


def mbwvd(x):
    high, centers, iterations, converged = vmd_high(np.asarray(x, float))
    # Half-sample analytic interpolation prevents the f -> 2f discrete-WVD alias.
    analytic = resample(hilbert(high), 2 * len(high))
    center = 2 * np.arange(len(high))
    lags = np.arange(-6, 7)  # 13 taps = 52 us, nearest odd length to 50 us.
    a = center[:, None] + lags[None]
    b = center[:, None] - lags[None]
    ok = (a >= 0) & (b >= 0) & (a < len(analytic)) & (b < len(analytic))
    product = (
        analytic[np.clip(a, 0, len(analytic) - 1)]
        * analytic[np.clip(b, 0, len(analytic) - 1)].conj()
    )
    product *= ok * np.hamming(13)
    lagbuf = np.zeros((len(high), 256), complex)
    lagbuf[:, lags % 256] = product
    tf = np.fft.fft(lagbuf, axis=1).real[:, :129] / 250000
    # Average all observations in each time bin rather than subselecting 100 times.
    tf = tf.reshape(100, 50, 129).mean(axis=1)
    out = np.stack([np.interp(np.linspace(0, 128, 100), np.arange(129), row) for row in tf])
    return out.astype(np.float32), centers, iterations, converged
