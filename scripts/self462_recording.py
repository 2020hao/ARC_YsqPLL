"""Read the reviewed NET9770_114 acquisition format without modifying the BIN."""
import hashlib
import struct

import numpy as np


def read_recording(path):
    with path.open('rb') as stream:
        header = stream.read(56)
    device, _, channels, rate, samples = struct.unpack('<32sQIIQ', header)
    if device.rstrip(b'\0') != b'NET9770_114' or channels != 8 or rate != 2_000_000:
        raise ValueError(f'Unexpected NET9770 device, channel count or sampling rate: {path}')
    if path.stat().st_size != 56 + samples * channels * 8:
        raise ValueError(f'Invalid BIN length: {path}')
    raw = np.memmap(path, dtype='<f8', mode='r', offset=56, shape=(samples, channels))[:, 0].copy()
    if not np.isfinite(raw).all():
        raise ValueError(f'Non-finite CH0 samples: {path}')
    return header, samples, raw, hashlib.sha256(memoryview(raw).cast('B')).hexdigest()
