"""Audited 2 MHz acquisition / 50 kHz detector / 5 kHz PLL coefficients."""

from scipy.signal import firwin, butter

FS_RAW = 2_000_000
FS = 50_000
FS_PLL = 5_000
FIR = firwin(801, 20_000, fs=FS_RAW, window=("kaiser", 7.5))
HF_SOS = [
    butter(3, b, fs=FS_RAW, btype="bandpass", output="sos")
    for b in [(10_000, 100_000), (100_000, 400_000)]
]
ENV_SOS = butter(2, 8_000, fs=FS_RAW, output="sos")
