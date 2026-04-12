"""
core/feature_extractor.py
--------------------------
Extracts 1280 features from one 2-second EEG epoch.
64 channels x 5 bands x 4 metrics = 1280 features.

Metrics per channel per band:
  abs  -- absolute band power (Welch PSD integrated over band)
  rel  -- relative power (band / total 1-45Hz)
  mob  -- Hjorth mobility
  cmp  -- Hjorth complexity

This matches EXACTLY what retrain_correct_labels.py produces,
so train and inference features are identical.
"""

import numpy as np
from scipy.signal import welch
# Hardcoded constants -- no config dependency to avoid key mismatches
SFREQ = 128.0
N_FFT = 256
N_OVL = 128
BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

# numpy >= 2.0 renamed trapz -> trapezoid
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


def _band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs <= hi)
    return float(_trapz(psd[mask], freqs[mask])) if mask.sum() > 0 else 0.0


def _hjorth(sig: np.ndarray):
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    var_x  = np.var(sig)  + 1e-12
    var_d1 = np.var(d1)   + 1e-12
    var_d2 = np.var(d2)   + 1e-12
    mob = np.sqrt(var_d1 / var_x)
    cmp = np.sqrt(var_d2 / var_d1) / mob
    return float(mob), float(cmp)


def extract_features(epoch: np.ndarray, sfreq: float = SFREQ) -> np.ndarray:
    """
    Parameters
    ----------
    epoch : np.ndarray, shape (n_channels, n_timepoints)
    sfreq : float, sampling frequency

    Returns
    -------
    np.ndarray, shape (n_channels * 5 * 4,)
    For 64 channels: 64 * 5 * 4 = 1280 features
    """
    n_channels = epoch.shape[0]
    features   = []

    for ch in range(n_channels):
        sig = epoch[ch]
        freqs, psd = welch(sig, fs=sfreq,
                           nperseg=min(N_FFT, len(sig)),
                           noverlap=N_OVL, window="hann")
        total = _band_power(freqs, psd, 1.0, 45.0) + 1e-12
        mob, cmp = _hjorth(sig)

        for lo, hi in BANDS.values():
            ab = _band_power(freqs, psd, lo, hi)
            features.extend([ab, ab / total, mob, cmp])

    return np.array(features, dtype=np.float32)


def feature_names(n_channels: int = 64) -> list:
    names = []
    for ch in range(n_channels):
        for band in BANDS:
            for m in ["abs", "rel", "mob", "cmp"]:
                names.append(f"{band}_ch{ch:02d}_{m}")
    return names