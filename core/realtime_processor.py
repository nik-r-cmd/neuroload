"""
core/realtime_processor.py
───────────────────────────
PURPOSE
-------
Applies the EXACT SAME preprocessing steps as datapreprocess.py,
but to a single 2-second epoch arriving from the live streaming buffer.

This prevents "train-serve skew" — the most common hidden bug in BCI ML.
If inference preprocessing differs from training preprocessing, the feature
distributions shift and the model degrades silently.

HOW IT WORKS
------------
1. Receives a raw epoch: np.ndarray, shape (n_channels, n_timepoints)
2. Applies stateful IIR filters (high-pass + notch) using scipy.signal.sosfilt
   with a persistent filter state (zi) so filter memory is maintained across
   consecutive windows — unlike batch mode which re-fits per recording
3. Applies Common Average Reference (CAR)
4. Extracts 313 features via core.feature_extractor
5. Returns feature vector — ready for StandardScaler → XGBoost

WHY STATEFUL FILTERS
--------------------
In batch mode you filter the entire recording at once (no edge effects).
In real-time mode you process one window at a time. Without filter state
(zi), the filter restarts from zero at every window → transient artifacts
at every edge → features that do not match training distribution.
sosfilt_zi + maintaining zi between calls solves this.
"""

import numpy as np
import yaml
from pathlib import Path
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, sos2tf
from loguru import logger

from core.feature_extractor import extract_features

# ── Config ────────────────────────────────────────────────────────────────────
# Hardcoded -- avoids config key mismatch issues
SFREQ    = 128.0
HP_FREQ  = 0.5
NOTCH_HZ = [50]  # 100Hz removed: above Nyquist at 128Hz sfreq


def _make_highpass_sos(cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """Butterworth high-pass filter in second-order-sections format."""
    nyq = fs / 2.0
    return butter(order, cutoff / nyq, btype="high", output="sos")


def _make_notch_sos(notch_freq: float, fs: float, Q: float = 30.0) -> np.ndarray:
    """
    Notch filter at notch_freq Hz.
    Q controls width: higher Q = narrower notch.
    scipy iirnotch returns (b, a), we convert to SOS for numerical stability.
    """
    from scipy.signal import tf2sos
    b, a = iirnotch(notch_freq / (fs / 2), Q)
    return tf2sos(b, a)


class RealtimeProcessor:
    """
    Stateful EEG preprocessor for real-time inference.

    Create ONE instance per session. Call process(epoch) on each incoming
    2-second window. The filter states (zi_hp, zi_notch) persist between calls.

    Parameters
    ----------
    n_channels : int   — number of EEG channels in the stream
    sfreq      : float — sampling frequency (Hz); should match config
    """

    def __init__(self, n_channels: int = 64, sfreq: float = SFREQ):
        self.n_channels = n_channels
        self.sfreq      = sfreq

        # Build filter coefficients
        self.sos_hp     = _make_highpass_sos(HP_FREQ, sfreq)
        self.sos_notch  = [_make_notch_sos(f, sfreq) for f in NOTCH_HZ]

        # Initialise stateful filter memory (shape: n_sections × n_channels × 2)
        self.zi_hp     = np.zeros((self.sos_hp.shape[0],    n_channels, 2))
        self.zi_notch  = [
            np.zeros((s.shape[0], n_channels, 2)) for s in self.sos_notch
        ]

        logger.debug(
            f"RealtimeProcessor ready: {n_channels} ch @ {sfreq} Hz, "
            f"HP={HP_FREQ}Hz, notch={NOTCH_HZ}"
        )

    def _apply_filter_stateful(
        self,
        sos: np.ndarray,
        zi: np.ndarray,
        data: np.ndarray,
    ):
        """
        Apply SOS filter to data (n_channels, n_samples) with state.
        Updates zi in-place. Returns filtered data.
        """
        out = np.zeros_like(data)
        for ch in range(data.shape[0]):
            out[ch], zi[:, ch, :] = sosfilt(sos, data[ch], zi=zi[:, ch, :])
        return out

    def _car(self, data: np.ndarray) -> np.ndarray:
        """Common Average Reference: subtract channel mean at each sample."""
        return data - data.mean(axis=0, keepdims=True)

    def process(self, epoch: np.ndarray) -> np.ndarray:
        """
        Preprocess one epoch and return 313-dimensional feature vector.

        Parameters
        ----------
        epoch : np.ndarray, shape (n_channels, n_timepoints)
                Raw EEG in Volts (or µV — scaler handles magnitude).

        Returns
        -------
        features : np.ndarray, shape (n_features,)
                   Ready for StandardScaler → XGBoost.
        """
        if epoch.shape[0] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {epoch.shape[0]}"
            )

        # 1. High-pass (stateful)
        data = self._apply_filter_stateful(self.sos_hp, self.zi_hp, epoch)

        # 2. Notch filters (stateful, one per frequency)
        for sos, zi in zip(self.sos_notch, self.zi_notch):
            data = self._apply_filter_stateful(sos, zi, data)

        # 3. Common Average Reference
        data = self._car(data)

        # 4. Feature extraction (Welch PSD + Hjorth)
        features = extract_features(data, sfreq=self.sfreq)
        return features

    def reset(self) -> None:
        """Reset filter state. Call between sessions."""
        self.zi_hp    = np.zeros_like(self.zi_hp)
        self.zi_notch = [np.zeros_like(zi) for zi in self.zi_notch]
        logger.debug("RealtimeProcessor: filter state reset")