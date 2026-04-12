"""
streaming/buffer.py
────────────────────
PURPOSE
-------
Maintains a rolling circular buffer of incoming EEG samples and yields
2-second windows with 1-second overlap — exactly matching the training
epoch specification in config.yaml.

WHY THIS MATTERS
----------------
During training, MNE created epochs with:
  - duration:  2.0 seconds  (256 samples at 128 Hz)
  - overlap:   1.0 second   (new epoch every 128 samples)

At inference, the model expects feature vectors computed from windows
of the same size and overlap. The buffer enforces this contract precisely.

HOW IT WORKS
------------
                       incoming samples (128 per call)
                              ↓
  ┌─────────────────────────────────────────────────┐
  │          circular buffer (1280 samples)         │  ← 10s at 128Hz
  └─────────────────────────────────────────────────┘
           ↓                        ↓
    every 128 new samples → emit window [ptr-256 : ptr]
    (256 samples = 2 seconds @ 128 Hz)

THREAD SAFETY
-------------
EegBuffer is NOT thread-safe by design — it's called from a single
inference loop in inference_engine.py. If you later add a background
acquisition thread, wrap with threading.Lock.
"""

import numpy as np
import yaml
from pathlib import Path

_cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
with open(_cfg_path) as f:
    CFG = yaml.safe_load(f)

SFREQ         = 128       # 128
EPOCH_DUR_S   = 2.0     # 2.0
EPOCH_OVL_S   = 1.0      # 1.0
EPOCH_SAMPLES = int(EPOCH_DUR_S * SFREQ)           # 256
STEP_SAMPLES  = int((EPOCH_DUR_S - EPOCH_OVL_S) * SFREQ)   # 128
BUF_SECONDS   = 10  # 10
BUF_SAMPLES   = int(BUF_SECONDS * SFREQ)           # 1280


class EegBuffer:
    """
    Circular ring buffer for EEG data.

    Parameters
    ----------
    n_channels : int — number of EEG channels (64)

    Usage
    -----
    buf = EegBuffer(n_channels=64)
    while streaming:
        chunk = board.get_latest_samples(128)    # shape: (64, 128)
        window = buf.push(chunk)
        if window is not None:
            features = processor.process(window) # shape: (313,)
    """

    def __init__(self, n_channels: int = 64):
        self.n_channels    = n_channels
        self._buf          = np.zeros((n_channels, BUF_SAMPLES), dtype=np.float32)
        self._write_ptr    = 0          # where next samples go
        self._samples_since_last_epoch = 0
        self._total_samples = 0
        self._warmup_samples = EPOCH_SAMPLES    # wait for first full window

    def push(self, chunk: np.ndarray) -> np.ndarray | None:
        """
        Push a chunk of new samples into the buffer.

        Parameters
        ----------
        chunk : np.ndarray, shape (n_channels, n_new_samples)

        Returns
        -------
        epoch : np.ndarray, shape (n_channels, EPOCH_SAMPLES) if a new window
                is ready, otherwise None.
        Only one window is returned per call (step = 1 second).
        """
        n_new = chunk.shape[1]

        # Write into circular buffer (wrap around if needed)
        for i in range(n_new):
            self._buf[:, self._write_ptr] = chunk[:, i]
            self._write_ptr = (self._write_ptr + 1) % BUF_SAMPLES

        self._total_samples        += n_new
        self._samples_since_last_epoch += n_new

        # Not enough data yet for first full window
        if self._total_samples < self._warmup_samples:
            return None

        # Emit a window every STEP_SAMPLES new samples
        if self._samples_since_last_epoch >= STEP_SAMPLES:
            self._samples_since_last_epoch = 0
            return self._extract_latest_window()

        return None

    def _extract_latest_window(self) -> np.ndarray:
        """
        Extract the most recent EPOCH_SAMPLES samples from the circular buffer.
        Handles wrap-around correctly.
        """
        end   = self._write_ptr
        start = (end - EPOCH_SAMPLES) % BUF_SAMPLES

        if start < end:
            return self._buf[:, start:end].copy()
        else:
            # Wrap-around: concatenate two slices
            part1 = self._buf[:, start:]
            part2 = self._buf[:, :end]
            return np.concatenate([part1, part2], axis=1)

    def get_rolling_raw(self, seconds: float = 5.0) -> np.ndarray:
        """
        Return the most recent `seconds` of raw data for waveform display.
        Returns np.ndarray shape (n_channels, n_samples).
        """
        n = min(int(seconds * SFREQ), BUF_SAMPLES, self._total_samples)
        end   = self._write_ptr
        start = (end - n) % BUF_SAMPLES

        if start < end:
            return self._buf[:, start:end].copy()
        else:
            return np.concatenate(
                [self._buf[:, start:], self._buf[:, :end]], axis=1
            )

    def reset(self) -> None:
        self._buf[:] = 0
        self._write_ptr = 0
        self._samples_since_last_epoch = 0
        self._total_samples = 0

    @property
    def is_warm(self) -> bool:
        """True once the buffer has enough data for the first window."""
        return self._total_samples >= self._warmup_samples

    @property
    def seconds_buffered(self) -> float:
        return min(self._total_samples, BUF_SAMPLES) / SFREQ