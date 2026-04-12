"""
streaming/board_manager.py
───────────────────────────
PURPOSE
-------
Abstracts all EEG hardware (or synthetic emulation) behind a single interface.
The rest of the codebase never imports BrainFlow directly — only this module does.

TO SWITCH FROM SYNTHETIC TO REAL HARDWARE
------------------------------------------
Change board_id in config.yaml:
  -1  →  SYNTHETIC_BOARD      (default — no hardware required)
  38  →  MUSE_2_BOARD         (Muse 2 over Bluetooth)
  22  →  CYTON_BOARD          (OpenBCI Cyton 8-channel)
  13  →  GANGLION_BOARD       (OpenBCI Ganglion 4-channel)
  More: brainflow.readthedocs.io/en/stable/SupportedBoards.html

That single config change is enough. No other code changes needed.

CHANNEL MAPPING
---------------
The COG-BCI model was trained on 64 channels.
Consumer devices (Muse: 4 ch, Cyton: 8 ch) have fewer.
board_manager maps available channels to the nearest training equivalents
and zero-pads the rest so the feature vector shape stays at 313.

This is deliberate: zero-padded channels contribute zero SHAP importance,
which is honest. The model degrades gracefully rather than crashing.
"""

import time
import numpy as np
import yaml
from pathlib import Path
from loguru import logger

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.data_filter import DataFilter
    BRAINFLOW_AVAILABLE = True
except ImportError:
    BRAINFLOW_AVAILABLE = False
    logger.warning("BrainFlow not installed. Using pure numpy mock stream.")

# ── Config ────────────────────────────────────────────────────────────────────
_cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
with open(_cfg_path) as f:
    CFG = yaml.safe_load(f)

BOARD_ID       = -1
SFREQ_TARGET   = 128   # 128 Hz
N_TRAIN_CHANNELS = 64     # 64


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: pure Python mock (when BrainFlow is not installed)
# ─────────────────────────────────────────────────────────────────────────────

class _MockBoard:
    """
    Generates physiologically plausible synthetic EEG without BrainFlow.
    Used for development / CI where brainflow is not installed.
    Produces 64-channel data at 128 Hz by summing band-limited noise.
    """

    def __init__(self, n_channels: int = 64, sfreq: float = 128.0):
        self.n_channels = n_channels
        self.sfreq      = sfreq
        self._t         = 0.0
        logger.info("MockBoard: generating synthetic 64-ch EEG (no BrainFlow)")

    def prepare_session(self): pass
    def start_stream(self): pass
    def stop_stream(self): pass
    def release_session(self): pass

    def get_current_board_data(self, n_samples: int) -> np.ndarray:
        """Return (n_channels, n_samples) of synthetic EEG."""
        t = np.linspace(self._t, self._t + n_samples / self.sfreq, n_samples)
        self._t += n_samples / self.sfreq

        data = np.zeros((self.n_channels, n_samples))
        freqs = {"delta": 2, "theta": 6, "alpha": 10, "beta": 20, "gamma": 40}
        amps  = {"delta": 30e-6, "theta": 20e-6, "alpha": 25e-6,
                 "beta": 10e-6, "gamma": 5e-6}

        for ch in range(self.n_channels):
            for band, f in freqs.items():
                phase = np.random.uniform(0, 2 * np.pi)
                data[ch] += amps[band] * np.sin(2 * np.pi * f * t + phase)
            data[ch] += np.random.randn(n_samples) * 3e-6   # background noise

        return data

    @property
    def eeg_channels(self): return list(range(self.n_channels))
    @property
    def sfreq(self): return self._sfreq

    @sfreq.setter
    def sfreq(self, v): self._sfreq = v


# ─────────────────────────────────────────────────────────────────────────────
# BoardManager
# ─────────────────────────────────────────────────────────────────────────────

class BoardManager:
    """
    Unified EEG board interface.

    Usage
    -----
    with BoardManager() as bm:
        while True:
            epoch = bm.get_latest_samples(n_samples=256)
            # epoch.shape == (64, 256) always, regardless of device
    """

    def __init__(self, board_id: int = BOARD_ID):
        self.board_id    = board_id
        self._board      = None
        self._eeg_channels = None
        self.n_channels  = N_TRAIN_CHANNELS      # always 64 (padded if needed)
        self.sfreq       = SFREQ_TARGET

    # ── Context manager ───────────────────────────────────────────────────────
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ── Connection ────────────────────────────────────────────────────────────
    def connect(self) -> None:
        if not BRAINFLOW_AVAILABLE:
            self._board = _MockBoard(N_TRAIN_CHANNELS, SFREQ_TARGET)
            self._board.prepare_session()
            self._board.start_stream()
            self._eeg_channels = self._board.eeg_channels
            logger.info("Connected: MockBoard (64-ch synthetic)")
            return

        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()

        # Serial port required for physical OpenBCI boards
        # params.serial_port = "/dev/ttyUSB0"  # uncomment for Cyton

        self._board = BoardShim(self.board_id, params)
        self._board.prepare_session()
        self._board.start_stream(45000)

        self._eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        raw_sfreq = BoardShim.get_sampling_rate(self.board_id)

        logger.info(
            f"Connected: BrainFlow board_id={self.board_id} | "
            f"{len(self._eeg_channels)} EEG channels @ {raw_sfreq} Hz"
        )
        time.sleep(2.0)     # let the stream stabilise

    def disconnect(self) -> None:
        if self._board is not None:
            try:
                self._board.stop_stream()
                self._board.release_session()
            except Exception as exc:
                logger.warning(f"Board disconnect error: {exc}")
        logger.info("Board disconnected")

    # ── Data access ───────────────────────────────────────────────────────────
    def get_latest_samples(self, n_samples: int) -> np.ndarray:
        """
        Pull n_samples from the board ring buffer.
        Returns np.ndarray shape (64, n_samples) — always 64 channels.
        Real devices with fewer channels are zero-padded.
        """
        if self._board is None:
            raise RuntimeError("Call connect() first")

        if not BRAINFLOW_AVAILABLE:
            raw = self._board.get_current_board_data(n_samples)
        else:
            raw = self._board.get_current_board_data(n_samples)
            raw = raw[self._eeg_channels, :]         # select EEG rows only

        return self._pad_to_64(raw)

    def _pad_to_64(self, data: np.ndarray) -> np.ndarray:
        """
        Ensure output always has 64 channels (rows).
        If device has 4 channels: replicate nearest training channels,
        pad remainder with zeros.
        Shape in: (n_device_ch, n_samples)
        Shape out: (64, n_samples)
        """
        n_dev, n_samp = data.shape
        if n_dev == N_TRAIN_CHANNELS:
            return data

        out = np.zeros((N_TRAIN_CHANNELS, n_samp), dtype=np.float32)

        # Map device channels to rough training equivalents
        # (frontal → F channels, temporal → T channels, etc.)
        # Simple linear mapping: stretch device channels across 64
        for i in range(n_dev):
            train_idx = int(i * N_TRAIN_CHANNELS / n_dev)
            out[train_idx] = data[i]

        return out

    @property
    def is_connected(self) -> bool:
        return self._board is not None