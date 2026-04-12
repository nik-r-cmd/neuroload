"""
core/datapreprocess.py
──────────────────────
PURPOSE
-------
Batch preprocessing pipeline for the COG-BCI dataset (Zenodo).
Reads all raw .set/.fdt EEG files → cleans → epochs → extracts features
→ saves features_sota.parquet.

RUN ONCE before training. Do NOT run at inference time.

USAGE
-----
  python -m core.datapreprocess \
      --data_dir  data/raw \
      --out_path  data/processed/features_sota.parquet

PIPELINE (mirrors MNE best practices for cognitive BCI)
---------------------------------------------------------
1.  Load raw .set (EEGLAB format) with MNE
2.  High-pass filter at 0.5 Hz  (remove DC drift)
3.  Notch filter at 50 / 100 Hz (line noise)
4.  Common Average Reference (CAR)
5.  ICA — remove ocular + muscle artifacts (automatic)
6.  Resample to 128 Hz
7.  Epoch into 2-second windows, 1-second overlap
8.  Auto-reject epochs exceeding ±150 µV
9.  Extract 313 features via core.feature_extractor
10. Attach subject ID + label → save parquet

EXPECTED DIRECTORY LAYOUT
--------------------------
data/raw/
  sub-S1/ses-S1/eeg/  *.set  *.fdt
  sub-S2/ses-S2/eeg/  ...
  sub-S3/ses-S3/eeg/  ...

Label mapping from task+condition:
  nback_0back → 0 (LOW)
  nback_1back → 1 (MEDIUM)
  nback_2back → 2 (HIGH)
  matb_easy   → 0
  matb_med    → 1
  matb_diff   → 2
"""

import argparse
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yaml
from loguru import logger

from core.feature_extractor import extract_features

mne.set_log_level("WARNING")

# ── Config ────────────────────────────────────────────────────────────────────
_cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
with open(_cfg_path) as f:
    CFG = yaml.safe_load(f)

SFREQ_RAW      = CFG["eeg"]["sfreq_raw"]
SFREQ_TARGET   = CFG["eeg"]["sfreq_resample"]
HIGHPASS       = CFG["eeg"]["highpass_hz"]
NOTCH          = CFG["eeg"]["notch_hz"]
EPOCH_DUR      = CFG["eeg"]["epoch_duration_s"]
EPOCH_OVERLAP  = CFG["eeg"]["epoch_overlap_s"]
EPOCH_STEP     = EPOCH_DUR - EPOCH_OVERLAP          # 1.0 s
REJECT_UV      = dict(eeg=150e-6)                   # 150 µV auto-reject threshold

LABEL_MAP = {
    "0back": 0, "easy": 0,
    "1back": 1, "med":  1,
    "2back": 2, "diff": 2,
    "hard":  2,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_label(filepath: Path) -> int | None:
    """
    Infer cognitive load label from filename.
    COG-BCI filenames contain task and condition strings.
    Returns 0/1/2 or None if unrecognised.
    """
    name = filepath.stem.lower()
    for key, val in LABEL_MAP.items():
        if key in name:
            return val
    return None


def _infer_subject(filepath: Path) -> str:
    """Extract subject ID (S1/S2/S3) from filepath."""
    for part in filepath.parts:
        m = re.search(r"(S\d)", part, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return "UNKNOWN"


def _process_file(set_path: Path, subject: str, label: int) -> list[np.ndarray]:
    """
    Load one .set file, run the full preprocessing chain, epoch, return list
    of feature vectors (one per good epoch).
    """
    try:
        raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
    except Exception as exc:
        logger.warning(f"Could not load {set_path.name}: {exc}")
        return []

    # ── Filter ───────────────────────────────────────────────────────────────
    raw.filter(l_freq=HIGHPASS, h_freq=None, method="fir",
               fir_window="hamming", verbose=False)
    raw.notch_filter(freqs=NOTCH, verbose=False)

    # ── Reference: Common Average ─────────────────────────────────────────────
    raw.set_eeg_reference("average", projection=False, verbose=False)

    # ── ICA — artifact rejection ──────────────────────────────────────────────
    n_components = min(20, len(raw.ch_names) - 1)
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="fastica",
        random_state=42,
        max_iter=400,
        verbose=False,
    )
    try:
        ica.fit(raw, verbose=False)
        # Automatically find EOG / muscle components
        eog_idx, _ = ica.find_bads_eog(raw, verbose=False) if any(
            "eog" in ch.lower() for ch in raw.ch_names
        ) else ([], None)
        ica.exclude = eog_idx[:3]   # remove at most 3 components
        ica.apply(raw, verbose=False)
    except Exception:
        logger.warning(f"ICA failed for {set_path.name}, skipping ICA step")

    # ── Resample ─────────────────────────────────────────────────────────────
    if raw.info["sfreq"] != SFREQ_TARGET:
        raw.resample(SFREQ_TARGET, verbose=False)

    # ── Epoch with sliding window ─────────────────────────────────────────────
    data   = raw.get_data()          # shape: (n_channels, n_samples)
    sfreq  = raw.info["sfreq"]
    win    = int(EPOCH_DUR  * sfreq)   # samples per window
    step   = int(EPOCH_STEP * sfreq)   # samples per step
    n_samp = data.shape[1]

    feature_rows = []
    bad_epochs   = 0

    for start in range(0, n_samp - win + 1, step):
        epoch = data[:, start:start + win]

        # Auto-reject: skip epochs with extreme amplitude
        if np.abs(epoch).max() > 150e-6:
            bad_epochs += 1
            continue

        features = extract_features(epoch, sfreq=sfreq)
        feature_rows.append(features)

    logger.debug(
        f"{set_path.name}: {len(feature_rows)} good epochs, "
        f"{bad_epochs} rejected"
    )
    return feature_rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(data_dir: str, out_path: str) -> None:
    data_dir = Path(data_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    set_files = sorted(data_dir.rglob("*.set"))
    if not set_files:
        raise FileNotFoundError(f"No .set files found under {data_dir}")

    logger.info(f"Found {len(set_files)} .set files")

    all_rows = []
    for set_path in set_files:
        label   = _infer_label(set_path)
        subject = _infer_subject(set_path)

        if label is None:
            logger.warning(f"Could not infer label for {set_path.name}, skipping")
            continue

        logger.info(f"Processing {set_path.name}  |  subject={subject}  label={label}")
        rows = _process_file(set_path, subject, label)

        for feat in rows:
            row = {"subject": subject, "label": label}
            row.update({f"f{i}": v for i, v in enumerate(feat)})
            all_rows.append(row)

    if not all_rows:
        raise RuntimeError("No feature rows produced. Check your data_dir.")

    df = pd.DataFrame(all_rows)
    df.to_parquet(out_path, index=False)
    logger.success(
        f"Saved {len(df):,} windows × {df.shape[1]-2} features → {out_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COG-BCI batch preprocessor")
    parser.add_argument("--data_dir",  default="data/raw",
                        help="Root dir containing sub-S*/ses-S*/eeg/*.set files")
    parser.add_argument("--out_path",  default="data/processed/features_sota.parquet",
                        help="Output parquet file path")
    args = parser.parse_args()
    run(args.data_dir, args.out_path)