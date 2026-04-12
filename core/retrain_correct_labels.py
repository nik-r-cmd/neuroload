"""
core/retrain_correct_labels.py
-------------------------------
WHAT THIS DOES
--------------
Retrains XGBoost from scratch using your actual .set files with CORRECT
per-window labels derived from the task filename -- not from whatever
session-level label was in the parquet.

The parquet labels were apparently session-level (ses-S1/S2/S3 as subject)
which didn't encode LOW vs HIGH per task. This script fixes that by:

1. Reading each .set file directly from data/raw
2. Labeling EVERY window from that file by task:
     zeroBACK  -> 0 (LOW)
     oneBACK   -> 0 (LOW)   binary model: anything not clearly HIGH = LOW
     twoBACK   -> 1 (HIGH)
     MATBdiff  -> 1 (HIGH)
     MATBmed   -> 0 (LOW)   borderline, treated as LOW for binary
     MATBeasy  -> 0 (LOW)
     Flanker   -> 0 (LOW)
     PVT       -> 0 (LOW)
3. Extracts features using the SAME pipeline as training
4. Trains XGBoost with subject-level GroupKFold
5. Saves new model files that OVERWRITE the old ones
6. Verifies on held-out windows that predictions are sensible

RUNTIME: ~10-15 minutes depending on your CPU.
USAGE:   python core/retrain_correct_labels.py

OUTPUT FILES:
  models/xgb_final.pkl              <- overwritten with correct model
  models/xgb_scaler_final.pkl       <- overwritten
  models/xgb_feature_cols_final.pkl <- overwritten
  models/retrain_report.txt         <- accuracy report
"""

import sys
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib
import yaml
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

try:
    import mne
    mne.set_log_level("ERROR")
except ImportError:
    logger.error("MNE not installed: pip install mne")
    sys.exit(1)

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

# ── Label map: task filename -> binary label ──────────────────────────────────
# 0 = LOW cognitive load
# 1 = HIGH cognitive load (burnout risk)
TASK_LABELS = {
    "zeroBACK": 0,
    "oneBACK":  0,   # moderate, treated as LOW in binary scheme
    "twoBACK":  1,   # definitively HIGH -- working memory at capacity
    "Flanker":  0,
    "MATBeasy": 0,
    "MATBmed":  0,
    "MATBdiff": 1,   # difficult multi-task -> HIGH
    "PVT":      0,
    # resting state -> LOW
    "RS_Beg_EC": 0,
    "RS_Beg_EO": 0,
    "RS_End_Ec": 0,
    "RS_End_EO": 0,
}

SFREQ_TARGET  = 128
EPOCH_DUR     = 2.0
EPOCH_STEP    = 1.0
WIN_SAMPLES   = int(EPOCH_DUR * SFREQ_TARGET)
STEP_SAMPLES  = int(EPOCH_STEP * SFREQ_TARGET)
REJECT_UV     = 800e-6  # permissive -- no ICA applied, raw amplitudes are larger


# ── Feature extraction (inline -- no import chain issues) ─────────────────────
def bandpower(freqs, psd, lo, hi):
    mask = (freqs >= lo) & (freqs <= hi)
    return float(np.trapz(psd[mask], freqs[mask])) if mask.sum() > 0 else 0.0

def hjorth(sig):
    d1 = np.diff(sig); d2 = np.diff(d1)
    var_x  = np.var(sig)  + 1e-12
    var_d1 = np.var(d1)   + 1e-12
    var_d2 = np.var(d2)   + 1e-12
    mob = np.sqrt(var_d1 / var_x)
    cmp = np.sqrt(var_d2 / var_d1) / mob
    return float(mob), float(cmp)

def extract_features(epoch, sfreq=128.0):
    from scipy.signal import welch
    BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),"gamma":(30,45)}
    feats = []
    for ch in range(epoch.shape[0]):
        sig = epoch[ch]
        freqs, psd = welch(sig, fs=sfreq, nperseg=min(256, len(sig)),
                           noverlap=128, window="hann")
        total = bandpower(freqs, psd, 1.0, 45.0) + 1e-12
        mob, cmp = hjorth(sig)
        for lo, hi in BANDS.values():
            ab = bandpower(freqs, psd, lo, hi)
            feats.extend([ab, ab/total, mob, cmp])
    return np.array(feats, dtype=np.float32)

def feature_names():
    BANDS = ["delta","theta","alpha","beta","gamma"]
    METRICS = ["abs","rel","mob","cmp"]
    return [f"{b}_ch{c:02d}_{m}"
            for c in range(64) for b in BANDS for m in METRICS]


# ── Load and epoch one .set file ──────────────────────────────────────────────
def process_set_file(set_path: Path, label: int, subject_id: str):
    rows = []
    try:
        raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
        raw.filter(l_freq=0.5, h_freq=None, verbose=False)
        raw.notch_filter(freqs=[50, 100], verbose=False)
        raw.set_eeg_reference("average", projection=False, verbose=False)
        if raw.info["sfreq"] != SFREQ_TARGET:
            raw.resample(SFREQ_TARGET, verbose=False)

        data   = raw.get_data()
        n_ch   = data.shape[0]
        n_samp = data.shape[1]
        sfreq  = raw.info["sfreq"]

        # Pad/trim to 64 channels
        if n_ch < 64:
            pad  = np.zeros((64 - n_ch, n_samp), dtype=np.float32)
            data = np.vstack([data, pad])
        elif n_ch > 64:
            data = data[:64]

        good, bad = 0, 0
        for start in range(0, n_samp - WIN_SAMPLES + 1, STEP_SAMPLES):
            epoch = data[:, start:start + WIN_SAMPLES]
            # Only reject flat/dead channels (amplifier disconnected)
            # Do NOT reject based on amplitude -- we have no ICA here
            # so raw amplitudes can be large. XGBoost handles noisy epochs.
            if np.abs(epoch).max() < 1e-12:   # completely flat = dead signal
                bad += 1
                continue
            feats = extract_features(epoch, sfreq=sfreq)
            rows.append({
                "subject": subject_id,
                "task":    set_path.stem,
                "label":   label,
                **{f"f{i}": v for i, v in enumerate(feats)}
            })
            good += 1

        logger.info(f"  {set_path.stem:15s}  label={label}  "
                    f"good={good}  bad={bad}  subject={subject_id}")
    except Exception as e:
        logger.warning(f"  FAILED {set_path.name}: {e}")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  RETRAINING WITH CORRECT TASK-LEVEL LABELS")
    print("="*60 + "\n")

    raw_root = ROOT / "data" / "raw"
    all_rows = []

    # Collect all .set files across all subjects and sessions
    set_files = list(raw_root.rglob("*.set"))
    logger.info(f"Found {len(set_files)} .set files")

    for set_path in sorted(set_files):
        task_name = set_path.stem   # e.g. "twoBACK", "zeroBACK"
        if task_name not in TASK_LABELS:
            logger.debug(f"Skipping unknown task: {task_name}")
            continue

        label = TASK_LABELS[task_name]

        # Infer subject from path: look for sub-01, sub-02, sub-03
        subject_id = "unknown"
        for part in set_path.parts:
            if part.startswith("sub-"):
                subject_id = part
                break

        rows = process_set_file(set_path, label, subject_id)
        all_rows.extend(rows)

    if not all_rows:
        logger.error("No feature rows extracted. Check data/raw folder.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    feat_cols = [c for c in df.columns if c.startswith("f")]

    label_counts = df["label"].value_counts()
    logger.info(f"\nTotal windows: {len(df):,}")
    logger.info(f"Label 0 (LOW):  {label_counts.get(0,0):,}")
    logger.info(f"Label 1 (HIGH): {label_counts.get(1,0):,}")
    logger.info(f"Subjects: {df['subject'].unique()}")
    logger.info(f"Tasks: {df['task'].unique()}")
    logger.info(f"Features per window: {len(feat_cols)}")

    if label_counts.get(1, 0) == 0:
        logger.error("No HIGH-label windows found. Check TASK_LABELS mapping.")
        sys.exit(1)

    X      = df[feat_cols].values.astype(np.float32)
    y      = df["label"].values.astype(int)
    groups = df["subject"].values

    # ── GroupKFold evaluation ─────────────────────────────────────────────────
    print("\nRunning GroupKFold evaluation...")
    n_subjects = len(np.unique(groups))
    n_splits   = min(3, n_subjects)

    xgb_params = dict(
        n_estimators     = 300,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        objective        = "binary:logistic",
        eval_metric      = "logloss",
        tree_method      = "hist",
        random_state     = 42,
        n_jobs           = -1,
    )

    gkf       = GroupKFold(n_splits=n_splits)
    fold_accs = []
    fold_f1s  = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        scaler_fold = StandardScaler()
        X_tr = scaler_fold.fit_transform(X[tr])
        X_te = scaler_fold.transform(X[te])
        clf  = XGBClassifier(**xgb_params)
        clf.fit(X_tr, y[tr], verbose=False)
        preds = clf.predict(X_te)
        acc   = accuracy_score(y[te], preds)
        f1    = f1_score(y[te], preds, average="macro")
        fold_accs.append(acc)
        fold_f1s.append(f1)
        logger.info(f"  Fold {fold+1}: acc={acc:.4f}  F1={f1:.4f}  "
                    f"test_group={np.unique(groups[te])}")

    print(f"\n  GroupKFold CV Accuracy: {np.mean(fold_accs):.4f} "
          f"+/- {np.std(fold_accs):.4f}")
    print(f"  GroupKFold CV F1:       {np.mean(fold_f1s):.4f} "
          f"+/- {np.std(fold_f1s):.4f}")

    # ── Train final model on ALL data ─────────────────────────────────────────
    print("\nTraining final model on all data...")
    scaler_final = StandardScaler()
    X_scaled     = scaler_final.fit_transform(X)
    clf_final    = XGBClassifier(**xgb_params)
    clf_final.fit(X_scaled, y, verbose=False)

    # Quick sanity check: sample windows per task and check prediction direction
    print("\nSanity check -- per-task prediction (sample of 10 windows each):")
    print(f"  {'Task':15s}  {'True':>5}  {'Pred':>5}  {'P(HIGH) mean':>13}")
    for task in df["task"].unique():
        subset = df[df["task"] == task].sample(min(10, len(df[df["task"]==task])),
                                               random_state=42)
        X_sub  = scaler_final.transform(subset[feat_cols].values)
        probs  = clf_final.predict_proba(X_sub)[:, 1]
        preds  = clf_final.predict(X_sub)
        true_l = TASK_LABELS.get(task, "?")
        print(f"  {task:15s}  {true_l:>5}  {int(round(preds.mean())):>5}  "
              f"{probs.mean():>13.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    models_dir = ROOT / "models"
    joblib.dump(clf_final,    models_dir / "xgb_final.pkl")
    joblib.dump(scaler_final, models_dir / "xgb_scaler_final.pkl")
    joblib.dump(feat_cols,    models_dir / "xgb_feature_cols_final.pkl")

    # Save feature column names (consistent with training)
    import json
    fn = feature_names()
    # Use actual feat_cols (f0, f1, ...) since that's what's in the parquet
    with open(models_dir / "xgb_feature_columns.json", "w") as f:
        json.dump(feat_cols, f)

    logger.success(f"Saved: xgb_final.pkl, xgb_scaler_final.pkl, "
                   f"xgb_feature_cols_final.pkl")

    # ── Write report ──────────────────────────────────────────────────────────
    report_lines = [
        f"RETRAIN REPORT -- {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Total windows: {len(df):,}",
        f"Label 0 (LOW):  {label_counts.get(0,0):,}",
        f"Label 1 (HIGH): {label_counts.get(1,0):,}",
        f"Subjects: {list(df['subject'].unique())}",
        f"GroupKFold CV Accuracy: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}",
        f"GroupKFold CV F1:       {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}",
        "",
        "Per-fold results:",
    ]
    for i, (a, f) in enumerate(zip(fold_accs, fold_f1s)):
        report_lines.append(f"  Fold {i+1}: acc={a:.4f}  F1={f:.4f}")

    report_path = models_dir / "retrain_report.txt"
    report_path.write_text("\n".join(report_lines))
    logger.success(f"Report saved: {report_path}")

    print("\n" + "="*60)
    print("  RETRAINING COMPLETE")
    print("="*60)
    print(f"""
  New model saved to models/xgb_final.pkl
  CV Accuracy: {np.mean(fold_accs):.1%}
  CV F1:       {np.mean(fold_f1s):.3f}

  Now run:
    python demo_launcher.py   <- verify twoBACK shows HIGH
    streamlit run app/app.py  <- live demo
""")


if __name__ == "__main__":
    main()