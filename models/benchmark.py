"""
models/benchmark.py
────────────────────
Runs LOSO evaluation on your existing features_sota.parquet
using your already-trained xgb_final.pkl and xgb_scaler_final.pkl.

FIX: Loads the exact feature columns from xgb_feature_cols_final.pkl
     (the same list used during training) instead of guessing from
     the parquet. This avoids string/filename columns leaking in.

USAGE:  python models/benchmark.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from loguru import logger


# ── Load parquet ───────────────────────────────────────────────────────────────
def load_data():
    parquet = ROOT / "data" / "processed" / "features_sota.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"Not found: {parquet}")
    df = pd.read_parquet(parquet)
    logger.info(f"Loaded {len(df):,} windows, {df.shape[1]} total columns")

    # Show all column dtypes so we can see what's in there
    non_numeric = [c for c in df.columns if df[c].dtype == object]
    logger.info(f"Non-numeric columns: {non_numeric}")
    return df


# ── Load model and scaler ──────────────────────────────────────────────────────
def load_model():
    for name in ["xgb_final.pkl", "xgb_model.pkl"]:
        p = ROOT / "models" / name
        if p.exists():
            logger.info(f"Model: {p.name}")
            return joblib.load(p)
    raise FileNotFoundError("No XGBoost pkl in models/")


def load_scaler():
    for name in ["xgb_scaler_final.pkl", "xgb_scaler.pkl"]:
        p = ROOT / "models" / name
        if p.exists():
            logger.info(f"Scaler: {p.name}")
            return joblib.load(p)
    raise FileNotFoundError("No scaler pkl in models/")


def load_feature_cols():
    """
    Load the EXACT feature columns used during training.
    This is the critical fix — we use the saved column list,
    NOT every column in the parquet.
    """
    # Priority 1: pkl file (xgb_feature_cols_final.pkl from your screenshot)
    pkl = ROOT / "models" / "xgb_feature_cols_final.pkl"
    if pkl.exists():
        cols = joblib.load(pkl)
        logger.info(f"Feature cols from pkl: {len(cols)}")
        return list(cols)

    # Priority 2: JSON in models/
    for jname in ["xgb_feature_columns.json", "rf_feature_columns.json"]:
        jp = ROOT / "models" / jname
        if jp.exists():
            with open(jp) as f:
                cols = json.load(f)
            logger.info(f"Feature cols from {jname}: {len(cols)}")
            return cols

    # Priority 3: JSON in data/processed/
    jp2 = ROOT / "data" / "processed" / "feature_columns.json"
    if jp2.exists():
        with open(jp2) as f:
            cols = json.load(f)
        logger.info(f"Feature cols from data/processed/feature_columns.json: {len(cols)}")
        return cols

    # Last resort: infer from parquet — only numeric, non-label, non-subject cols
    logger.warning("No feature column file found — inferring from parquet numeric cols")
    return None   # signal to caller to infer


# ── Identify subject column ────────────────────────────────────────────────────
def find_subject_col(df: pd.DataFrame) -> str:
    """Find whichever column holds subject IDs."""
    for candidate in ["subject", "Subject", "participant", "sub", "session"]:
        if candidate in df.columns:
            return candidate
    # Try columns that look like subject IDs
    for c in df.columns:
        if df[c].dtype == object and df[c].nunique() <= 10:
            sample = str(df[c].iloc[0])
            if any(k in sample.lower() for k in ["sub", "s0", "s1", "s2", "s3", "01", "02", "03"]):
                logger.info(f"Using '{c}' as subject column (sample: {sample!r})")
                return c
    raise ValueError(f"Cannot find subject column. Columns: {list(df.columns)}")


def find_label_col(df: pd.DataFrame) -> str:
    for candidate in ["label", "Label", "y", "target", "cognitive_load"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Cannot find label column. Columns: {list(df.columns)}")


# ── LOSO ──────────────────────────────────────────────────────────────────────
def run_loso(df, model, scaler, feat_cols, subj_col, label_col):
    subjects = df[subj_col].unique()
    logger.info(f"Subjects found: {sorted(subjects)}")

    results = {}
    all_true, all_pred = [], []

    for subj in sorted(subjects):
        mask   = df[subj_col] == subj
        subset = df.loc[mask, feat_cols]

        # Drop any remaining non-numeric (shouldn't happen with feat_cols, but safe)
        subset = subset.select_dtypes(include=[np.number])

        X_test = scaler.transform(subset.values)
        y_test = df.loc[mask, label_col].values

        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="macro")

        results[str(subj)] = {
            "accuracy":  round(float(acc), 4),
            "f1_macro":  round(float(f1), 4),
            "n_windows": int(mask.sum()),
        }
        all_true.extend(y_test)
        all_pred.extend(y_pred)

        logger.info(
            f"  {str(subj):12s}  acc={acc:.4f}  F1={f1:.4f}  "
            f"n={mask.sum()}"
        )

    overall_acc = float(accuracy_score(all_true, all_pred))
    overall_f1  = float(f1_score(all_true, all_pred, average="macro"))

    return {
        "per_subject":     results,
        "overall_acc":     round(overall_acc, 4),
        "overall_f1":      round(overall_f1, 4),
        "n_subjects":      len(subjects),
        "n_windows_total": len(df),
    }


# ── Save results ───────────────────────────────────────────────────────────────
def save_results(loso: dict):
    out = ROOT / "models" / "loso_results.json"
    with open(out, "w") as f:
        json.dump(loso, f, indent=2)
    logger.success(f"Saved: {out}")

    card = ROOT / "models" / "model_card.md"
    from datetime import datetime
    block = f"\n## LOSO Results ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n"
    block += f"Overall Accuracy: **{loso['overall_acc']:.1%}**  \n"
    block += f"Overall Macro F1: **{loso['overall_f1']:.3f}**  \n\n"
    block += "| Subject | Accuracy | F1 | Windows |\n"
    block += "|---------|----------|----|---------|\n"
    for subj, res in loso["per_subject"].items():
        block += f"| {subj} | {res['accuracy']:.1%} | {res['f1_macro']:.3f} | {res['n_windows']} |\n"

    with open(card, "a" if card.exists() else "w") as f:
        f.write(block)
    logger.success(f"Model card updated: {card}")
    print(block)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "═"*55)
    print("  LOSO Benchmark — using trained xgb_final.pkl")
    print("═"*55 + "\n")

    df           = load_data()
    model        = load_model()
    scaler       = load_scaler()
    feat_cols_raw = load_feature_cols()

    subj_col  = find_subject_col(df)
    label_col = find_label_col(df)

    logger.info(f"Subject col: '{subj_col}', Label col: '{label_col}'")

    # If no saved feature col list, infer from parquet (numeric only, not subject/label)
    if feat_cols_raw is None:
        exclude = {subj_col, label_col}
        feat_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
        logger.info(f"Inferred {len(feat_cols)} numeric feature cols from parquet")
    else:
        # Use saved list, but verify each column actually exists in the parquet
        feat_cols = [c for c in feat_cols_raw if c in df.columns]
        missing   = [c for c in feat_cols_raw if c not in df.columns]
        if missing:
            logger.warning(f"{len(missing)} saved feature cols not in parquet — will be ignored")
        logger.info(f"Using {len(feat_cols)} of {len(feat_cols_raw)} saved feature cols")

    if len(feat_cols) == 0:
        raise ValueError("No usable feature columns found. Check your parquet structure.")

    logger.info(f"Model estimators: {model.n_estimators}")
    logger.info(f"Running LOSO...\n")

    loso = run_loso(df, model, scaler, feat_cols, subj_col, label_col)
    save_results(loso)

    print(f"\n{'─'*45}")
    print(f"  LOSO Overall Accuracy : {loso['overall_acc']:.1%}")
    print(f"  LOSO Overall Macro F1 : {loso['overall_f1']:.3f}")
    print(f"  Subjects              : {loso['n_subjects']}")
    print(f"  Total windows         : {loso['n_windows_total']:,}")
    print(f"{'─'*45}")
    print(f"\n  Saved → models/loso_results.json")
    print(f"  Saved → models/model_card.md\n")


if __name__ == "__main__":
    main()