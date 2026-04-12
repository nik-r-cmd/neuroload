"""
core/train_model.py
────────────────────
PURPOSE
-------
Trains XGBoost (production model) and Random Forest (baseline) on
features_sota.parquet. Runs TWO evaluation schemes:

  1. GroupKFold CV  — 5-fold, groups by subject (your existing approach)
  2. LOSO           — Leave-One-Subject-Out  ← NEW, critical for credibility

LOSO is the harder and more honest test. It answers: "Would this model
work on a person it has never seen?" — exactly what a device company needs.

Saves:
  models/xgb_final.pkl
  models/xgb_scaler_final.pkl
  models/baseline_rf.pkl
  models/model_card.md    ← auto-generated, ready to share

USAGE
-----
  python -m core.train_model
  python -m core.train_model --parquet data/processed/features_sota.parquet
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import yaml
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ── Config ────────────────────────────────────────────────────────────────────
_cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
with open(_cfg_path) as f:
    CFG = yaml.safe_load(f)

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)

LABEL_MAP = CFG["labels"]          # {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
SUBJECTS  = CFG["model"]["subjects"]
CV_SPLITS = CFG["model"]["cv_splits"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df):,} windows, {df.shape[1]-2} features")

    feature_cols = [c for c in df.columns if c not in ("subject", "label")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(int)
    groups = df["subject"].values          # used for GroupKFold
    return X, y, groups, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gkf_eval(model_cls, params: dict, X, y, groups) -> dict:
    """GroupKFold cross-validation. Returns per-fold + aggregate metrics."""
    gkf    = GroupKFold(n_splits=min(CV_SPLITS, len(np.unique(groups))))
    scaler = StandardScaler()

    fold_accs, fold_f1s = [], []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])

        clf = model_cls(**params)
        clf.fit(X_tr, y[tr])
        preds = clf.predict(X_te)

        acc = accuracy_score(y[te], preds)
        f1  = f1_score(y[te], preds, average="macro")
        fold_accs.append(acc)
        fold_f1s.append(f1)
        logger.debug(f"  Fold {fold+1}: acc={acc:.4f}  F1={f1:.4f}")

    return {
        "cv_accuracy_mean": float(np.mean(fold_accs)),
        "cv_accuracy_std":  float(np.std(fold_accs)),
        "cv_f1_macro_mean": float(np.mean(fold_f1s)),
        "cv_f1_macro_std":  float(np.std(fold_f1s)),
        "fold_accuracies":  [round(a, 4) for a in fold_accs],
    }


def _loso_eval(model_cls, params: dict, X, y, groups) -> dict:
    """
    Leave-One-Subject-Out evaluation.
    For each subject: train on remaining subjects, test on held-out subject.
    This is the honest, generalisable accuracy figure.
    """
    subject_ids = np.unique(groups)
    sub_accs, sub_f1s, sub_reports = {}, {}, {}

    for held_out in subject_ids:
        train_mask = groups != held_out
        test_mask  = groups == held_out

        if test_mask.sum() == 0:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_mask])
        X_te = scaler.transform(X[test_mask])

        clf = model_cls(**params)
        clf.fit(X_tr, y[train_mask])
        preds = clf.predict(X_te)

        acc = accuracy_score(y[test_mask], preds)
        f1  = f1_score(y[test_mask], preds, average="macro")
        sub_accs[held_out]    = round(acc, 4)
        sub_f1s[held_out]     = round(f1, 4)
        sub_reports[held_out] = classification_report(
            y[test_mask], preds,
            target_names=list(LABEL_MAP.values()),
            output_dict=True,
        )
        logger.info(f"  LOSO hold-out {held_out}: acc={acc:.4f}  F1={f1:.4f}")

    loso_accs = list(sub_accs.values())
    loso_f1s  = list(sub_f1s.values())
    return {
        "loso_accuracy_mean": float(np.mean(loso_accs)),
        "loso_accuracy_std":  float(np.std(loso_accs)),
        "loso_f1_macro_mean": float(np.mean(loso_f1s)),
        "loso_f1_macro_std":  float(np.std(loso_f1s)),
        "per_subject_accuracy": sub_accs,
        "per_subject_f1":       sub_f1s,
        "per_subject_report":   sub_reports,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SHAP global importance (top 20 features)
# ─────────────────────────────────────────────────────────────────────────────

def _shap_importance(clf, X_scaled: np.ndarray, feature_names: list) -> dict:
    logger.info("Computing SHAP global importance (sample of 500 windows)...")
    sample = X_scaled[np.random.choice(len(X_scaled), min(500, len(X_scaled)), replace=False)]
    explainer = shap.TreeExplainer(clf)
    shap_vals  = explainer.shap_values(sample)

    # For multi-class, average importance across classes
    if isinstance(shap_vals, list):
        mean_abs = np.mean([np.abs(sv).mean(0) for sv in shap_vals], axis=0)
    else:
        mean_abs = np.abs(shap_vals).mean(0)

    top_idx = np.argsort(mean_abs)[::-1][:20]
    return {
        feature_names[i]: round(float(mean_abs[i]), 4) for i in top_idx
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model card
# ─────────────────────────────────────────────────────────────────────────────

def _write_model_card(xgb_results: dict, rf_results: dict) -> None:
    card_path = MODEL_DIR / "model_card.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    loso = xgb_results["loso"]
    gkf  = xgb_results["gkf"]

    lines = [
        "# NeuroLoad — Model Card",
        f"\n_Generated: {ts}_\n",
        "## Dataset",
        "- **Source**: COG-BCI (Zenodo) — 3 subjects, 64-channel EEG at 500 Hz",
        "- **Tasks**: N-Back (0/1/2-back), MATB (easy/med/diff), Flanker, PVT",
        f"- **Windows**: {CFG['eeg']['epoch_duration_s']}s epochs, "
        f"{CFG['eeg']['epoch_overlap_s']}s overlap",
        "- **Features**: 313 per window (bandpower + Hjorth, 5 bands × 64 channels)",
        "",
        "## XGBoost — Production Model",
        "",
        "### GroupKFold CV (within-distribution)",
        f"- Accuracy: **{gkf['cv_accuracy_mean']:.1%}** ± {gkf['cv_accuracy_std']:.1%}",
        f"- Macro F1: **{gkf['cv_f1_macro_mean']:.3f}** ± {gkf['cv_f1_macro_std']:.3f}",
        f"- Per-fold accuracies: {gkf['fold_accuracies']}",
        "",
        "### LOSO — Leave-One-Subject-Out (generalisation estimate)",
        f"- Accuracy: **{loso['loso_accuracy_mean']:.1%}** ± {loso['loso_accuracy_std']:.1%}",
        f"- Macro F1: **{loso['loso_f1_macro_mean']:.3f}** ± {loso['loso_f1_macro_std']:.3f}",
        "",
        "| Subject | LOSO Accuracy | LOSO F1 |",
        "|---------|--------------|---------|",
    ]
    for subj, acc in loso["per_subject_accuracy"].items():
        f1 = loso["per_subject_f1"][subj]
        lines.append(f"| {subj} | {acc:.1%} | {f1:.3f} |")

    lines += [
        "",
        "## Random Forest — Baseline",
        f"- LOSO Accuracy: **{rf_results['loso']['loso_accuracy_mean']:.1%}**",
        f"- LOSO F1: **{rf_results['loso']['loso_f1_macro_mean']:.3f}**",
        "",
        "## Top 20 SHAP Features (XGBoost, global)",
        "",
        "| Feature | Mean |SHAP| |",
        "|---------|-------------|",
    ]
    for feat, val in xgb_results["shap_importance"].items():
        lines.append(f"| {feat} | {val:.4f} |")

    lines += [
        "",
        "## Known Limitations",
        "- Training set: only 3 subjects — LOSO results reflect small-N variance.",
        "- No clinical validation against burnout/fatigue self-report scales.",
        "- Preprocessing assumes stationarity within 2-second windows.",
        "- Real-time inference uses synthetic channel mapping when hardware channels < 64.",
    ]

    card_path.write_text("\n".join(lines))
    logger.success(f"Model card saved → {card_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(parquet_path: str) -> None:
    X, y, groups, feat_names = load_data(parquet_path)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_params = dict(
        n_estimators      = CFG["model"]["n_estimators"],
        max_depth         = CFG["model"]["max_depth"],
        learning_rate     = CFG["model"]["learning_rate"],
        subsample         = CFG["model"]["subsample"],
        colsample_bytree  = CFG["model"]["colsample_bytree"],
        objective         = "multi:softprob",
        num_class         = 3,
        tree_method       = "hist",
        random_state      = 42,
        n_jobs            = -1,
        eval_metric       = "mlogloss",
    )

    logger.info("=== XGBoost: GroupKFold CV ===")
    xgb_gkf = _gkf_eval(XGBClassifier, xgb_params, X, y, groups)

    logger.info("=== XGBoost: LOSO ===")
    xgb_loso = _loso_eval(XGBClassifier, xgb_params, X, y, groups)

    # ── Train final XGBoost on ALL data ───────────────────────────────────────
    logger.info("Training final XGBoost on full dataset...")
    scaler_final = StandardScaler()
    X_scaled     = scaler_final.fit_transform(X)
    xgb_final    = XGBClassifier(**xgb_params)
    xgb_final.fit(X_scaled, y)

    shap_imp = _shap_importance(xgb_final, X_scaled, feat_names)

    joblib.dump(xgb_final,    MODEL_DIR / "xgb_final.pkl")
    joblib.dump(scaler_final, MODEL_DIR / "xgb_scaler_final.pkl")
    logger.success("Saved xgb_final.pkl + xgb_scaler_final.pkl")

    # ── Random Forest baseline ────────────────────────────────────────────────
    rf_params = dict(
        n_estimators=200, max_depth=10, n_jobs=-1, random_state=42
    )

    logger.info("=== Random Forest: GroupKFold CV ===")
    rf_gkf = _gkf_eval(RandomForestClassifier, rf_params, X, y, groups)

    logger.info("=== Random Forest: LOSO ===")
    rf_loso = _loso_eval(RandomForestClassifier, rf_params, X, y, groups)

    # Train final RF
    rf_final = RandomForestClassifier(**rf_params)
    rf_final.fit(X_scaled, y)
    joblib.dump(rf_final, MODEL_DIR / "baseline_rf.pkl")
    logger.success("Saved baseline_rf.pkl")

    # ── Collect and report ────────────────────────────────────────────────────
    xgb_results = {"gkf": xgb_gkf, "loso": xgb_loso, "shap_importance": shap_imp}
    rf_results  = {"gkf": rf_gkf,  "loso": rf_loso}

    _write_model_card(xgb_results, rf_results)

    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info(f"XGBoost  GKF  acc: {xgb_gkf['cv_accuracy_mean']:.1%}")
    logger.info(f"XGBoost  LOSO acc: {xgb_loso['loso_accuracy_mean']:.1%}")
    logger.info(f"RF       LOSO acc: {rf_loso['loso_accuracy_mean']:.1%}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        default="data/processed/features_sota.parquet",
        help="Path to features parquet file",
    )
    args = parser.parse_args()
    run(args.parquet)