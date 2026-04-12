"""
streaming/inference_engine.py
──────────────────────────────
Loads your trained models and runs prediction + SHAP per window.

BINARY MODEL SUPPORT
--------------------
Your model has classes [0, 1]:
  0 = LOW cognitive load
  1 = HIGH cognitive load  (burnout risk)

prob_low  = probs[0]
prob_high = probs[1]
prob_medium is set to 0.0 (not used in binary model)

All downstream code (dashboard, alerts, PDF) handles both binary and 3-class.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import yaml
from loguru import logger

_ROOT     = Path(__file__).resolve().parents[1]
_cfg_path = _ROOT / "config.yaml"
with open(_cfg_path) as f:
    CFG = yaml.safe_load(f)

N_CHANNELS = 64

# Label map — handles binary [0,1] and 3-class [0,1,2]
LABEL_MAP_BINARY = {0: "LOW", 1: "HIGH"}
LABEL_MAP_3CLASS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}


@dataclass
class InferenceResult:
    timestamp   : float
    label       : int
    label_str   : str
    prob_low    : float
    prob_high   : float
    prob_medium : float = 0.0      # 0 for binary model
    shap_values : dict  = field(default_factory=dict)
    latency_ms  : float = 0.0
    features    : Optional[np.ndarray] = None

    @property
    def probs(self):
        return {"LOW": self.prob_low, "MEDIUM": self.prob_medium, "HIGH": self.prob_high}

    @property
    def dominant_prob(self):
        return max(self.prob_low, self.prob_medium, self.prob_high)


def _load_model():
    for name in ["xgb_final.pkl", "xgb_model.pkl"]:
        p = _ROOT / "models" / name
        if p.exists():
            logger.info(f"Loading model: {p.name}")
            return joblib.load(p)
    raise FileNotFoundError("No XGBoost model found in models/")


def _load_scaler():
    for name in ["xgb_scaler_final.pkl", "xgb_scaler.pkl"]:
        p = _ROOT / "models" / name
        if p.exists():
            logger.info(f"Loading scaler: {p.name}")
            return joblib.load(p)
    raise FileNotFoundError("No scaler pkl found in models/")


def _load_feature_cols() -> list:
    for path in [
        _ROOT / "models" / "xgb_feature_cols_final.pkl",
        _ROOT / "models" / "xgb_feature_columns.json",
        _ROOT / "data" / "processed" / "feature_columns.json",
    ]:
        if path.exists():
            if path.suffix == ".pkl":
                cols = joblib.load(path)
            else:
                with open(path) as f:
                    cols = json.load(f)
            logger.info(f"Feature cols from {path.name}: {len(cols)} features")
            return list(cols)

    logger.warning("No feature column file found — generating default names")
    bands, metrics = ["delta","theta","alpha","beta","gamma"], ["abs","rel","mob","cmp"]
    return [f"{b}_ch{c:02d}_{m}" for c in range(N_CHANNELS) for b in bands for m in metrics]


class InferenceEngine:
    def __init__(self, n_channels: int = N_CHANNELS, shap_every_n: int = 2):
        self.n_channels    = n_channels
        self.shap_every_n  = shap_every_n
        self._window_count = 0

        self.model      = _load_model()
        self.scaler     = _load_scaler()
        self._feat_cols = _load_feature_cols()

        # Detect binary vs 3-class
        self.n_classes  = len(self.model.classes_)
        self._label_map = LABEL_MAP_BINARY if self.n_classes == 2 else LABEL_MAP_3CLASS
        logger.info(f"Model classes: {self.model.classes_} → {'binary' if self.n_classes==2 else '3-class'}")

        try:
            import shap
            self.explainer = shap.TreeExplainer(self.model)
            self._shap_ok  = True
            logger.info("SHAP ready")
        except Exception as e:
            logger.warning(f"SHAP unavailable: {e}")
            self.explainer = None
            self._shap_ok  = False

        from core.realtime_processor import RealtimeProcessor
        self.processor = RealtimeProcessor(n_channels=n_channels)
        logger.success(f"InferenceEngine ready | {len(self._feat_cols)} features | {self.n_classes} classes")

    def predict(self, epoch: np.ndarray) -> InferenceResult:
        t0 = time.time()
        self._window_count += 1

        features         = self.processor.process(epoch)
        features_aligned = self._align(features)
        X                = self.scaler.transform(features_aligned.reshape(1, -1))
        probs            = self.model.predict_proba(X)[0]
        label            = int(np.argmax(probs))

        # Map probabilities correctly for binary vs 3-class
        if self.n_classes == 2:
            prob_low    = float(probs[0])
            prob_high   = float(probs[1])
            prob_medium = 0.0
        else:
            prob_low    = float(probs[0])
            prob_medium = float(probs[1])
            prob_high   = float(probs[2])

        # SHAP every N windows
        shap_dict = {}
        if self._shap_ok and self._window_count % self.shap_every_n == 0:
            try:
                sv = self.explainer.shap_values(X)
                # Binary: sv is (1, n_features) or list of 2
                if isinstance(sv, list):
                    high_sv = sv[-1][0]   # last class = HIGH
                elif sv.ndim == 3:
                    high_sv = sv[0, :, -1]
                else:
                    high_sv = sv[0]

                top = np.argsort(np.abs(high_sv))[::-1][:20]
                fn  = (self._feat_cols if len(self._feat_cols) == len(high_sv)
                       else [f"f{i}" for i in range(len(high_sv))])
                shap_dict = {fn[i]: float(high_sv[i]) for i in top}
            except Exception as e:
                logger.debug(f"SHAP error: {e}")

        return InferenceResult(
            timestamp   = t0,
            label       = label,
            label_str   = self._label_map[label],
            prob_low    = prob_low,
            prob_medium = prob_medium,
            prob_high   = prob_high,
            shap_values = shap_dict,
            latency_ms  = round((time.time() - t0) * 1000, 1),
            features    = features,
        )

    def _align(self, features: np.ndarray) -> np.ndarray:
        n_train, n_live = len(self._feat_cols), len(features)
        if n_live == n_train:
            return features
        elif n_live > n_train:
            return features[:n_train]
        else:
            out = np.zeros(n_train, dtype=np.float32)
            out[:n_live] = features
            return out

    def reset_session(self):
        self.processor.reset()
        self._window_count = 0