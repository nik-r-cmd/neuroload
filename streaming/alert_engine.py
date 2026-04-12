"""
streaming/alert_engine.py
──────────────────────────
Monitors InferenceResult stream and fires alerts.

HANDLES BOTH binary (0=LOW, 1=HIGH) and 3-class (0=LOW,1=MED,2=HIGH) models.
Your model has classes [0,1] — so label 0=LOW, label 1=HIGH.
"""

import time
import copy
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from typing import Optional
import yaml
from pathlib import Path
from loguru import logger

_cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
with open(_cfg_path) as f:
    CFG = yaml.safe_load(f)

CONSEC_HIGH_N = 3
HIGH_PROB_TH  = 0.70


class AlertLevel(str, Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    level        : AlertLevel
    code         : str
    message      : str
    suggestion   : str
    timestamp    : float = field(default_factory=time.time)
    auto_clear_s : Optional[float] = None


ALERT_TEMPLATES = {
    "BURNOUT_DETECTED": Alert(
        level      = AlertLevel.CRITICAL,
        code       = "BURNOUT_DETECTED",
        message    = "Sustained cognitive overload detected across multiple windows.",
        suggestion = "Take a 10-minute break. Step away from the screen. Close eyes.",
    ),
    "HIGH_SINGLE": Alert(
        level      = AlertLevel.WARNING,
        code       = "HIGH_SINGLE",
        message    = "High cognitive load spike detected.",
        suggestion = "Consider reducing task complexity or taking a short pause.",
        auto_clear_s = 30.0,
    ),
    "SUSTAINED_MEDIUM": Alert(
        level      = AlertLevel.INFO,
        code       = "SUSTAINED_MEDIUM",
        message    = "Prolonged moderate load — early fatigue pattern.",
        suggestion = "5-minute micro-break recommended.",
        auto_clear_s = 60.0,
    ),
    "RECOVERY": Alert(
        level      = AlertLevel.INFO,
        code       = "RECOVERY",
        message    = "Cognitive load returning to baseline.",
        suggestion = "Good — continue with moderate task complexity.",
        auto_clear_s = 15.0,
    ),
}


class AlertEngine:
    def __init__(self):
        self._history        = deque(maxlen=30)
        self._consec_high    = 0
        self._consec_medium  = 0
        self._was_high       = False
        self._burnout_fired  = False
        self.active_alerts   : list[Alert] = []
        self.all_alerts      : list[Alert] = []

    def evaluate(self, result) -> Optional[Alert]:
        self._history.append(result)
        self._expire_alerts()

        # Works for both binary [0,1] and 3-class [0,1,2]
        label = result.label

        # "high" is the last class index
        is_high   = (label == result.label and result.dominant_prob >= HIGH_PROB_TH
                     and label >= 1)
        is_medium = (label == 1 and hasattr(result, 'prob_medium')
                     and result.prob_medium > 0.4)

        # Simpler: use prob_high if available, else dominant
        high_prob = getattr(result, 'prob_high', result.dominant_prob if label >= 1 else 0.0)

        if high_prob >= HIGH_PROB_TH:
            self._consec_high   += 1
            self._consec_medium  = 0
        else:
            if label == 1 and high_prob < HIGH_PROB_TH:
                self._consec_medium += 1
            else:
                self._consec_medium = 0
            self._consec_high = 0

        alert = None

        if self._consec_high >= CONSEC_HIGH_N and not self._burnout_fired:
            alert = self._make(ALERT_TEMPLATES["BURNOUT_DETECTED"])
            self._burnout_fired = True
            self._was_high = True
            logger.warning("ALERT: BURNOUT_DETECTED")

        elif high_prob >= HIGH_PROB_TH and self._consec_high < CONSEC_HIGH_N:
            alert = self._make(ALERT_TEMPLATES["HIGH_SINGLE"])

        elif self._consec_medium >= 10:
            alert = self._make(ALERT_TEMPLATES["SUSTAINED_MEDIUM"])
            self._consec_medium = 0

        elif high_prob < HIGH_PROB_TH and self._was_high:
            alert = self._make(ALERT_TEMPLATES["RECOVERY"])
            self._was_high      = False
            self._burnout_fired = False

        if alert:
            self.active_alerts.append(alert)
            self.all_alerts.append(alert)

        return alert

    def _make(self, template: Alert) -> Alert:
        a = copy.copy(template)
        a.timestamp = time.time()
        return a

    def _expire_alerts(self):
        now = time.time()
        self.active_alerts = [
            a for a in self.active_alerts
            if a.auto_clear_s is None or (now - a.timestamp) < a.auto_clear_s
        ]

    def session_summary(self) -> dict:
        if not self._history:
            return {}
        labels  = [r.label for r in self._history]
        ph_vals = [r.prob_high for r in self._history]
        pl_vals = [r.prob_low  for r in self._history]
        n = len(labels)
        return {
            "total_windows"  : n,
            "pct_low"        : round(labels.count(0) / n * 100, 1),
            "pct_high"       : round(sum(1 for l in labels if l >= 1) / n * 100, 1),
            "mean_p_low"     : round(float(sum(pl_vals) / n), 3),
            "mean_p_high"    : round(float(sum(ph_vals) / n), 3),
            "peak_p_high"    : round(float(max(ph_vals)), 3),
            "n_alerts"       : len(self.all_alerts),
            "burnout_events" : sum(1 for a in self.all_alerts if a.code == "BURNOUT_DETECTED"),
        }

    def reset(self):
        self._history.clear()
        self._consec_high   = 0
        self._consec_medium = 0
        self._was_high      = False
        self._burnout_fired = False
        self.active_alerts  = []
        self.all_alerts     = []