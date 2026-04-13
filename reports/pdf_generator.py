import io
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# =========================
# Safe fpdf2 import
# =========================
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# =========================
# Safe matplotlib import
# =========================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


# =========================
# Config
# =========================
_cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"

if _cfg_path.exists():
    with open(_cfg_path, "r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f)
    COLORS = CFG.get(
        "labels",
        {}
    ).get(
        "colors",
        {
            "LOW": "#2ecc71",
            "MEDIUM": "#f39c12",
            "HIGH": "#e74c3c"
        }
    )
else:
    COLORS = {
        "LOW": "#2ecc71",
        "MEDIUM": "#f39c12",
        "HIGH": "#e74c3c"
    }

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# ASCII-safe sanitizer
# =========================
def safe_text(text):
    """
    Replace unsupported Unicode chars with ASCII-safe equivalents.
    Prevents Helvetica font crashes in fpdf2.
    """
    if text is None:
        return ""

    text = str(text)

    replacements = {
        "—": "--",
        "–": "-",
        "•": "|",
        "→": "->",
        "≤": "<=",
        "≥": ">=",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "…": "...",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    return text.encode("latin-1", "replace").decode("latin-1")


# =========================
# Plot helpers
# =========================
def _pie_chart(summary: dict, tmp_dir: Path) -> Optional[Path]:
    if not MPL_AVAILABLE:
        return None

    fig, ax = plt.subplots(figsize=(4, 4))

    labels = ["LOW", "MEDIUM", "HIGH"]
    sizes = [
        summary.get("pct_low", 0),
        summary.get("pct_medium", 0),
        summary.get("pct_high", 0),
    ]
    colors = [
        COLORS["LOW"],
        COLORS["MEDIUM"],
        COLORS["HIGH"]
    ]

    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        textprops={"fontsize": 10}
    )

    ax.set_title("Cognitive Load Distribution", fontsize=11)

    path = tmp_dir / "pie.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return path


def _timeline_chart(results, tmp_dir: Path) -> Optional[Path]:
    if not MPL_AVAILABLE or not results:
        return None

    probs_h = [r.prob_high for r in results]
    t = list(range(len(probs_h)))

    colors = [
        COLORS["HIGH"] if p >= 0.70
        else COLORS["MEDIUM"] if p >= 0.40
        else COLORS["LOW"]
        for p in probs_h
    ]

    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.bar(t, probs_h, color=colors, width=1.0, edgecolor="none")
    ax.axhline(
        0.70,
        color="#c0392b",
        linewidth=0.8,
        linestyle="--",
        label="Alert threshold"
    )

    ax.set_ylim(0, 1)
    ax.set_xlabel("Window #")
    ax.set_ylabel("P(HIGH)")
    ax.set_title("HIGH-load probability across session")
    ax.legend(fontsize=8)

    path = tmp_dir / "timeline.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return path


def _shap_bar_chart(shap_values: dict, tmp_dir: Path) -> Optional[Path]:
    if not MPL_AVAILABLE or not shap_values:
        return None

    top = sorted(
        shap_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]

    feats = [f[0][:18] for f in top]
    vals = [f[1] for f in top]

    colors = [
        "#e74c3c" if v > 0 else "#2980b9"
        for v in vals
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feats[::-1], vals[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("SHAP value contribution to HIGH class")
    ax.set_title("Top 10 features -- last SHAP window")

    fig.tight_layout(pad=2.0)

    path = tmp_dir / "shap.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return path


# =========================
# PDF Generator
# =========================
def generate_report(
    session_id: str,
    results: list,
    alerts: list,
    summary: dict,
    subject_name: str = "Anonymous",
    shap_values: Optional[dict] = None,
) -> Path:

    if not FPDF_AVAILABLE:
        raise ImportError("Please install fpdf2: pip install fpdf2")

    tmp_dir = Path(tempfile.mkdtemp())

    pie_path = _pie_chart(summary, tmp_dir)
    timeline_path = _timeline_chart(results, tmp_dir)
    shap_path = _shap_bar_chart(shap_values or {}, tmp_dir)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_left_margin(12)
    pdf.set_right_margin(12)
    pdf.add_page()

    eff_w = pdf.w - pdf.l_margin - pdf.r_margin

    # =========================
    # Header
    # =========================
    pdf.set_fill_color(30, 30, 50)
    pdf.rect(0, 0, 210, 40, "F")

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(12, 8)
    pdf.cell(
        eff_w,
        10,
        safe_text("NeuroLoad -- Cognitive Burnout Assessment"),
        ln=True
    )

    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(12, 22)
    pdf.cell(
        eff_w,
        8,
        safe_text("EEG-Based Real-Time Monitoring System | COG-BCI Pipeline | v0.2"),
        ln=True
    )

    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(12, 50)

    # =========================
    # Session Info
    # =========================
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(eff_w, 8, "Session Information", ln=True)

    pdf.set_font("Helvetica", "", 10)
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    meta = [
        ("Session ID", session_id),
        ("Subject", subject_name),
        ("Date / Time", ts_str),
        ("Windows analysed", str(summary.get("total_windows", len(results)))),
        ("Duration", f"{summary.get('total_windows', 0)} seconds approx"),
        ("Model", "XGBoost -- COG-BCI (64-ch, 313 features)"),
    ]

    for k, v in meta:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(55, 6, safe_text(k + ":"), ln=False)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, safe_text(v), ln=True)

    pdf.ln(4)

    # =========================
    # Summary
    # =========================
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(eff_w, 8, "Session Summary", ln=True)

    stats = [
        ("% Time LOW load", f"{summary.get('pct_low', 0):.1f}%"),
        ("% Time MEDIUM load", f"{summary.get('pct_medium', 0):.1f}%"),
        ("% Time HIGH load", f"{summary.get('pct_high', 0):.1f}%"),
        ("Peak HIGH prob.", f"{summary.get('peak_high_prob', 0):.3f}"),
        ("Mean HIGH prob.", f"{summary.get('mean_high_prob', 0):.3f}"),
        ("Burnout events", str(summary.get("burnout_events", 0))),
        ("Total alerts fired", str(summary.get("n_alerts", 0))),
    ]

    for k, v in stats:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(65, 6, safe_text(k + ":"), ln=False)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, safe_text(v), ln=True)

    pdf.ln(4)

    # =========================
    # Charts
    # =========================
    if pie_path and pie_path.exists():
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(eff_w, 8, "Load Distribution", ln=True)
        pdf.image(str(pie_path), x=12, w=80)
        pdf.ln(4)

    if timeline_path and timeline_path.exists():
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(eff_w, 8, "Risk Timeline", ln=True)
        pdf.image(str(timeline_path), x=12, w=170)
        pdf.ln(4)

    # =========================
    # Alerts (FIXED WRAPPING)
    # =========================
    if alerts:
        pdf.add_page()
        pdf.set_left_margin(12)
        pdf.set_right_margin(12)
        pdf.set_x(12)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(eff_w, 8, "Alert Log", ln=True)

        for alert in alerts:
            ts = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S")

            pdf.set_x(12)
            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(
                eff_w,
                6,
                safe_text(f"[{ts}] {alert.level.value} -- {alert.code}")
            )

            pdf.set_x(12)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(
                eff_w,
                5,
                safe_text(alert.message)
            )

            pdf.set_x(12)
            pdf.set_font("Helvetica", "I", 9)
            pdf.multi_cell(
                eff_w,
                5,
                safe_text(f"Suggestion: {alert.suggestion}")
            )

            pdf.ln(2)

    # =========================
    # Save
    # =========================
    out_path = OUTPUT_DIR / f"{session_id}.pdf"
    pdf.output(str(out_path))

    return out_path