"""
demo_launcher.py
-----------------
System check + demo using REAL COG-BCI .set files streamed at speed.
No synthetic numpy sine waves -- actual recorded EEG data.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

print("\n" + "="*65)
print("  NEUROLOAD -- EEG Cognitive Burnout Detection")
print("  Demo Launcher + System Check")
print("="*65 + "\n")

# ── STEP 1: Imports ────────────────────────────────────────────────────────────
print("STEP 1  Checking imports...")
errors = []

for pkg, label, install in [
    ("numpy",     "numpy",     None),
    ("pandas",    "pandas",    None),
    ("scipy",     "scipy",     None),
    ("xgboost",   "xgboost",   "pip install xgboost"),
    ("joblib",    "joblib",    None),
    ("shap",      "shap",      "pip install shap"),
    ("streamlit", "streamlit", "pip install streamlit"),
    ("plotly",    "plotly",    "pip install plotly"),
]:
    try:
        m = __import__(pkg)
        ver = getattr(m, "__version__", "")
        print(f"  ok  {label} {ver}")
    except ImportError:
        print(f"  MISSING  {label}  ->  {install}")
        if install: errors.append(label)

try:
    from fpdf import FPDF
    print("  ok  fpdf2")
except ImportError:
    print("  warn  fpdf2 not installed (PDF export disabled)")

try:
    import mne
    print(f"  ok  mne {mne.__version__} (real EEG streaming ready)")
except ImportError:
    print("  warn  mne not installed -- will use numpy fallback for streaming")

try:
    import brainflow
    print("  ok  brainflow (hardware-ready)")
except ImportError:
    print("  info  brainflow not installed -- fine for demo")

if errors:
    print(f"\n  MISSING: {errors}. Run: pip install -r requirements.txt\n")
    sys.exit(1)

print("\n  All critical imports OK\n")

# ── STEP 2: Model files ────────────────────────────────────────────────────────
print("STEP 2  Checking model files...")

checks = {
    "XGBoost model":  ["models/xgb_final.pkl",        "models/xgb_model.pkl"],
    "XGBoost scaler": ["models/xgb_scaler_final.pkl", "models/xgb_scaler.pkl"],
    "Feature cols":   ["models/xgb_feature_cols_final.pkl",
                       "models/xgb_feature_columns.json"],
    "Parquet data":   ["data/processed/features_sota.parquet"],
}

missing_critical = False
for label, paths in checks.items():
    found = next((p for p in paths if (ROOT / p).exists()), None)
    if found:
        size = (ROOT / found).stat().st_size / 1024
        print(f"  ok  {label:20s}  {found}  ({size:.0f} KB)")
    else:
        critical = any(k in label.lower() for k in ["model", "scaler"])
        print(f"  {'MISSING' if critical else 'warn'}  {label:20s}  not found")
        if critical: missing_critical = True

if missing_critical:
    print("\n  Critical model files missing.\n")
    sys.exit(1)

print("\n  Model files OK\n")

# ── STEP 3: Find a real .set file to stream ────────────────────────────────────
print("STEP 3  Finding real EEG data to stream...")

raw_root = ROOT / "data" / "raw"
# Try to find twoBACK (HIGH load) and zeroBACK (LOW load)
found_files = {}
for subj in ["sub-01", "sub-02", "sub-03"]:
    for ses in ["ses-S1", "ses-S2", "ses-S3"]:
        for task in ["twoBACK", "zeroBACK", "MATBdiff", "MATBeasy"]:
            # Handle sub-03/sub-03 nesting
            candidates = [
                raw_root / subj / ses / "eeg" / f"{task}.set",
                raw_root / subj / subj / ses / "eeg" / f"{task}.set",
            ]
            for c in candidates:
                if c.exists() and task not in found_files:
                    found_files[task] = (subj, ses, c)
                    break

if not found_files:
    print("  WARN  No .set files found in data/raw. Using numpy fallback.")
    USE_REAL_DATA = False
else:
    for task, (subj, ses, path) in found_files.items():
        size = path.stat().st_size / 1024 / 1024
        print(f"  ok  {task:12s}  {subj}/{ses}  ({size:.1f} MB)")
    USE_REAL_DATA = True

print()

# ── STEP 4: Load models ────────────────────────────────────────────────────────
print("STEP 4  Loading trained models...")

from streaming.inference_engine import _load_model, _load_scaler, _load_feature_cols

try:
    model     = _load_model()
    scaler    = _load_scaler()
    feat_cols = _load_feature_cols()
    n_classes = len(model.classes_)
    print(f"  ok  XGBoost  |  {model.n_estimators} trees  |  classes={model.classes_}")
    print(f"  ok  Scaler loaded")
    print(f"  ok  Feature columns: {len(feat_cols)}")
    print(f"  ok  {'BINARY (LOW/HIGH)' if n_classes==2 else '3-CLASS'}")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

print()

# ── STEP 5: Real streaming demo ────────────────────────────────────────────────
print("STEP 5  Running real EEG streaming demo...")
print("        Using actual COG-BCI .set files streamed at 4x speed\n")

import numpy as np
from streaming.eeg_streamer     import EEGStreamer
from streaming.buffer           import EegBuffer
from streaming.inference_engine import InferenceEngine
from streaming.alert_engine     import AlertEngine

engine       = InferenceEngine(shap_every_n=3)
alert_engine = AlertEngine()
all_results  = []
all_alerts   = []

def run_task_demo(subject, session, task, n_windows_target=15, speed=4.0):
    """Stream a real .set file and collect N inference windows."""
    print(f"  Streaming: {task} ({subject}/{session}) at {speed}x speed")
    label_map = {0: "LOW", 1: "HIGH"}
    task_truth = {"twoBACK":1,"zeroBACK":0,"MATBdiff":1,"MATBeasy":0,
                  "oneBACK":0,"Flanker":0,"MATBmed":0,"PVT":0}

    streamer = EEGStreamer(data_dir=str(ROOT/"data"/"raw"), speed=speed)
    ok = streamer.load_session(subject=subject, session=session,
                               task=task, loop=True)
    if not ok:
        print(f"  FAILED to load {task}")
        return []

    streamer.start()
    buf     = EegBuffer(n_channels=64)
    results = []

    print(f"  {'Win':>4}  {'Label':>6}  {'P(LOW)':>7}  {'P(HIGH)':>8}  "
          f"{'ms':>5}  {'Truth':>6}")
    print(f"  {'----':>4}  {'------':>6}  {'-------':>7}  {'--------':>8}  "
          f"{'-----':>5}  {'------':>6}")

    timeout = time.time() + 60   # max 60s per task
    truth_str = label_map.get(task_truth.get(task, 0), "LOW")

    while len(results) < n_windows_target and time.time() < timeout:
        chunk = streamer.get_chunk(128)
        if chunk is None:
            time.sleep(0.1)
            continue

        window = buf.push(chunk)
        if window is not None:
            r = engine.predict(window)
            results.append(r)
            all_results.append(r)
            alert = alert_engine.evaluate(r)
            if alert:
                all_alerts.append(alert)
                print(f"\n  ALERT: {alert.code} -- {alert.message}\n")

            match = "OK" if r.label_str == truth_str else "!!"
            print(f"  {len(results):>4}  {r.label_str:>6}  "
                  f"{r.prob_low:>7.3f}  {r.prob_high:>8.3f}  "
                  f"{r.latency_ms:>5.1f}  "
                  f"{truth_str:>5} {match}")

        time.sleep(0.05)

    streamer.stop()
    n_correct = sum(1 for r in results if r.label_str == truth_str)
    pct = n_correct/len(results)*100 if results else 0
    print(f"\n  Task accuracy on {task}: {n_correct}/{len(results)} = {pct:.0f}%")
    print()
    return results


# Run LOW task first, then HIGH task
if USE_REAL_DATA:
    # Find best subject/session combo
    demo_subj = list(found_files.values())[0][0]
    demo_ses  = list(found_files.values())[0][1]

    if "zeroBACK" in found_files:
        s, ses, _ = found_files["zeroBACK"]
        run_task_demo(s, ses, "zeroBACK", n_windows_target=12, speed=4.0)

    if "twoBACK" in found_files:
        s, ses, _ = found_files["twoBACK"]
        run_task_demo(s, ses, "twoBACK", n_windows_target=12, speed=4.0)

    if "MATBdiff" in found_files and "MATBdiff" not in ["twoBACK","zeroBACK"]:
        s, ses, _ = found_files["MATBdiff"]
        run_task_demo(s, ses, "MATBdiff", n_windows_target=8, speed=4.0)
else:
    print("  No real data found -- skipping streaming demo")
    print("  Place .set files in data/raw/sub-01/ses-S1/eeg/ to enable\n")

# ── STEP 6: Summary ────────────────────────────────────────────────────────────
print("STEP 6  Full demo session summary")
print("-"*50)
if all_results:
    labels   = [r.label_str for r in all_results]
    ph_vals  = [r.prob_high for r in all_results]
    pl_vals  = [r.prob_low  for r in all_results]
    n_low    = labels.count("LOW")
    n_high   = labels.count("HIGH")
    print(f"  Total windows   : {len(all_results)}")
    print(f"  LOW predictions : {n_low}  ({n_low/len(all_results)*100:.0f}%)")
    print(f"  HIGH predictions: {n_high}  ({n_high/len(all_results)*100:.0f}%)")
    print(f"  Mean P(HIGH)    : {np.mean(ph_vals):.3f}")
    print(f"  Peak P(HIGH)    : {max(ph_vals):.3f}")
    print(f"  Total alerts    : {len(all_alerts)}")
    print(f"  Avg latency     : {np.mean([r.latency_ms for r in all_results]):.1f} ms")
else:
    print("  No results collected.")
print()

# ── STEP 7: PDF ────────────────────────────────────────────────────────────────
print("STEP 7  Generating PDF report...")
if all_results:
    try:
        from reports.pdf_generator import generate_report
        summary = alert_engine.session_summary()
        sid     = f"DEMO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        last_shap = next((r.shap_values for r in reversed(all_results)
                          if r.shap_values), {})
        path = generate_report(
            session_id   = sid,
            results      = all_results,
            alerts       = all_alerts,
            summary      = summary,
            subject_name = "Demo Session (COG-BCI)",
            shap_values  = last_shap,
        )
        print(f"  ok  PDF saved: {path}\n")
    except Exception as e:
        print(f"  warn  PDF skipped: {e}\n")
else:
    print("  No results to report.\n")

# ── Done ───────────────────────────────────────────────────────────────────────
print("="*65)
print("  SYSTEM CHECK COMPLETE")
print("="*65)
print("""
  LAUNCH STREAMLIT:
    streamlit run app/app.py

  IN THE APP:
    Select sub-01 / ses-S1 / twoBACK / 2x speed -> Start
    Watch HIGH predictions as the model sees real N-Back EEG
    Then switch to zeroBACK -> watch it flip to LOW

  WHAT TO SAY:
    "This is streaming real recorded EEG from a human performing
     a 2-Back cognitive task -- the gold standard workload paradigm.
     The model detects high cognitive load in real time, with SHAP
     showing which brain regions are driving each prediction."
""")