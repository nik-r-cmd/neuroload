import time, sys
from pathlib import Path
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SFREQ = 128.0
TASK_LABEL = {"zeroBACK":0,"oneBACK":0,"twoBACK":1,"Flanker":0,
              "MATBeasy":0,"MATBmed":0,"MATBdiff":1,"PVT":0}

BASE = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d14",
            font=dict(family="Inter", color="#555570", size=11),
            margin=dict(l=8,r=8,t=8,b=8))

@st.cache_resource
def get_engine():
    from streaming.inference_engine import InferenceEngine
    return InferenceEngine(shap_every_n=3)

# ── Gauge (speedometer) ────────────────────────────────────────────────────────
def _gauge(value: float, label: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        number=dict(
            suffix="%",
            font=dict(size=28, color=color, family="JetBrains Mono")
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=0,
                tickcolor="rgba(0,0,0,0)",
                ticklen=0,
                tickfont=dict(size=9, color="#444460")
            ),
            bar=dict(color=color, thickness=0.25),
            bgcolor="#0d0d14",
            borderwidth=0,
            steps=[
                dict(range=[0, 40], color="#111118"),
                dict(range=[40, 70], color="#161620"),
                dict(range=[70, 100], color="#1a1016"),
            ],
            threshold=dict(
                line=dict(color=color, width=2),
                value=70
            ) if label == "P(HIGH)" else dict(
                line=dict(color=color, width=0),
                value=0
            ),
        ),
        title=dict(text=label, font=dict(size=11, color="#555570")),
    ))
    fig.update_layout(**BASE, height=180)
    return fig

# ── Raw waveform ───────────────────────────────────────────────────────────────
def _waveform(raw: np.ndarray) -> go.Figure:
    fig = go.Figure()
    t = np.arange(raw.shape[1]) / SFREQ
    colors = ["#6c63ff","#3dd68c","#ff5757","#ffb347"]
    for i, ch in enumerate([0, 8, 16, 32]):
        if ch < raw.shape[0]:
            fig.add_trace(go.Scatter(
                x=t, y=raw[ch] + i * 80e-6, mode="lines",
                line=dict(width=0.8, color=colors[i]), name=f"ch{ch:02d}"))
    fig.update_layout(**BASE, height=170,
        xaxis=dict(gridcolor="#1a1a2e", showline=False, zeroline=False, title="time (s)"),
        yaxis=dict(gridcolor="#1a1a2e", showline=False, zeroline=False, showticklabels=False),
        legend=dict(orientation="h", y=1.1, font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        showlegend=True)
    return fig

# ── Timeline ──────────────────────────────────────────────────────────────────
def _timeline(results: list) -> go.Figure:
    if not results:
        return go.Figure()
    recent = results[-90:]
    ph     = [r.prob_high for r in recent]
    colors = ["#ff5757" if p >= 0.70 else "#3dd68c" for p in ph]
    fig = go.Figure(go.Bar(y=ph, marker_color=colors, marker_line_width=0))
    fig.add_hline(y=0.70, line_dash="dot", line_color="rgba(255,87,87,0.5)", line_width=1)
    fig.update_layout(**BASE, height=160, bargap=0.08,
        xaxis=dict(gridcolor="#1a1a2e", showline=False, zeroline=False),
        yaxis=dict(range=[0,1], gridcolor="#1a1a2e", showline=False, zeroline=False))
    return fig

# ── SHAP bar ──────────────────────────────────────────────────────────────────
def _shap_fig(shap_vals: dict) -> go.Figure:
    if not shap_vals:
        return go.Figure()
    top    = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    feats  = [t[0][:22] for t in top]
    vals   = [t[1] for t in top]
    colors = ["#ff5757" if v > 0 else "#6c63ff" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=feats, orientation="h",
        marker_color=colors, marker_line_width=0))
    fig.add_vline(x=0, line_color="#1e1e30", line_width=1)
    fig.update_layout(**BASE, height=260,
        xaxis=dict(title="SHAP contribution", gridcolor="#1a1a2e", showline=False, zeroline=False),
        yaxis=dict(autorange="reversed", gridcolor="#1a1a2e", showline=False, zeroline=False))
    return fig

# ── Main render ───────────────────────────────────────────────────────────────
def render():
    for k, v in [("running",False),("results",[]),("alerts",[]),
                 ("session_start",None),("last_shap",{}),
                 ("streamer",None),("buffer",None),("alert_engine",None)]:
        if k not in st.session_state:
            st.session_state[k] = v

    st.markdown('<div style="max-width:1100px;margin:0 auto;padding:28px 40px 60px 40px">',
                unsafe_allow_html=True)

    # ── Config row ────────────────────────────────────────────────────────────
    st.markdown('<div style="font-size:10px;font-weight:600;letter-spacing:.12em;'
                'text-transform:uppercase;color:#444460;margin-bottom:10px">Session Configuration</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns([2,2,2,2,1,1])
    with c1: subject = st.selectbox("Subject", ["sub-01","sub-02","sub-03"], label_visibility="collapsed")
    with c2: session = st.selectbox("Session", ["ses-S1","ses-S2","ses-S3"], label_visibility="collapsed")
    with c3: task    = st.selectbox("Task", ["MATBdiff","twoBACK","zeroBACK","MATBeasy","MATBmed","Flanker","oneBACK","PVT"], label_visibility="collapsed")
    with c4: speed_s = st.selectbox("Speed", ["2x","1x (real-time)","4x"], label_visibility="collapsed")
    with c5: start_btn = st.button("Start", disabled=st.session_state.running, use_container_width=True)
    with c6: stop_btn  = st.button("Stop",  disabled=not st.session_state.running, use_container_width=True)

    speed_val = float(speed_s.split("x")[0])

    if start_btn:
        from streaming.eeg_streamer  import EEGStreamer
        from streaming.buffer        import EegBuffer
        from streaming.alert_engine  import AlertEngine
        streamer = EEGStreamer(data_dir=str(ROOT/"data"/"raw"), speed=speed_val)
        ok = streamer.load_session(subject=subject, session=session, task=task)
        if not ok:
            st.error(f"Could not load {task} for {subject}/{session}.")
            st.stop()
        streamer.start()
        st.session_state.update(dict(
            streamer=streamer, buffer=EegBuffer(n_channels=64),
            alert_engine=AlertEngine(), running=True,
            results=[], alerts=[], last_shap={},
            session_start=time.time()))
        st.rerun()

    if stop_btn:
        if st.session_state.streamer:
            st.session_state.streamer.stop()
        st.session_state.running = False
        st.rerun()

    st.markdown('<hr style="border:none;border-top:1px solid #1a1a2e;margin:16px 0">',
                unsafe_allow_html=True)

    results  = st.session_state.results
    last     = results[-1] if results else None
    elapsed  = int(time.time() - st.session_state.session_start) if st.session_state.session_start else 0
    pl       = last.prob_low  if last else 0.0
    ph       = last.prob_high if last else 0.0
    lat      = f"{last.latency_ms:.0f}" if last else "--"
    lbl      = last.label_str if last else "WAITING"
    n_win    = len(results)

    truth_lbl   = TASK_LABEL.get(task, 0)
    truth_str   = "HIGH" if truth_lbl == 1 else "LOW"
    truth_color = "#ff5757" if truth_lbl == 1 else "#3dd68c"

    if lbl == "HIGH":
        state_html = '<span style="color:#ff5757;background:rgba(255,87,87,.08);border:1px solid rgba(255,87,87,.2);padding:5px 16px;border-radius:20px;font-size:13px;font-weight:600">HIGH LOAD</span>'
    elif lbl == "LOW":
        state_html = '<span style="color:#3dd68c;background:rgba(61,214,140,.08);border:1px solid rgba(61,214,140,.2);padding:5px 16px;border-radius:20px;font-size:13px;font-weight:600">LOW LOAD</span>'
    else:
        state_html = '<span style="color:#555570;background:rgba(85,85,112,.08);border:1px solid rgba(85,85,112,.2);padding:5px 16px;border-radius:20px;font-size:13px;font-weight:500">WAITING</span>'

    # ── Top row: gauges + info ─────────────────────────────────────────────────
    g1, g2, info = st.columns([2, 2, 3])

    with g1:
        st.plotly_chart(_gauge(pl, "P(LOW)", "#3dd68c"),
                        use_container_width=True, config={"displayModeBar": False})
    with g2:
        st.plotly_chart(_gauge(ph, "P(HIGH)", "#ff5757"),
                        use_container_width=True, config={"displayModeBar": False})
    with info:
        st.markdown(f'''
        <div style="background:#111118;border:1px solid #1e1e30;border-radius:12px;
                    padding:20px 24px;height:100%;box-sizing:border-box">
          <div style="font-size:10px;text-transform:uppercase;letter-spacing:.1em;
                      color:#444460;margin-bottom:12px">Current State</div>
          <div style="margin-bottom:16px">{state_html}</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
            <div>
              <div style="font-size:10px;color:#444460;text-transform:uppercase;
                          letter-spacing:.08em">Latency</div>
              <div style="font-size:20px;font-weight:600;color:#e8e8f0;
                          font-family:JetBrains Mono">{lat}<span style="font-size:12px;color:#444460"> ms</span></div>
            </div>
            <div>
              <div style="font-size:10px;color:#444460;text-transform:uppercase;
                          letter-spacing:.08em">Windows</div>
              <div style="font-size:20px;font-weight:600;color:#e8e8f0;
                          font-family:JetBrains Mono">{n_win}</div>
            </div>
            <div>
              <div style="font-size:10px;color:#444460;text-transform:uppercase;
                          letter-spacing:.08em">Elapsed</div>
              <div style="font-size:20px;font-weight:600;color:#e8e8f0;
                          font-family:JetBrains Mono">{elapsed}<span style="font-size:12px;color:#444460"> s</span></div>
            </div>
            <div>
              <div style="font-size:10px;color:#444460;text-transform:uppercase;
                          letter-spacing:.08em">Ground Truth</div>
              <div style="font-size:14px;font-weight:600;color:{truth_color};
                          font-family:JetBrains Mono">{truth_str}</div>
            </div>
          </div>
        </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Middle row: waveform full width ───────────────────────────────────────
    ph_wave = st.empty()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bottom row: timeline + SHAP ───────────────────────────────────────────
    ph_bottom = st.empty()

    # ── Alerts ────────────────────────────────────────────────────────────────
    ph_alerts = st.empty()

    # ── PDF ───────────────────────────────────────────────────────────────────
    ph_pdf = st.empty()

    # ── Live inference loop ───────────────────────────────────────────────────
    if st.session_state.running:
        engine   = get_engine()
        streamer = st.session_state.streamer
        buf      = st.session_state.buffer
        alrt     = st.session_state.alert_engine

        chunk = streamer.get_chunk(128)
        if chunk is not None:
            window = buf.push(chunk)
            if window is not None:
                result = engine.predict(window)
                st.session_state.results.append(result)
                if result.shap_values:
                    st.session_state.last_shap = result.shap_values
                alert = alrt.evaluate(result)
                if alert:
                    st.session_state.alerts.append(alert)
        time.sleep(0.8)
        st.rerun()

    # ── Render waveform ───────────────────────────────────────────────────────
    with ph_wave.container():
        st.markdown('<div style="font-size:10px;font-weight:600;letter-spacing:.12em;'
                    'text-transform:uppercase;color:#444460;margin-bottom:8px">'
                    'Raw EEG Signal — channels 0, 8, 16, 32</div>', unsafe_allow_html=True)
        buf_obj = st.session_state.buffer
        if buf_obj and buf_obj.is_warm:
            st.plotly_chart(_waveform(buf_obj.get_rolling_raw(5.0)),
                            use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown('<div style="background:#111118;border:1px solid #1e1e30;border-radius:10px;'
                        'height:170px;display:flex;align-items:center;justify-content:center">'
                        '<span style="font-family:JetBrains Mono;font-size:12px;color:#6c63ff">'
                        'awaiting signal...</span></div>', unsafe_allow_html=True)

    # ── Render timeline + SHAP ────────────────────────────────────────────────
    with ph_bottom.container():
        c_tl, c_shap = st.columns([3, 2])
        with c_tl:
            st.markdown('<div style="font-size:10px;font-weight:600;letter-spacing:.12em;'
                        'text-transform:uppercase;color:#444460;margin-bottom:8px">'
                        'Risk Timeline (last 90 windows)</div>', unsafe_allow_html=True)
            if results:
                st.plotly_chart(_timeline(results), use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.markdown('<div style="background:#111118;border:1px solid #1e1e30;border-radius:10px;'
                            'height:160px;display:flex;align-items:center;justify-content:center">'
                            '<span style="font-family:JetBrains Mono;font-size:12px;color:#444460">'
                            'no data yet</span></div>', unsafe_allow_html=True)
        with c_shap:
            st.markdown('<div style="font-size:10px;font-weight:600;letter-spacing:.12em;'
                        'text-transform:uppercase;color:#444460;margin-bottom:8px">'
                        'SHAP Attribution</div>', unsafe_allow_html=True)
            if st.session_state.last_shap:
                st.plotly_chart(_shap_fig(st.session_state.last_shap),
                                use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown('<div style="background:#111118;border:1px solid #1e1e30;border-radius:10px;'
                            'height:260px;display:flex;align-items:center;justify-content:center">'
                            '<span style="font-family:JetBrains Mono;font-size:12px;color:#444460">'
                            'computed every 3rd window</span></div>', unsafe_allow_html=True)

    # ── Render alerts ─────────────────────────────────────────────────────────
    with ph_alerts.container():
        if st.session_state.alerts:
            st.markdown('<hr style="border:none;border-top:1px solid #1a1a2e;margin:16px 0">',
                        unsafe_allow_html=True)
            st.markdown('<div style="font-size:10px;font-weight:600;letter-spacing:.12em;'
                        'text-transform:uppercase;color:#444460;margin-bottom:8px">'
                        'Alerts</div>', unsafe_allow_html=True)
            for a in reversed(st.session_state.alerts[-4:]):
                ts  = datetime.fromtimestamp(a.timestamp).strftime("%H:%M:%S")
                css = {"CRITICAL":"alert-critical","WARNING":"alert-warning",
                       "INFO":"alert-info"}.get(a.level.value, "alert-info")
                st.markdown(
                    f'<div class="{css}">'
                    f'<div class="alert-title">{a.code} '
                    f'<span style="font-size:10px;color:#444460;font-weight:400">{ts}</span></div>'
                    f'<div class="alert-msg">{a.message}</div>'
                    f'<div class="alert-suggest">{a.suggestion}</div></div>',
                    unsafe_allow_html=True)

    # ── PDF export ────────────────────────────────────────────────────────────
    with ph_pdf.container():
        if results and not st.session_state.running:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Export Session Report (PDF)"):
                _export_pdf()

    st.markdown('</div>', unsafe_allow_html=True)


def _export_pdf():
    from reports.pdf_generator  import generate_report
    from streaming.alert_engine import AlertEngine
    ae      = st.session_state.get("alert_engine", AlertEngine())
    summary = ae.session_summary()
    sid     = f"SES_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with st.spinner("Generating report..."):
        path = generate_report(
            session_id=sid,
            results=st.session_state.results,
            alerts=st.session_state.alerts,
            summary=summary,
            subject_name="Study Session",
            shap_values=st.session_state.last_shap,
        )
    with open(path, "rb") as f:
        st.download_button("Download PDF", f.read(),
                           file_name=path.name, mime="application/pdf")