import sys, time
from pathlib import Path
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PL = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d14",
          font=dict(family="Inter",color="#555570",size=11),
          margin=dict(l=8,r=8,t=8,b=8),
          xaxis=dict(gridcolor="#1a1a2e",showline=False,zeroline=False),
          yaxis=dict(gridcolor="#1a1a2e",showline=False,zeroline=False))

def render():
    st.markdown('<div class="page-wrap" style="max-width:1100px;margin:0 auto;padding:32px 48px 60px 48px">', unsafe_allow_html=True)
    st.markdown("## Session Review")
    results = st.session_state.get("results", [])
    alerts  = st.session_state.get("alerts",  [])
    shap    = st.session_state.get("last_shap", {})

    if not results:
        st.markdown('<div class="card"><span class="mono">No session data. Run a monitoring session first.</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    labels  = [r.label_str for r in results]
    ph_low  = labels.count("LOW")  / len(labels) * 100
    ph_high = labels.count("HIGH") / len(labels) * 100
    peak_h  = max(r.prob_high for r in results)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total windows", len(results))
    c2.metric("% LOW",  f"{ph_low:.1f}%")
    c3.metric("% HIGH", f"{ph_high:.1f}%")
    c4.metric("Peak P(HIGH)", f"{peak_h:.3f}")
    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1,2])
    with col_a:
        st.markdown('<div class="section-label">Load distribution</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(labels=["LOW","HIGH"], values=[ph_low,ph_high],
            marker=dict(colors=["#3dd68c","#ff5757"],line=dict(width=0)),
            hole=0.6, textinfo="percent+label", textfont=dict(size=12)))
        fig.update_layout(**PL, height=240, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with col_b:
        st.markdown('<div class="section-label">Full risk timeline</div>', unsafe_allow_html=True)
        ph = [r.prob_high for r in results]
        colors = ["#ff5757" if p>=0.70 else "#3dd68c" for p in ph]
        fig2 = go.Figure(go.Bar(y=ph, marker_color=colors, marker_line_width=0))
        fig2.add_hline(y=0.70, line_dash="dot", line_color="rgba(255,87,87,0.4)", line_width=1)
        layout_cfg = {
            **PL,
            "height": 240,
            "bargap": 0.05,
            "yaxis": {
                **PL.get("yaxis", {}),
                "range": [0, 1],
                "gridcolor": "#1a1a2e"
            }
        }

        fig2.update_layout(**layout_cfg)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

    if shap:
        st.markdown('<div class="section-label">SHAP feature attribution (last window)</div>', unsafe_allow_html=True)
        top    = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
        feats  = [t[0][:22] for t in top]
        vals   = [t[1] for t in top]
        colors = ["#ff5757" if v>0 else "#6c63ff" for v in vals]
        fig3   = go.Figure(go.Bar(x=vals, y=feats, orientation="h",
                           marker_color=colors, marker_line_width=0))
        fig3.add_vline(x=0, line_color="#1e1e30", line_width=1)
        layout_cfg = {
            **PL,
            "height": 360,
            "yaxis": {
                **PL.get("yaxis", {}),
                "autorange": "reversed"
            }
        }

        fig3.update_layout(**layout_cfg)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-label">Alert log ({len(alerts)} alerts)</div>', unsafe_allow_html=True)
    if alerts:
        for a in alerts:
            ts  = datetime.fromtimestamp(a.timestamp).strftime("%H:%M:%S")
            css = {"CRITICAL":"alert-critical","WARNING":"alert-warning","INFO":"alert-info"}.get(a.level.value,"alert-info")
            st.markdown(f'<div class="{css}"><div class="alert-title">{a.code} '
                        f'<span style="font-size:10px;color:#444460">{ts}</span></div>'
                        f'<div class="alert-msg">{a.message}</div>'
                        f'<div class="alert-suggest">{a.suggestion}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card-sm"><span class="mono">No alerts fired.</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Export PDF Report"):
        from reports.pdf_generator import generate_report
        from streaming.alert_engine import AlertEngine
        ae      = st.session_state.get("alert_engine", AlertEngine())
        summary = ae.session_summary()
        sid     = f"SES_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with st.spinner("Generating..."):
            path = generate_report(session_id=sid, results=results, alerts=alerts,
                summary=summary, subject_name="Study Session", shap_values=shap)
        with open(path,"rb") as f:
            st.download_button("Download PDF", f.read(), file_name=path.name, mime="application/pdf")

    st.markdown('</div>', unsafe_allow_html=True)