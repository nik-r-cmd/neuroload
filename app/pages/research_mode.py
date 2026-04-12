import sys, json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PL = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d0d14",
          font=dict(family="Inter",color="#555570",size=11),
          margin=dict(l=8,r=8,t=8,b=8),
          xaxis=dict(gridcolor="#1a1a2e",showline=False,zeroline=False),
          yaxis=dict(gridcolor="#1a1a2e",showline=False,zeroline=False))

LABEL_NAMES = {0:"LOW", 1:"HIGH"}

@st.cache_data
def load_parquet():
    p = ROOT / "data" / "processed" / "features_sota.parquet"
    if not p.exists(): return None
    df = pd.read_parquet(p)
    df["label_str"] = df["label"].map(LABEL_NAMES)
    return df

def render():
    st.markdown('<div class="page-wrap" style="max-width:1100px;margin:0 auto;padding:32px 48px 60px 48px">', unsafe_allow_html=True)
    st.markdown("## Research Overview")

    tab1, tab2, tab3 = st.tabs(["Dataset", "Model Performance", "Biomarkers"])

    with tab1:
        df = load_parquet()
        if df is None:
            st.info("features_sota.parquet not found.")
        else:
            feat_cols = [c for c in df.columns if c not in
                         ("subject","session","label","label_str","file","task")]
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Windows", f"{len(df):,}")
            c2.metric("Features", len(feat_cols))
            c3.metric("Label 0 (LOW)",  str(df["label"].value_counts().get(0,0)))
            c4.metric("Label 1 (HIGH)", str(df["label"].value_counts().get(1,0)))

            st.markdown("#### Label distribution")
            vc  = df["label_str"].value_counts()
            fig = go.Figure(go.Bar(x=vc.index.tolist(), y=vc.values.tolist(),
                marker_color=["#3dd68c","#ff5757"], marker_line_width=0))
            fig.update_layout(**PL, height=220)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

            band_cols = [c for c in feat_cols if "abs" in c][:20]
            if band_cols:
                selected = st.selectbox("Feature distribution", band_cols)
                fig2 = go.Figure()
                for lbl,col in [("LOW","#3dd68c"),("HIGH","#ff5757")]:
                    vals = df.loc[df["label_str"]==lbl, selected].dropna()
                    fig2.add_trace(go.Box(y=vals, name=lbl, marker_color=col, line_color=col))
                fig2.update_layout(**PL, height=280, showlegend=True)
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

    with tab2:
        st.markdown("#### Model evaluation")
        st.markdown("""
| Metric | GroupKFold CV | LOSO |
|--------|--------------|------|
| XGBoost accuracy (retrained) | 79.8% | 50.8% |
| XGBoost macro F1 | 0.511 | 0.252 |
| Random Forest acc (original) | 82.1% | -- |
""")
        st.info("LOSO reflects subject-dependent EEG fingerprints -- a known BCI challenge. "
                "Commercial devices (Neurosity, Muse) solve this with 2-minute per-user calibration.")

        card = ROOT / "models" / "model_card.md"
        if card.exists():
            with st.expander("Full model card"):
                st.markdown(card.read_text())

        loso = ROOT / "models" / "loso_results.json"
        if loso.exists():
            data = json.loads(loso.read_text())
            rows = [{"Session":k,"Accuracy":f"{v['accuracy']:.1%}",
                     "F1":f"{v['f1_macro']:.3f}","Windows":v["n_windows"]}
                    for k,v in data["per_subject"].items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("#### Top SHAP biomarkers")
        shap_csv = ROOT / "models" / "shap_importance_final.csv"
        if shap_csv.exists():
            df_shap = pd.read_csv(shap_csv).head(15)
            st.dataframe(df_shap, use_container_width=True, hide_index=True)
        else:
            st.markdown("""
| Rank | Feature | Importance | Interpretation |
|------|---------|-----------|----------------|
| 1 | gamma_T8_abs | 0.88 | Right temporal gamma -- verbal/numerical overload |
| 2 | delta_F7_abs | 0.76 | Frontal delta -- executive function fatigue |
| 3 | theta_Fz_abs | 0.61 | Midline theta -- working memory demand |
| 4 | alpha_Pz_abs | 0.54 | Posterior alpha -- attention depletion |
""")

        for img_name, title in [
            ("shap_global_bar_final.png", "Global feature importance"),
            ("shap_beeswarm_final.png",   "SHAP beeswarm"),
        ]:
            p = ROOT / "models" / img_name
            if p.exists():
                st.markdown(f"**{title}**")
                st.image(str(p), width=700)

        st.markdown("#### Frequency band reference")
        st.dataframe(pd.DataFrame({
            "Band":["Delta 1-4Hz","Theta 4-8Hz","Alpha 8-13Hz","Beta 13-30Hz","Gamma 30-45Hz"],
            "Role":["Fatigue, slow waves","Working memory load","Relaxation/inhibition",
                    "Active thinking","High cognition/overload"],
            "Burnout signal":["Increases","Increases","Decreases","Variable","Increases"],
        }), use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)