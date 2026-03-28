"""
dashboard/app.py
----------------
Streamlit dashboard for India Smart Stock Screener.
Run with: streamlit run dashboard/app.py
"""

import sys
import os
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="India Stock Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {background:#1e2130;border-radius:12px;padding:16px;margin:4px;}
  .score-high   {color:#00e676;font-size:2.2rem;font-weight:700;}
  .score-mid    {color:#ffeb3b;font-size:2.2rem;font-weight:700;}
  .score-low    {color:#ef5350;font-size:2.2rem;font-weight:700;}
  .badge-intraday{background:#0288d1;color:#fff;padding:2px 8px;border-radius:4px;font-size:0.75rem;}
  .badge-btst   {background:#7b1fa2;color:#fff;padding:2px 8px;border-radius:4px;font-size:0.75rem;}
  .badge-swing  {background:#2e7d32;color:#fff;padding:2px 8px;border-radius:4px;font-size:0.75rem;}
  [data-testid="stSidebar"] {background:#111827;}
</style>
""", unsafe_allow_html=True)


# ── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_db_results(db_path: str = "data/screener.db") -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM results ORDER BY timestamp DESC LIMIT 500", conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def load_csv_results(csv_dir: str = "data/results") -> pd.DataFrame:
    if not os.path.exists(csv_dir):
        return pd.DataFrame()
    files = sorted(
        [f for f in os.listdir(csv_dir) if f.endswith(".csv")],
        reverse=True,
    )
    if not files:
        return pd.DataFrame()
    return pd.read_csv(os.path.join(csv_dir, files[0]))


def run_live_screen(mode: str):
    """Trigger a live screener run from the UI."""
    from screener.screener import IndiaStockScreener
    with st.spinner(f"Running {mode.upper()} screen... (may take 2-5 min)"):
        screener = IndiaStockScreener()
        df = screener.run(mode=mode, max_workers=6)
    return df


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/4/41/Flag_of_India.svg", width=60)
    st.title("🇮🇳 India\nStock Screener")
    st.divider()

    data_source = st.radio("Data Source", ["Load from DB", "Load from CSV", "Run Live Screen"])

    if data_source == "Run Live Screen":
        mode = st.selectbox("Screen Mode", ["all", "intraday", "btst", "swing"])
        if st.button("▶ Run Screener", type="primary"):
            st.session_state["live_df"] = run_live_screen(mode)

    st.divider()
    st.subheader("Filters")
    min_score    = st.slider("Min Score", 0, 100, 55)
    setup_filter = st.multiselect("Setup Type", ["INTRADAY", "BTST", "SWING", "WATCH"], default=["INTRADAY", "BTST", "SWING"])

    st.divider()
    st.caption("⚠️ Not financial advice. DYOR.")


# ── Main ─────────────────────────────────────────────────────────────────────

st.title("📈 India Smart Stock Screener")
st.caption(f"Institutional-grade signals · Refreshed: {datetime.now().strftime('%d %b %Y %H:%M IST')}")

# Load data
if data_source == "Load from DB":
    df = load_db_results()
elif data_source == "Load from CSV":
    df = load_csv_results()
else:
    df = st.session_state.get("live_df", pd.DataFrame())

if df.empty:
    st.info("No data yet. Run a live screen or check your data directory.")
    st.stop()

# Normalize column names
df.columns = [c.strip() for c in df.columns]
score_col = "Score" if "Score" in df.columns else "score"
signal_col = "Signal" if "Signal" in df.columns else "signal"
setup_col  = "Setup"  if "Setup"  in df.columns else "setup_type"

# Apply filters
filtered = df[df[score_col] >= min_score]
if setup_filter:
    filtered = filtered[filtered[setup_col].isin(setup_filter)]

# ── KPI Row ──────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Setups", len(filtered))
k2.metric("Strong Buys", len(filtered[filtered[signal_col].str.contains("STRONG", na=False)]))
k3.metric("Intraday", len(filtered[filtered[setup_col] == "INTRADAY"]))
k4.metric("BTST", len(filtered[filtered[setup_col] == "BTST"]))
k5.metric("Swing", len(filtered[filtered[setup_col] == "SWING"]))

st.divider()

# ── Results Table ─────────────────────────────────────────────────────────────

st.subheader("📋 Screener Results")
display_cols = [c for c in [
    "Symbol", "LTP", score_col, signal_col, setup_col,
    "RSI", "Vol_Spike", "Delivery%",
    "SM_Score", "Vol_Score", "Tech_Score", "News_Score", "Fund_Score",
    "Target", "SL", "RR",
] if c in filtered.columns]

st.dataframe(
    filtered[display_cols].head(50),
    use_container_width=True,
    column_config={
        score_col:   st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
        "Vol_Spike": st.column_config.NumberColumn("Vol ×", format="%.1fx"),
        "Delivery%": st.column_config.NumberColumn("Del %", format="%.0f%%"),
        "RR":        st.column_config.NumberColumn("R/R", format="%.1fx"),
    }
)

# ── Charts ────────────────────────────────────────────────────────────────────

st.divider()
col1, col2 = st.columns(2)

with col1:
    st.subheader("Score Distribution")
    fig = px.histogram(
        filtered, x=score_col, nbins=20,
        color=setup_col, barmode="overlay",
        color_discrete_map={"INTRADAY": "#0288d1", "BTST": "#7b1fa2", "SWING": "#2e7d32"},
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Setup Breakdown")
    setup_counts = filtered[setup_col].value_counts().reset_index()
    fig2 = px.pie(
        setup_counts, values="count", names=setup_col,
        color=setup_col,
        color_discrete_map={"INTRADAY": "#0288d1", "BTST": "#7b1fa2", "SWING": "#2e7d32"},
        template="plotly_dark",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Top Picks Detail ──────────────────────────────────────────────────────────

st.divider()
st.subheader("🔴 Top 5 Strong Setups — Signal Radar")

top5 = filtered.nlargest(5, score_col)
if not top5.empty:
    score_cols = ["SM_Score", "Vol_Score", "Tech_Score", "News_Score", "Fund_Score"]
    score_cols = [c for c in score_cols if c in top5.columns]

    if score_cols:
        fig3 = go.Figure()
        categories = ["Smart Money", "Volume", "Technical", "News", "Fundamental"]
        for _, row in top5.iterrows():
            vals = [row.get(c, 0) for c in score_cols]
            fig3.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=str(row.get("Symbol", "")),
            ))
        fig3.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            template="plotly_dark",
            showlegend=True,
        )
        st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.caption("Built with ❤️ for Indian traders | Data from NSE, Yahoo Finance, NewsAPI")
