"""
dashboard/research_dashboard.py
--------------------------------
Streamlit dashboard for AI Research Layer.
Shows Groq + Gemini analysis, news sentiment, trade setups.

Run: streamlit run dashboard/research_dashboard.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="AI Research — India Screener",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
  .report-card {background:var(--secondary-background-color);border-radius:12px;padding:20px;margin:8px 0;}
  .conviction-badge {font-size:2rem;font-weight:700;}
  .reco-STRONG\.BUY {color:#00e676;}
  .reco-BUY {color:#69f0ae;}
  .reco-HOLD {color:#ffeb3b;}
  .reco-AVOID {color:#ef5350;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=30)
def load_research_db(db_path="data/screener.db"):
    if not os.path.exists(db_path):
        return pd.DataFrame(), pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        research = pd.read_sql("SELECT * FROM ai_research ORDER BY timestamp DESC LIMIT 200", conn)
    except Exception:
        research = pd.DataFrame()
    try:
        briefings = pd.read_sql("SELECT * FROM market_briefings ORDER BY timestamp DESC LIMIT 5", conn)
    except Exception:
        briefings = pd.DataFrame()
    conn.close()
    return research, briefings


def parse_full_report(row) -> dict:
    try:
        return json.loads(row.get("full_report_json", "{}"))
    except Exception:
        return {}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 AI Research")
    st.caption("Groq L1 + Gemini L2")
    st.divider()

    st.subheader("Run Research")
    sym_input = st.text_input("Symbols (comma-separated)", placeholder="RELIANCE, TCS, INFY")
    force_deep = st.checkbox("Force Gemini (skip Groq gating)", value=False)
    if st.button("🔬 Run Research", type="primary"):
        symbols = [s.strip().upper() for s in sym_input.split(",") if s.strip()]
        if symbols:
            with st.spinner(f"Researching {', '.join(symbols)}..."):
                try:
                    import yaml
                    with open("config.yaml") as f:
                        config = yaml.safe_load(f)
                    from ai_research.research_engine import ResearchEngine
                    engine = ResearchEngine(config=config)
                    for sym in symbols:
                        dummy = {
                            "symbol": sym, "ltp": 0, "composite_score": 70,
                            "setup_type": "BTST", "sector": "",
                            "scores": {}, "technical": {}, "volume": {},
                            "smart_money": {}, "trade_setup": {}, "all_details": {}, "news": {},
                        }
                        engine.research(sym, dummy, force_deep=force_deep)
                    st.success(f"Research complete for {', '.join(symbols)}")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Research failed: {e}")
        else:
            st.warning("Enter at least one symbol")

    st.divider()

    if st.button("📊 Daily Briefing"):
        with st.spinner("Generating market briefing..."):
            try:
                import yaml
                with open("config.yaml") as f:
                    config = yaml.safe_load(f)
                from ai_research.research_engine import ResearchEngine
                engine = ResearchEngine(config=config)
                b = engine.get_market_briefing()
                st.session_state["latest_briefing"] = b
                st.cache_data.clear()
                st.success("Briefing generated!")
            except Exception as e:
                st.error(str(e))

    st.divider()
    min_conv = st.slider("Min conviction", 1, 10, 5)
    reco_filter = st.multiselect(
        "Recommendation", ["STRONG BUY", "BUY", "HOLD", "AVOID"],
        default=["STRONG BUY", "BUY"]
    )


# ── Load data ─────────────────────────────────────────────────────────────────
research_df, briefings_df = load_research_db()
st.title("🧠 AI Research Dashboard")
st.caption(f"Groq + Gemini · {datetime.now().strftime('%d %b %Y %H:%M IST')}")

# ── Latest Briefing ───────────────────────────────────────────────────────────
briefing_data = st.session_state.get("latest_briefing")
if briefing_data is None and not briefings_df.empty:
    try:
        briefing_data = json.loads(briefings_df.iloc[0]["full_json"])
    except Exception:
        pass

if briefing_data:
    mood_color = {"RISK_ON": "🟢", "RISK_OFF": "🔴", "NEUTRAL": "🟡"}.get(briefing_data.get("market_mood"), "⚪")
    with st.expander(f"{mood_color} Daily Market Briefing — {briefing_data.get('market_mood')} | {briefing_data.get('nifty_bias')}", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**FII/DII Interpretation**")
            st.write(briefing_data.get("fii_dii_interpretation", "N/A"))
        with c2:
            st.markdown("**Sectors to Watch**")
            for s in briefing_data.get("sectors_to_watch", []):
                st.write(f"✅ {s}")
        with c3:
            st.markdown("**Trader Guidance**")
            st.write(briefing_data.get("trader_guidance", "N/A"))
        if briefing_data.get("risk_warning"):
            st.warning(f"⚠️ {briefing_data['risk_warning']}")

st.divider()

# ── KPIs ──────────────────────────────────────────────────────────────────────
if not research_df.empty:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Reports", len(research_df))
    k2.metric("Strong Buys",   len(research_df[research_df["gemini_recommendation"] == "STRONG BUY"]))
    k3.metric("Avg Conviction",f"{research_df['gemini_conviction'].mean():.1f}/10")
    k4.metric("Avg News Sentiment", f"{research_df['news_sentiment'].mean():+.3f}")
    k5.metric("Articles Analyzed",  int(research_df["article_count"].sum()))
    st.divider()

# ── Research Reports Table ────────────────────────────────────────────────────
st.subheader("📋 AI Research Results")

if not research_df.empty:
    filtered = research_df[
        (research_df["gemini_conviction"] >= min_conv) &
        (research_df["gemini_recommendation"].isin(reco_filter))
    ].copy()

    st.dataframe(
        filtered[["symbol", "ltp", "gemini_recommendation", "gemini_conviction",
                  "groq_verdict", "setup_type", "entry_low", "entry_high",
                  "target_1", "stop_loss", "risk_reward", "news_sentiment",
                  "article_count", "timestamp"]].head(30),
        use_container_width=True,
        column_config={
            "gemini_conviction": st.column_config.ProgressColumn("Conviction", min_value=0, max_value=10),
            "news_sentiment":    st.column_config.NumberColumn("Sentiment", format="%.3f"),
            "risk_reward":       st.column_config.NumberColumn("R/R", format="%.1fx"),
        }
    )

    # ── Detail View ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔬 Deep Report Viewer")
    symbols_with_reports = research_df["symbol"].unique().tolist()
    selected_sym = st.selectbox("Select symbol", symbols_with_reports)

    if selected_sym:
        row = research_df[research_df["symbol"] == selected_sym].iloc[0]
        full = parse_full_report(row)

        if full:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Recommendation",  full.get("final_recommendation", "N/A"))
            col2.metric("Conviction",       f"{full.get('conviction_score', 0)}/10")
            col3.metric("Setup",            full.get("setup_type", "N/A"))
            col4.metric("Time Horizon",     full.get("time_horizon", "N/A"))

            # Executive summary
            if full.get("executive_summary"):
                st.info(full["executive_summary"])

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Trade Setup", "💡 Bull/Bear", "📰 News", "🏦 Smart Money", "⚙️ Raw JSON"
            ])

            with tab1:
                ts = full.get("trade_setup", {})
                tech = full.get("technical_analysis", {})
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Trade Parameters**")
                    st.table(pd.DataFrame({
                        "Parameter": ["Entry Low", "Entry High", "Target 1", "Target 2", "Stop Loss", "R/R"],
                        "Value": [f"₹{ts.get('entry_zone_low','N/A')}",
                                  f"₹{ts.get('entry_zone_high','N/A')}",
                                  f"₹{ts.get('target_1','N/A')}",
                                  f"₹{ts.get('target_2','N/A')}",
                                  f"₹{ts.get('stop_loss','N/A')}",
                                  f"{ts.get('risk_reward','N/A')}x"],
                    }))
                    st.caption(ts.get("position_sizing_note", ""))
                with c2:
                    st.markdown("**Technical Analysis**")
                    st.write(tech.get("trend_structure", "N/A"))
                    if tech.get("key_support_levels"):
                        st.write(f"Support: {tech['key_support_levels']}")
                    if tech.get("key_resistance_levels"):
                        st.write(f"Resistance: {tech['key_resistance_levels']}")

            with tab2:
                bull = full.get("bull_case", {})
                bear = full.get("bear_case", {})
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### 🟢 Bull Case")
                    st.write(bull.get("thesis", ""))
                    for d in bull.get("key_drivers", []):
                        st.write(f"✅ {d}")
                with c2:
                    st.markdown("### 🔴 Bear Case")
                    st.write(bear.get("thesis", ""))
                    for r in bear.get("key_risks", []):
                        st.write(f"⚠️ {r}")

            with tab3:
                news_cat = full.get("news_catalyst_analysis", {})
                news_d   = full.get("_news_data", {})
                st.metric("Articles Analyzed", news_d.get("article_count", 0))
                st.metric("Avg Sentiment", f"{news_d.get('avg_sentiment', 0):+.3f}")
                st.write(f"**Top catalyst:** {news_cat.get('top_catalyst', 'N/A')}")
                st.write(f"**Sentiment:** {news_cat.get('sentiment_direction', 'N/A')}")
                st.markdown("**Recent headlines:**")
                for h in news_d.get("top_headlines", []):
                    st.write(f"• {h[:120]}")
                if news_d.get("corporate_actions"):
                    st.markdown("**Corporate Actions:**")
                    for ca in news_d["corporate_actions"]:
                        st.write(f"• {ca.get('action')} | Ex: {ca.get('ex_date', 'N/A')}")

            with tab4:
                sm = full.get("smart_money_analysis", {})
                st.write(sm.get("summary", "N/A"))
                st.write(f"**FII:** {sm.get('fii_interpretation', 'N/A')}")
                st.write(f"**Bulk deals:** {sm.get('bulk_deal_significance', 'N/A')}")
                st.metric("Smart Money Confidence", sm.get("confidence", "N/A"))

            with tab5:
                st.json(full)
        else:
            st.info("No full report data. Run a fresh research to get detailed analysis.")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Analytics")
    c1, c2 = st.columns(2)

    with c1:
        reco_counts = research_df["gemini_recommendation"].value_counts().reset_index()
        fig = px.pie(reco_counts, values="count", names="gemini_recommendation",
                     title="Recommendation Distribution",
                     color_discrete_map={"STRONG BUY": "#00e676", "BUY": "#69f0ae", "HOLD": "#ffeb3b", "AVOID": "#ef5350"},
                     template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if len(research_df) > 1:
            fig2 = px.scatter(research_df, x="news_sentiment", y="gemini_conviction",
                              color="gemini_recommendation", text="symbol",
                              title="Conviction vs News Sentiment",
                              template="plotly_dark",
                              color_discrete_map={"STRONG BUY": "#00e676", "BUY": "#69f0ae", "HOLD": "#ffeb3b", "AVOID": "#ef5350"})
            st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("No AI research data yet. Use the sidebar to run research on symbols.")

st.divider()
st.caption("🧠 Two-layer AI: Groq (Llama 3.3 70B) + Gemini (2.0 Flash) | MarketAux News API | NSE Data")
