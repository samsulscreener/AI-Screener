"""
prompts.py
----------
All prompt templates for the two-layer AI research engine.
Keeping prompts in one place makes iteration and A/B testing easy.
"""

# ─────────────────────────────────────────────────────────────────────────────
#  LAYER 1 — Groq / Llama 3.3 70B
#  Goal: Fast bull/bear triage. Return structured JSON only.
#  Target latency: < 2 seconds
# ─────────────────────────────────────────────────────────────────────────────

GROQ_SYSTEM = """You are a senior Indian equity trader with 15 years of experience on NSE/BSE.
You specialise in spotting institutional setups for intraday, BTST, and swing trades.

Your job is FAST TRIAGE. Given screener signals and recent news headlines for a stock,
you must decide quickly: is this worth a deeper research dive or not?

ALWAYS respond with ONLY valid JSON — no markdown, no explanation, no preamble.
Your JSON must exactly match the schema shown in the user prompt.
Think in terms of what FIIs, prop desks, and hedge funds would do with this data."""


GROQ_TRIAGE_PROMPT = """Perform fast triage on this NSE stock setup.

=== STOCK DATA ===
Symbol: {symbol}
LTP: ₹{ltp}
Composite Score: {composite_score}/100
Setup Type: {setup_type}
Sector: {sector}

=== SIGNAL BREAKDOWN ===
Smart Money Score : {smart_money_score}/100  | {smart_money_detail}
Volume Score      : {volume_score}/100       | Vol spike: {vol_spike}x | Delivery: {delivery_pct}%
Technical Score   : {technical_score}/100    | RSI: {rsi} | Supertrend: {supertrend} | EMA aligned: {ema_aligned}
News Score        : {news_score}/100         | Avg sentiment: {avg_sentiment}
Fundamental Score : {fund_score}/100         | {fundamental_detail}

=== RECENT NEWS HEADLINES (last 24h) ===
{news_headlines}

=== MARKET CONTEXT ===
FII Activity Today: {fii_activity}
India VIX: {india_vix}
Global Sentiment: {global_sentiment}

Respond with ONLY this JSON structure (no markdown fences, no extra text):
{{
  "proceed_to_deep_research": true or false,
  "quick_verdict": "STRONG BUY" or "BUY" or "NEUTRAL" or "AVOID",
  "conviction": 1-10,
  "bull_thesis": "2-3 sentence bull case based on the signals above",
  "bear_thesis": "2-3 sentence bear case / key risks",
  "key_catalysts": ["catalyst 1", "catalyst 2", "catalyst 3"],
  "key_risks": ["risk 1", "risk 2"],
  "best_setup_type": "INTRADAY" or "BTST" or "SWING",
  "suggested_holding_period": "e.g. 1-2 days",
  "entry_strategy": "one sentence on entry approach",
  "institutional_interest": "HIGH" or "MEDIUM" or "LOW",
  "groq_reasoning": "1-2 sentences explaining your verdict"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
#  LAYER 2 — Google Gemini 2.0 Flash
#  Goal: Deep, comprehensive research report with actionable trade advice.
#  Target latency: < 10 seconds
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_SYSTEM = """You are a top-tier equity research analyst at a leading Indian institutional fund.
You have expertise in:
- Indian equity markets (NSE/BSE), SEBI regulations, F&O dynamics
- Macro factors affecting Indian markets (Fed policy, crude oil, INR/USD, China)  
- Fundamental analysis (financial ratios, earnings quality, management quality)
- Technical analysis (price action, volume, institutional accumulation patterns)
- Smart money flows (FII/DII behavior, bulk deals, insider patterns)

You receive a pre-screened stock with quantitative signals PLUS a Layer 1 AI quick assessment.
Your job is to synthesize everything into a comprehensive research brief that a professional
trader can act on immediately.

Write in a clear, professional tone. Be specific with numbers. Cite the evidence.
Do not be vague — give exact entry zones, targets, and stop-losses.
Acknowledge uncertainty where it exists; do not pretend to know what you don't.

ALWAYS respond with ONLY valid JSON — no markdown, no explanation outside JSON."""


GEMINI_DEEP_RESEARCH_PROMPT = """Conduct deep research on this NSE stock and produce a comprehensive trade brief.

=== SYMBOL: {symbol} | ₹{ltp} | {sector} ===

=== QUANTITATIVE SIGNALS ===
Composite Score  : {composite_score}/100
Setup Type       : {setup_type}
RSI              : {rsi}
MACD             : {macd_histogram}
Supertrend       : {supertrend}
EMA Stack        : {ema_aligned}
Candlestick      : {patterns}
Volume Spike     : {vol_spike}x 20-day avg
Delivery %       : {delivery_pct}%
PCR              : {pcr}
Max Pain         : ₹{max_pain}

=== SMART MONEY ===
FII Net Today    : ₹{fii_net}Cr ({fii_direction})
DII Net Today    : ₹{dii_net}Cr
Bulk Deals       : {bulk_deals}
Insider Activity : {insider_activity}
Promoter Holding : {promoter_holding}%

=== FUNDAMENTALS ===
ROE   : {roe}%
PE    : {pe}
D/E   : {debt_equity}
ROCE  : {roce}%
Recent announcements: {announcements}
Earnings: {earnings_surprise}

=== MARKET NEWS (last 48h) — Full Articles ===
{full_news_articles}

=== LAYER 1 GROQ ASSESSMENT ===
Verdict   : {groq_verdict}
Conviction: {groq_conviction}/10
Bull Thesis: {groq_bull}
Bear Thesis: {groq_bear}
Catalysts  : {groq_catalysts}
Risks      : {groq_risks}

=== MARKET CONTEXT ===
India VIX      : {india_vix}
Nifty 50 trend : {nifty_trend}
Sector trend   : {sector_trend}
FII YTD flows  : {fii_ytd}
Global markets : {global_markets}
Crude oil      : {crude_price} USD/bbl
INR/USD        : {inr_usd}

Produce a comprehensive research report as ONLY this JSON
(no markdown fences, no preamble, no explanation outside the JSON):
{{
  "symbol": "{symbol}",
  "report_timestamp": "ISO timestamp",
  "final_recommendation": "STRONG BUY" or "BUY" or "HOLD" or "AVOID",
  "conviction_score": 1-10,
  "setup_type": "INTRADAY" or "BTST" or "SWING",
  "time_horizon": "e.g. 2-3 days",

  "trade_setup": {{
    "entry_zone_low": number,
    "entry_zone_high": number,
    "target_1": number,
    "target_2": number,
    "stop_loss": number,
    "risk_reward": number,
    "position_sizing_note": "e.g. Risk max 2% of capital"
  }},

  "executive_summary": "3-4 sentence summary of the opportunity",

  "bull_case": {{
    "thesis": "Detailed 4-6 sentence bull case",
    "key_drivers": ["driver 1", "driver 2", "driver 3"],
    "upside_scenario": "What happens if everything goes right"
  }},

  "bear_case": {{
    "thesis": "Detailed 3-4 sentence bear case",
    "key_risks": ["risk 1", "risk 2", "risk 3"],
    "downside_scenario": "What happens if trade goes wrong"
  }},

  "smart_money_analysis": {{
    "summary": "2-3 sentences on institutional activity",
    "fii_interpretation": "What FII behavior means for this stock",
    "bulk_deal_significance": "Assessment of recent bulk/block deals",
    "confidence": "HIGH" or "MEDIUM" or "LOW"
  }},

  "technical_analysis": {{
    "trend_structure": "Description of overall trend",
    "key_support_levels": [number, number],
    "key_resistance_levels": [number, number],
    "momentum_assessment": "Analysis of RSI, MACD, volume",
    "pattern_significance": "What the candlestick pattern implies"
  }},

  "news_catalyst_analysis": {{
    "top_catalyst": "Most significant recent news catalyst",
    "sentiment_direction": "POSITIVE" or "NEGATIVE" or "NEUTRAL",
    "news_surprise_factor": "HIGH" or "MEDIUM" or "LOW",
    "upcoming_events": ["event 1", "event 2"],
    "international_angle": "How global news affects this stock"
  }},

  "fundamental_snapshot": {{
    "valuation_comment": "Is the stock cheap/fair/expensive?",
    "quality_assessment": "ROE, management, business moat",
    "earnings_outlook": "Near-term earnings expectation",
    "sector_positioning": "How this stock stands in its sector"
  }},

  "risk_management": {{
    "max_loss_scenario": "Worst case price and scenario",
    "invalidation_level": "Price at which thesis is wrong",
    "key_watch_levels": ["watch 1", "watch 2"],
    "contingency": "What to do if SL hits"
  }},

  "gemini_confidence_note": "1-2 sentences on the quality and completeness of available data for this call"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
#  MARKET BRIEFING — Daily market context prompt for Gemini
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_MARKET_BRIEFING_PROMPT = """Produce a concise daily market briefing for Indian equity traders.
Focus on what matters most for intraday and BTST setups TODAY.

=== TODAY'S DATA ===
Date           : {date}
Nifty 50       : {nifty_ltp} ({nifty_change}%)
Bank Nifty     : {banknifty_ltp} ({banknifty_change}%)
India VIX      : {india_vix} ({vix_change}%)
FII net today  : ₹{fii_net}Cr
DII net today  : ₹{dii_net}Cr
USD/INR        : {inr_usd}
Crude oil      : ${crude_price}/bbl
US market (prev close): {us_market}
Asian markets  : {asian_markets}

=== GLOBAL NEWS (last 12h) ===
{global_news}

=== INDIA MARKET NEWS ===
{india_news}

Produce ONLY valid JSON:
{{
  "market_mood": "RISK_ON" or "RISK_OFF" or "NEUTRAL",
  "nifty_bias": "BULLISH" or "BEARISH" or "SIDEWAYS",
  "key_levels": {{
    "nifty_support": [number, number],
    "nifty_resistance": [number, number],
    "banknifty_support": [number, number],
    "banknifty_resistance": [number, number]
  }},
  "fii_dii_interpretation": "2-sentence interpretation of today's institutional flows",
  "global_impact": "2-sentence summary of how global events affect India today",
  "sectors_to_watch": ["sector 1", "sector 2", "sector 3"],
  "sectors_to_avoid": ["sector 1"],
  "top_3_themes": ["theme 1", "theme 2", "theme 3"],
  "trader_guidance": "3-4 actionable sentences for traders today",
  "risk_warning": "Any specific risk to watch today"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
#  WATCHLIST RANKING — Multi-stock comparative ranking
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_RANK_WATCHLIST_PROMPT = """You have screened {n} stocks and need to rank them by conviction for trading today.
Consider all signals holistically. Do not just rank by composite score — also factor in
setup quality, risk/reward, liquidity, and the macro backdrop.

=== STOCKS (sorted by screener score) ===
{stocks_json}

=== MARKET CONTEXT ===
{market_context}

Respond with ONLY valid JSON:
{{
  "ranked_picks": [
    {{
      "rank": 1,
      "symbol": "SYMBOL",
      "setup_type": "INTRADAY/BTST/SWING",
      "one_liner": "Why this is the top pick today in one sentence",
      "conviction": 1-10,
      "entry": number,
      "target": number,
      "stop_loss": number
    }}
  ],
  "top_pick_rationale": "2-3 sentences on why #1 is the best trade today",
  "avoid_today": ["SYMBOL1", "SYMBOL2"],
  "avoid_reason": "Brief reason to avoid those symbols today"
}}"""
