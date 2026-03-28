# 🧠 AI Research Layer — Groq + Gemini + MarketNews

> Drop this into your existing `india-stock-screener` repo.
> Two-layer AI analysis turns screener signals into actionable trade briefs.

---

## Architecture

```
Screener signals  ──┐
MarketAux news    ──┤──▶  Groq L1 (Llama 3.3 70B, ~1s)  ──▶  Gemini L2 (2.0 Flash, ~6s)  ──▶  Report
NSE corporate     ──┘         Fast triage                       Deep research
Alpha Vantage                 Bull/bear verdict                 Entry / Target / SL
                              Proceed? yes/no                   Catalyst analysis
                                                                Smart money view
```

---

## New Files to Add to Your Repo

```
india-stock-screener/
├── ai_research/
│   ├── __init__.py
│   ├── prompts.py              ← All LLM prompt templates
│   ├── market_news_fetcher.py  ← MarketAux + Alpha Vantage + NSE news
│   ├── groq_analyzer.py        ← Layer 1: Llama 3.3 70B triage
│   ├── gemini_analyzer.py      ← Layer 2: Gemini 2.0 Flash deep research
│   ├── research_engine.py      ← Main orchestrator
│   └── alert_formatter.py      ← Telegram/email formatter for AI reports
├── dashboard/
│   └── research_dashboard.py   ← Streamlit AI research dashboard
├── research_main.py            ← CLI entrypoint
├── requirements_ai.txt         ← Additional dependencies
└── .env.example                ← Updated with new API keys
```

---

## API Keys — All Free Tiers

| API | Purpose | Free Tier | Sign Up |
|---|---|---|---|
| **Groq** | Layer 1 LLM (Llama 3.3 70B) | 500 req/day | [console.groq.com](https://console.groq.com) |
| **Google AI** | Layer 2 LLM (Gemini 2.0 Flash) | 1,500 req/day | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| **MarketAux** | Stock news + built-in sentiment | 100 req/day | [marketaux.com](https://www.marketaux.com) |

> **Total cost to run: ₹0**

---

## Setup

### 1. Install new dependencies
```bash
pip install groq google-generativeai
# Or: pip install -r requirements_ai.txt
```

### 2. Add API keys to `.env`
```bash
GROQ_API_KEY=gsk_...        # From console.groq.com
GOOGLE_API_KEY=AIza...       # From aistudio.google.com
MARKETAUX_API_KEY=...        # From marketaux.com
```

### 3. Run

```bash
# Research specific stocks
python research_main.py --symbols TATAPOWER RELIANCE

# Research top results from today's screener run
python research_main.py --from-screener --mode btst

# Daily market briefing only
python research_main.py --briefing-only

# Force all through Gemini (skip Groq gating)
python research_main.py --symbols BAJFINANCE --force-deep

# Launch research dashboard
streamlit run dashboard/research_dashboard.py
```

---

## What Each Layer Does

### Layer 1 — Groq (Llama 3.3 70B) ~1-2 seconds
Receives screener signals + recent headlines. Produces:
- `quick_verdict`: STRONG BUY / BUY / NEUTRAL / AVOID
- `conviction`: 1–10
- `bull_thesis` + `bear_thesis`
- `proceed_to_deep_research`: true/false (gates Layer 2)

### Layer 2 — Gemini 2.0 Flash ~5-8 seconds
Only runs when Groq says proceed. Receives everything + full article text. Produces:
- `final_recommendation` with detailed rationale
- Specific entry zone, two targets, stop-loss, R/R
- News catalyst analysis
- Smart money interpretation
- Full risk management section
- `conviction_score` 1–10

---

## MarketAux API — What it Returns

MarketAux is purpose-built for financial news. Each article includes:

```json
{
  "title": "Tata Power wins 500MW solar tender",
  "description": "...",
  "published_at": "2025-01-15T10:00:00Z",
  "source": "Economic Times",
  "relevance_score": 0.95,
  "entities": [
    {
      "symbol": "TATAPOWER",
      "name": "Tata Power Company Limited",
      "country": "in",
      "sentiment_score": 0.72
    }
  ]
}
```

**Key advantage**: Sentiment is pre-computed per entity, not just per article. So even if an article mentions 5 companies, you get the sentiment specifically for your stock.

---

## Telegram Alert Format

```
🔴🔴 STRONG BUY — Conviction: 9/10

📌 TATAPOWER | ₹412.50 | BTST | 1-2 days

Strong FII buying + 500MW solar order catalyst + technical breakout above 20-EMA.

🏦 Smart Money: FII net buy ₹250Cr + bulk deal ₹45Cr. Institutional accumulation confirmed.

📈 Trade Setup:
Entry: ₹408 – ₹415
Target 1: ₹440 | Target 2: ₹460
Stop Loss: ₹398 | R/R: 2.8x
Risk max 2% of capital

📰 News (2 articles | sentiment: +0.62):
  • [Economic Times] Tata Power wins 500MW solar bid (+0.72)
  • [BS] Power sector outlook remains strong (+0.45)

✅ Bull drivers:
  • 500MW solar order win
  • FII accumulation
  • Supertrend BUY signal

⚠️ Key risks:
  • Regulatory changes
  • High PE multiple

⚡ Groq: Multiple confluences aligned — institutional setup confirmed.
```

---

## Disclaimer

> For educational and research purposes only. Not financial advice. 
> Always do your own research. The authors are not SEBI-registered advisors.
