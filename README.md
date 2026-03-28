# 🇮🇳 India Smart Stock Screener

> Institutional-grade stock screener for **Intraday**, **BTST**, and **Swing** trades on NSE/BSE — powered by smart money signals, volume analytics, news sentiment, FII/DII flows, and more.

---

## 📌 What This Screener Does

| Signal Layer | Description |
|---|---|
| 📰 **News Sentiment** | Scrapes international & domestic news, runs NLP sentiment on stocks |
| 🏦 **Smart Money** | Tracks FII/DII daily flows, bulk deals, block deals from NSE |
| 📊 **Volume Profile** | Detects unusual volume spikes, OI buildup, delivery % anomalies |
| ⭐ **Analyst Ratings** | Monitors broker upgrades/downgrades, target price revisions |
| 🤝 **Corporate Actions** | M&A, mergers, insider buying/selling, buybacks, pledging changes |
| 📈 **Technical Signals** | RSI, MACD, Supertrend, EMA crossovers, breakout patterns |
| 💡 **Options Flow** | PCR, max pain, unusual options activity (institutional footprints) |
| 🔔 **Alerts** | Telegram + Email alerts on high-conviction setups |

---

## 🗂️ Project Structure

```
india-stock-screener/
├── screener/
│   ├── __init__.py
│   ├── data_fetcher.py       # NSE/BSE + Yahoo Finance data
│   ├── news_analyzer.py      # News scraping + NLP sentiment
│   ├── smart_money.py        # FII/DII, bulk/block deals
│   ├── volume_analyzer.py    # Volume spike + delivery %
│   ├── technical_analyzer.py # RSI, MACD, Supertrend, patterns
│   ├── options_analyzer.py   # PCR, OI, max pain
│   ├── fundamental_analyzer.py # Ratings, M&A, insider trades
│   └── scorer.py             # Composite signal scorer
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── tests/
│   └── test_screener.py
├── scripts/
│   └── run_screener.sh       # Cron-ready run script
├── .github/
│   └── workflows/
│       └── screener.yml      # GitHub Actions (auto-run)
├── config.yaml               # All tunable parameters
├── main.py                   # CLI entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## ⚡ Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/india-stock-screener.git
cd india-stock-screener

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys (see API Keys section below)
```

### 3. Run Screener

```bash
# Full screen — all signals
python main.py --mode all

# Intraday only (fast, volume + technicals)
python main.py --mode intraday

# BTST screen (evening run, next-day setups)
python main.py --mode btst

# Swing (weekly setups)
python main.py --mode swing

# Launch dashboard
streamlit run dashboard/app.py
```

---

## 🔑 API Keys Required

| Service | Purpose | Free Tier |
|---|---|---|
| [NewsAPI](https://newsapi.org) | International news | ✅ 100 req/day |
| [Alpha Vantage](https://alphavantage.co) | Fundamentals + earnings | ✅ 25 req/day |
| [Telegram Bot](https://t.me/BotFather) | Alert notifications | ✅ Free |
| [Screener.in](https://screener.in) | Fundamental ratios | ✅ Free |
| [NSE India](https://nseindia.com) | FII/DII, bulk deals | ✅ Public |

> **No paid APIs are required to run the core screener.** Paid tiers improve rate limits and data depth.

---

## 📊 Signal Scoring System

Each stock gets a composite score **(0–100)**:

```
Smart Money Score     → 25 pts  (FII buying, bulk deals, insider)
Volume Score          → 20 pts  (Volume spike, delivery %, OI)
Technical Score       → 25 pts  (RSI, MACD, Supertrend, patterns)
News Sentiment Score  → 15 pts  (NLP on recent news)
Fundamental Score     → 15 pts  (Ratings, upgrades, earnings surprise)
```

**Score Thresholds:**
- `70+` → 🔴 **Strong Buy Setup** (alert sent)
- `55–69` → 🟡 **Watch** (added to watchlist)
- `<55` → ⚪ No signal

---

## ⚙️ Configuration (`config.yaml`)

```yaml
screening:
  mode: all               # intraday | btst | swing | all
  universe: nifty500      # nifty50 | nifty200 | nifty500 | custom
  min_price: 20           # Skip penny stocks
  max_price: 10000
  min_volume: 500000      # Min daily volume (shares)
  min_market_cap_cr: 500  # Min market cap in crores

signals:
  smart_money_weight: 0.25
  volume_weight: 0.20
  technical_weight: 0.25
  news_weight: 0.15
  fundamental_weight: 0.15

  # Volume thresholds
  volume_spike_multiplier: 2.5     # 2.5x avg volume = spike
  delivery_pct_threshold: 60       # >60% delivery = strong buying

  # Technical
  rsi_oversold: 35
  rsi_overbought: 65
  supertrend_period: 10
  supertrend_multiplier: 3

  # News
  news_lookback_hours: 24          # Scan news from last N hours
  sentiment_threshold: 0.3         # Min positive score

alerts:
  telegram_enabled: true
  email_enabled: false
  min_score_to_alert: 70

schedule:
  intraday_times: ["09:20", "10:00", "11:00", "13:00", "14:30"]
  btst_time: "15:10"
  swing_time: "16:00"
```

---

## 🤖 GitHub Actions — Auto Scheduler

The screener runs automatically via GitHub Actions:

- **Intraday** → Every 30 min between 9:15 AM – 3:30 PM IST (weekdays)
- **BTST** → 3:15 PM IST daily
- **Swing** → Every Friday 4 PM IST

Results are saved as GitHub Artifacts and alerts sent via Telegram.

---

## 🐳 Docker

```bash
docker-compose up -d
# Dashboard at http://localhost:8501
```

---

## 📱 Telegram Alerts

Alerts look like this:

```
🚨 HIGH CONVICTION SETUP — Score: 82/100

📌 TATAPOWER | NSE | ₹412.50
📊 Type: BTST Long Setup

✅ Smart Money: FII bought ₹245Cr today
✅ Volume: 3.2x avg | Delivery: 72%
✅ Technical: Supertrend BUY | RSI 58 | MACD crossover
✅ News: "Tata Power wins solar bid" — Positive
✅ Analyst: ICICI upgraded → ₹480 target

⚡ Entry Zone: ₹408–415
🎯 Target: ₹440 | 🛡️ SL: ₹398
```

---

## ⚠️ Disclaimer

> This tool is for **educational and research purposes only**. It does not constitute financial advice. Always do your own research before trading. The authors are not SEBI-registered advisors.

---

## 🤝 Contributing

PRs welcome! See `CONTRIBUTING.md`.
