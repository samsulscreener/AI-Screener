"""
research_engine.py
------------------
Main orchestrator that chains:
  Data → Groq (L1) → Gemini (L2) → Report → Alert

Usage:
    engine = ResearchEngine()

    # Research a single stock
    report = engine.research("TATAPOWER", screener_result, ltp=412.0)

    # Research all high-score results from screener run
    reports = engine.research_all(screener_df)

    # Daily market briefing
    briefing = engine.get_market_briefing()
"""

import os
import json
import sqlite3
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
import pandas as pd
import pytz
from loguru import logger

from .market_news_fetcher import MarketNewsFetcher
from .groq_analyzer import GroqAnalyzer
from .gemini_analyzer import GeminiAnalyzer

IST = pytz.timezone("Asia/Kolkata")


class ResearchEngine:
    def __init__(self, config: dict = None, screener=None):
        """
        Args:
            config  : Config dict (from config.yaml). Uses defaults if None.
            screener: Optional IndiaStockScreener instance for fetching context data.
                      If None, market context is fetched independently.
        """
        self.config   = config or {}
        self.screener = screener

        self.news_fetcher = MarketNewsFetcher(config)
        self.groq         = GroqAnalyzer(config)
        self.gemini       = GeminiAnalyzer(config)

        self._init_db()
        self._log_api_status()

    def _log_api_status(self):
        logger.info("─── AI Research Engine ───────────────────────────")
        logger.info(f"  Layer 1 Groq:    {'✅ Ready' if self.groq.is_available()   else '❌ No GROQ_API_KEY'}")
        logger.info(f"  Layer 2 Gemini:  {'✅ Ready' if self.gemini.is_available() else '❌ No GOOGLE_API_KEY'}")
        logger.info(f"  MarketAux News:  {'✅ Ready' if os.getenv('MARKETAUX_API_KEY') else '⚠️  No key (RSS fallback)'}")
        logger.info("──────────────────────────────────────────────────")

    # ─────────────────────────────────────────────────────────────────────────
    #  Database
    # ─────────────────────────────────────────────────────────────────────────

    def _init_db(self):
        db_path = self.config.get("output", {}).get("db_path", "data/screener.db")
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.db_path = db_path
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ai_research (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                ltp REAL,
                groq_verdict TEXT,
                groq_conviction INTEGER,
                gemini_recommendation TEXT,
                gemini_conviction INTEGER,
                setup_type TEXT,
                entry_low REAL,
                entry_high REAL,
                target_1 REAL,
                target_2 REAL,
                stop_loss REAL,
                risk_reward REAL,
                news_sentiment REAL,
                article_count INTEGER,
                executive_summary TEXT,
                full_report_json TEXT,
                timestamp TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_briefings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                market_mood TEXT,
                nifty_bias TEXT,
                trader_guidance TEXT,
                full_json TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _save_report(self, report: dict, news: dict):
        ts  = report.get("trade_setup", {})
        groq = report.get("_groq_layer", {})
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO ai_research (
                    symbol, ltp, groq_verdict, groq_conviction,
                    gemini_recommendation, gemini_conviction,
                    setup_type, entry_low, entry_high, target_1, target_2,
                    stop_loss, risk_reward, news_sentiment, article_count,
                    executive_summary, full_report_json, timestamp
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                report.get("symbol"), report.get("_ltp"),
                groq.get("quick_verdict"), groq.get("conviction"),
                report.get("final_recommendation"), report.get("conviction_score"),
                report.get("setup_type"),
                ts.get("entry_zone_low"), ts.get("entry_zone_high"),
                ts.get("target_1"), ts.get("target_2"), ts.get("stop_loss"),
                ts.get("risk_reward"),
                news.get("avg_sentiment"), news.get("article_count"),
                report.get("executive_summary", "")[:500],
                json.dumps(report),
                datetime.now(IST).isoformat(),
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save research report for {report.get('symbol')}: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Market context builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_market_context(self, fii_dii: dict = None, vix: dict = None) -> dict:
        """Build the shared market context dict passed to both AI layers."""
        ctx = {}

        # Use screener's pre-fetched data if available
        if fii_dii:
            ctx["fii_net_cr"]   = fii_dii.get("fii_net_cr", "N/A")
            ctx["dii_net_cr"]   = fii_dii.get("dii_net_cr", "N/A")
            ctx["fii_positive"] = fii_dii.get("fii_positive", False)
            fii_sign = "+" if fii_dii.get("fii_positive") else "-"
            ctx["fii_activity"] = f"FII net {fii_sign}₹{abs(fii_dii.get('fii_net_cr',0))}Cr"

        if vix:
            ctx["india_vix"]    = vix.get("vix", "N/A")
            ctx["vix_bullish"]  = vix.get("bullish_environment", True)

        ctx["global_sentiment"] = "Positive" if ctx.get("fii_positive") else "Mixed"
        ctx["crude_price"]      = os.getenv("CRUDE_PRICE_MANUAL", "N/A")  # Override via env if needed
        ctx["inr_usd"]          = os.getenv("INR_USD_MANUAL", "N/A")

        return ctx

    # ─────────────────────────────────────────────────────────────────────────
    #  Single-stock research
    # ─────────────────────────────────────────────────────────────────────────

    def research(
        self,
        symbol: str,
        screener_result: dict,
        fii_dii: dict = None,
        vix_data: dict = None,
        force_deep: bool = False,
    ) -> dict:
        """
        Full two-layer research on a single stock.

        Args:
            symbol         : NSE symbol
            screener_result: From screener.scorer.Scorer.build_result()
            fii_dii        : From SmartMoneyAnalyzer.get_fii_dii_activity()
            vix_data       : From OptionsAnalyzer.get_india_vix()
            force_deep     : Skip Groq gating, go straight to Gemini

        Returns:
            Combined research report dict
        """
        logger.info(f"🔬 Researching {symbol}...")

        # 1. Fetch news from all sources
        news_data = self.news_fetcher.fetch_all_news(symbol)
        logger.info(f"  News: {news_data['article_count']} articles | sentiment: {news_data['avg_sentiment']:+.3f}")

        # 2. Build market context
        ctx = self._build_market_context(fii_dii, vix_data)

        # 3. Layer 1 — Groq fast triage
        groq_result = self.groq.triage(screener_result, news_data, ctx)
        proceed     = force_deep or groq_result.get("proceed_to_deep_research", False)

        logger.info(f"  Groq: {groq_result.get('quick_verdict')} | conviction={groq_result.get('conviction')}/10 | proceed={proceed}")

        # 4. Layer 2 — Gemini deep research (only if Groq says proceed)
        if proceed:
            # Get additional fundamentals for Gemini
            extra_fund = self._get_extra_fundamentals(symbol)

            gemini_result = self.gemini.deep_research(
                screener_result  = screener_result,
                groq_result      = groq_result,
                news_data        = news_data,
                market_context   = ctx,
                extra_fundamental= extra_fund,
            )
        else:
            logger.info(f"  Gemini skipped (Groq verdict: {groq_result.get('quick_verdict')})")
            gemini_result = self.gemini._fallback_report(screener_result, groq_result)

        # 5. Merge both layers into final report
        final_report = {
            **gemini_result,
            "_ltp":        screener_result.get("ltp"),
            "_groq_layer": groq_result,
            "_news_data": {
                "article_count": news_data["article_count"],
                "avg_sentiment": news_data["avg_sentiment"],
                "top_headlines": news_data["top_headlines"][:5],
                "corporate_actions": news_data.get("corporate_actions", []),
            },
            "_proceeded_to_deep": proceed,
            "_timestamp": datetime.now(IST).isoformat(),
        }

        # 6. Save to database
        self._save_report(final_report, news_data)

        return final_report

    # ─────────────────────────────────────────────────────────────────────────
    #  Batch research (all screener results)
    # ─────────────────────────────────────────────────────────────────────────

    def research_all(
        self,
        screener_df: pd.DataFrame,
        screener_results_raw: list,
        fii_dii: dict = None,
        vix_data: dict = None,
        min_score: int = 60,
        max_symbols: int = 10,
    ) -> List[dict]:
        """
        Research all high-scoring stocks from a screener run.

        1. Filters to top candidates by score
        2. Batch-triages with Groq (parallel-safe)
        3. Deep-researches with Gemini (sequential, rate-limited)

        Returns list of full research reports.
        """
        if screener_df.empty:
            return []

        score_col = "Score" if "Score" in screener_df.columns else "score"
        sym_col   = "Symbol" if "Symbol" in screener_df.columns else "symbol"

        # Filter & limit
        top = screener_df[screener_df[score_col] >= min_score].head(max_symbols)
        symbols = top[sym_col].tolist()
        logger.info(f"Researching {len(symbols)} symbols (score ≥ {min_score})")

        # Build symbol → screener_result map
        result_map = {r["symbol"]: r for r in screener_results_raw if r.get("symbol") in symbols}

        # Fetch all news in parallel
        news_map = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {ex.submit(self.news_fetcher.fetch_all_news, sym): sym for sym in symbols}
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    news_map[sym] = future.result()
                except Exception as e:
                    logger.warning(f"News fetch failed for {sym}: {e}")
                    news_map[sym] = {"articles": [], "avg_sentiment": 0, "article_count": 0, "top_headlines": [], "full_articles_text": ""}

        # Build context once
        ctx = self._build_market_context(fii_dii, vix_data)

        # Layer 1: Batch Groq triage
        qualified = self.groq.batch_triage(
            screener_results = [result_map[s] for s in symbols if s in result_map],
            news_map         = news_map,
            market_context   = ctx,
        )

        # Layer 2: Sequential Gemini deep research (respect rate limits)
        reports = []
        for screener_r, groq_r in qualified:
            sym        = screener_r["symbol"]
            news       = news_map.get(sym, {})
            extra_fund = self._get_extra_fundamentals(sym)

            gemini_r = self.gemini.deep_research(
                screener_result   = screener_r,
                groq_result       = groq_r,
                news_data         = news,
                market_context    = ctx,
                extra_fundamental = extra_fund,
            )

            final = {
                **gemini_r,
                "_ltp":        screener_r.get("ltp"),
                "_groq_layer": groq_r,
                "_news_data": {
                    "article_count":    news.get("article_count", 0),
                    "avg_sentiment":    news.get("avg_sentiment", 0),
                    "top_headlines":    news.get("top_headlines", [])[:5],
                    "corporate_actions": news.get("corporate_actions", []),
                },
                "_proceeded_to_deep": True,
                "_timestamp": datetime.now(IST).isoformat(),
            }
            reports.append(final)
            self._save_report(final, news)
            time.sleep(2)  # Gemini rate limit courtesy

        logger.info(f"✅ AI Research complete: {len(reports)} reports generated")
        return reports

    # ─────────────────────────────────────────────────────────────────────────
    #  Daily Market Briefing
    # ─────────────────────────────────────────────────────────────────────────

    def get_market_briefing(self, market_data: dict = None) -> dict:
        """
        Generate + cache daily market briefing.
        Fetches global and India news, asks Gemini for interpretation.
        """
        market_data = market_data or {}

        global_news  = self.news_fetcher.fetch_global_macro_news(hours_back=12)
        india_news   = self.news_fetcher._fetch_macro_rss()

        news_bundle = {
            "global_articles": global_news,
            "india_articles":  india_news,
        }

        briefing = self.gemini.generate_market_briefing(market_data, news_bundle)

        # Save to DB
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO market_briefings (date, market_mood, nifty_bias, trader_guidance, full_json, timestamp)
                VALUES (?,?,?,?,?,?)
            """, (
                datetime.now(IST).strftime("%Y-%m-%d"),
                briefing.get("market_mood", ""),
                briefing.get("nifty_bias", ""),
                briefing.get("trader_guidance", ""),
                json.dumps(briefing),
                datetime.now(IST).isoformat(),
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not save market briefing: {e}")

        return briefing

    # ─────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_extra_fundamentals(self, symbol: str) -> dict:
        """Try to get extra fundamentals from Screener.in for Gemini context."""
        try:
            if self.screener:
                return self.screener.fundamental.get_screener_data(symbol)
        except Exception:
            pass
        return {}

    def load_recent_reports(self, limit: int = 50) -> pd.DataFrame:
        """Load recent AI research reports from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql(
                f"SELECT * FROM ai_research ORDER BY timestamp DESC LIMIT {limit}", conn
            )
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Failed to load reports: {e}")
            return pd.DataFrame()

    def get_report_json(self, symbol: str) -> Optional[dict]:
        """Retrieve the latest full JSON report for a symbol."""
        try:
            conn = sqlite3.connect(self.db_path)
            row = conn.execute(
                "SELECT full_report_json FROM ai_research WHERE symbol=? ORDER BY timestamp DESC LIMIT 1",
                (symbol,)
            ).fetchone()
            conn.close()
            return json.loads(row[0]) if row else None
        except Exception:
            return None
