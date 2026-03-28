"""
gemini_analyzer.py
------------------
Layer 2 — Deep research using Google Gemini.

Model: gemini-2.0-flash-exp (fast, long context, great for finance)
Fallback: gemini-1.5-flash-8b (if quota exhausted)

Google AI Studio: https://aistudio.google.com/app/apikey (free)
Free tier: 15 req/min, 1M tokens/min, 1500 req/day (gemini-2.0-flash)

pip install google-generativeai
"""

import os
import json
import time
from typing import Optional
from datetime import datetime
from loguru import logger

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Run: pip install google-generativeai")

from .prompts import GEMINI_SYSTEM, GEMINI_DEEP_RESEARCH_PROMPT, GEMINI_MARKET_BRIEFING_PROMPT


class GeminiAnalyzer:
    """
    Layer 2 — Comprehensive AI research using Google Gemini.

    Receives: Screener signals + Groq quick assessment + full news articles
    Returns : Complete trade research report with specific recommendations

    Typical cost: Free (15 req/min on free tier is plenty for EOD runs)
    """

    PRIMARY_MODEL  = "gemini-2.0-flash-exp"
    FALLBACK_MODEL = "gemini-1.5-flash-8b"

    # Safety settings — relax for financial content (not harmful)
    SAFETY_SETTINGS = {
        HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    } if GEMINI_AVAILABLE else {}

    GENERATION_CONFIG = {
        "temperature":        0.15,
        "top_p":              0.95,
        "top_k":              40,
        "max_output_tokens":  2048,
        "response_mime_type": "application/json",
    }

    def __init__(self, config: dict = None):
        self.api_key = os.getenviron("GOOGLE_API_KEY", "")
        self.model   = None
        self.config  = config or {}
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not set. Layer 2 deep research will be skipped.")
            return
        if not GEMINI_AVAILABLE:
            return
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=self.PRIMARY_MODEL,
                system_instruction=GEMINI_SYSTEM,
                generation_config=self.GENERATION_CONFIG,
                safety_settings=self.SAFETY_SETTINGS,
            )
            logger.debug(f"Gemini client initialized: {self.PRIMARY_MODEL}")
        except Exception as e:
            logger.error(f"Gemini init failed: {e}")

    def is_available(self) -> bool:
        return bool(self.model and self.api_key)

    # ─────────────────────────────────────────────────────────────────────────
    #  Deep research for a single stock
    # ─────────────────────────────────────────────────────────────────────────

    def deep_research(
        self,
        screener_result: dict,
        groq_result: dict,
        news_data: dict,
        market_context: dict,
        extra_fundamental: dict = None,
    ) -> dict:
        """
        Full Gemini research report.

        Args:
            screener_result   : Result from screener
            groq_result       : Output from GroqAnalyzer.triage()
            news_data         : Output from MarketNewsFetcher.fetch_all_news()
            market_context    : Global market context
            extra_fundamental : Optional extra fundamentals (Screener.in, earnings)

        Returns:
            Complete research report dict
        """
        if not self.is_available():
            return self._fallback_report(screener_result, groq_result)

        prompt = self._build_research_prompt(
            screener_result, groq_result, news_data, market_context, extra_fundamental or {}
        )

        for attempt in range(3):
            try:
                start = time.time()
                response = self.model.generate_content(prompt)
                elapsed  = round(time.time() - start, 2)

                raw_text = response.text.strip()
                # Clean any accidental markdown fences
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]

                result = json.loads(raw_text)
                result["_layer"]       = "gemini"
                result["_model"]       = self.PRIMARY_MODEL
                result["_latency_sec"] = elapsed

                logger.info(
                    f"Gemini [{screener_result['symbol']}]: "
                    f"{result.get('final_recommendation')} "
                    f"conviction={result.get('conviction_score')}/10 ({elapsed}s)"
                )
                return result

            except json.JSONDecodeError as e:
                logger.warning(f"Gemini JSON parse error (attempt {attempt+1}): {e}")
                if attempt == 2:
                    return self._fallback_report(screener_result, groq_result)
                time.sleep(1)

            except Exception as e:
                err_str = str(e)
                # Rate limit: back off and retry
                if "429" in err_str or "quota" in err_str.lower():
                    logger.warning(f"Gemini rate limit hit. Waiting 30s...")
                    time.sleep(30)
                elif "503" in err_str or "overloaded" in err_str.lower():
                    logger.warning(f"Gemini overloaded. Switching to fallback model...")
                    self._switch_to_fallback_model()
                    time.sleep(5)
                else:
                    logger.error(f"Gemini deep research failed for {screener_result['symbol']}: {e}")
                    return self._fallback_report(screener_result, groq_result)

        return self._fallback_report(screener_result, groq_result)

    # ─────────────────────────────────────────────────────────────────────────
    #  Daily market briefing
    # ─────────────────────────────────────────────────────────────────────────

    def generate_market_briefing(self, market_data: dict, news_data: dict) -> dict:
        """
        Generate a daily market briefing for Indian traders.
        Call once per day, before the screener runs.
        """
        if not self.is_available():
            return {"market_mood": "NEUTRAL", "trader_guidance": "Gemini not configured."}

        news_global = "\n".join(
            f"• {a['title']} ({a.get('source_name','')})"
            for a in news_data.get("global_articles", [])[:10]
        )
        news_india = "\n".join(
            f"• {a['title']} ({a.get('source_name','')})"
            for a in news_data.get("india_articles", [])[:10]
        )

        prompt = GEMINI_MARKET_BRIEFING_PROMPT.format(
            date              = datetime.now().strftime("%d %b %Y"),
            nifty_ltp         = market_data.get("nifty_ltp", "N/A"),
            nifty_change      = market_data.get("nifty_change_pct", "0"),
            banknifty_ltp     = market_data.get("banknifty_ltp", "N/A"),
            banknifty_change  = market_data.get("banknifty_change_pct", "0"),
            india_vix         = market_data.get("india_vix", "N/A"),
            vix_change        = market_data.get("vix_change_pct", "0"),
            fii_net           = market_data.get("fii_net_cr", "N/A"),
            dii_net           = market_data.get("dii_net_cr", "N/A"),
            inr_usd           = market_data.get("inr_usd", "N/A"),
            crude_price       = market_data.get("crude_price", "N/A"),
            us_market         = market_data.get("us_market_summary", "N/A"),
            asian_markets     = market_data.get("asian_markets_summary", "N/A"),
            global_news       = news_global or "No global news fetched.",
            india_news        = news_india  or "No India news fetched.",
        )

        try:
            response = self.model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Market briefing generation failed: {e}")
            return {"market_mood": "NEUTRAL", "trader_guidance": "Unable to generate briefing."}

    # ─────────────────────────────────────────────────────────────────────────
    #  Prompt construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_research_prompt(
        self, r: dict, groq: dict, news: dict, ctx: dict, fund: dict
    ) -> str:
        scores = r.get("scores", {})
        tech   = r.get("technical", {})
        vol    = r.get("volume", {})
        smart  = r.get("smart_money", {})
        oi     = vol.get("oi", {})
        ts     = r.get("trade_setup", {})

        # Groq context
        catalysts = "\n".join(f"  - {c}" for c in groq.get("key_catalysts", []))
        risks_l   = "\n".join(f"  - c" for c in groq.get("key_risks", []))

        # Corp actions
        corp = news.get("corporate_actions", [])
        corp_str = "\n".join(
            f"  • {c['action']} | Ex-date: {c.get('ex_date','?')} | Record: {c.get('record_date','?')}"
            for c in corp
        ) if corp else "No upcoming corporate actions found."

        return GEMINI_DEEP_RESEARCH_PROMPT.format(
            symbol             = r.get("symbol", ""),
            ltp                = r.get("ltp", 0),
            sector             = r.get("sector", "Unknown"),
            composite_score    = r.get("composite_score", 0),
            setup_type         = r.get("setup_type", ""),
            rsi                = tech.get("rsi", "N/A"),
            macd_histogram     = r.get("macd_histogram", "N/A"),
            supertrend         = "BUY" if tech.get("supertrend_buy") else "SELL",
            ema_aligned        = "Yes" if tech.get("ema_aligned") else "No",
            patterns           = ", ".join(tech.get("patterns", [])) or "None",
            vol_spike          = vol.get("spike_ratio", "N/A"),
            delivery_pct       = vol.get("delivery_pct", "N/A"),
            pcr                = oi.get("pcr", "N/A"),
            max_pain           = oi.get("max_pain", "N/A"),
            fii_net            = ctx.get("fii_net_cr", "N/A"),
            fii_direction      = "Buying" if ctx.get("fii_positive") else "Selling",
            dii_net            = ctx.get("dii_net_cr", "N/A"),
            bulk_deals         = smart.get("bulk_deals") or "None noted",
            insider_activity   = smart.get("insider") or "None noted",
            promoter_holding   = fund.get("promoter_holding", "N/A"),
            roe                = fund.get("roe", "N/A"),
            pe                 = fund.get("pe", "N/A"),
            debt_equity        = fund.get("debt_equity", "N/A"),
            roce               = fund.get("roce", "N/A"),
            announcements      = corp_str,
            earnings_surprise  = fund.get("earnings_surprise", "N/A"),
            full_news_articles = news.get("full_articles_text", "No articles fetched."),
            groq_verdict       = groq.get("quick_verdict", "N/A"),
            groq_conviction    = groq.get("conviction", "N/A"),
            groq_bull          = groq.get("bull_thesis", "N/A"),
            groq_bear          = groq.get("bear_thesis", "N/A"),
            groq_catalysts     = catalysts or "None identified",
            groq_risks         = risks_l or "None identified",
            india_vix          = ctx.get("india_vix", "N/A"),
            nifty_trend        = ctx.get("nifty_trend", "N/A"),
            sector_trend       = ctx.get("sector_trend", "N/A"),
            fii_ytd            = ctx.get("fii_ytd", "N/A"),
            global_markets     = ctx.get("global_markets", "N/A"),
            crude_price        = ctx.get("crude_price", "N/A"),
            inr_usd            = ctx.get("inr_usd", "N/A"),
        )

    def _switch_to_fallback_model(self):
        try:
            self.model = genai.GenerativeModel(
                model_name=self.FALLBACK_MODEL,
                system_instruction=GEMINI_SYSTEM,
                generation_config=self.GENERATION_CONFIG,
                safety_settings=self.SAFETY_SETTINGS,
            )
            logger.info(f"Switched to fallback model: {self.FALLBACK_MODEL}")
        except Exception as e:
            logger.error(f"Failed to switch to fallback model: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Fallback report (when Gemini unavailable)
    # ─────────────────────────────────────────────────────────────────────────

    def _fallback_report(self, r: dict, groq: dict) -> dict:
        ts  = r.get("trade_setup", {})
        ltp = r.get("ltp", 0)
        return {
            "symbol":             r.get("symbol"),
            "report_timestamp":   datetime.now().isoformat(),
            "final_recommendation": groq.get("quick_verdict", "HOLD"),
            "conviction_score":   groq.get("conviction", 5),
            "setup_type":         groq.get("best_setup_type", r.get("setup_type", "BTST")),
            "time_horizon":       groq.get("suggested_holding_period", "1-2 days"),
            "trade_setup": {
                "entry_zone_low":  ts.get("entry_low", ltp * 0.995),
                "entry_zone_high": ts.get("entry_high", ltp * 1.005),
                "target_1":        ts.get("target", ltp * 1.03),
                "target_2":        ts.get("target", ltp * 1.06),
                "stop_loss":       ts.get("stop_loss", ltp * 0.975),
                "risk_reward":     ts.get("rr_ratio", 2.0),
                "position_sizing_note": "Risk max 2% of capital",
            },
            "executive_summary": groq.get("bull_thesis", "No Gemini analysis — configure GOOGLE_API_KEY."),
            "bull_case": {
                "thesis": groq.get("bull_thesis", ""),
                "key_drivers": groq.get("key_catalysts", []),
            },
            "bear_case": {
                "thesis": groq.get("bear_thesis", ""),
                "key_risks": groq.get("key_risks", []),
            },
            "gemini_confidence_note": "Fallback report — GOOGLE_API_KEY not configured.",
            "_layer": "fallback",
            "_model": "rule-based",
        }
