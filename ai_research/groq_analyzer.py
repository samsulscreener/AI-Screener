"""
groq_analyzer.py
----------------
Layer 1 — Fast triage using Groq's ultra-low-latency API.

Model: llama-3.3-70b-versatile (best for structured JSON + finance)
Speed: ~1–2 seconds for a full analysis
Purpose: Quickly decide if a screener hit deserves deep Gemini research.

Groq API docs: https://console.groq.com/docs/openai
Free tier: 14,400 tokens/min, 500 req/day

pip install groq
"""

import os
import json
import time
from typing import Optional
from loguru import logger

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq package not installed. Run: pip install groq")

from .prompts import GROQ_SYSTEM, GROQ_TRIAGE_PROMPT


class GroqAnalyzer:
    """
    Layer 1 — Groq fast triage.

    Receives: Screener signals + recent news headlines
    Returns : Quick verdict + bull/bear thesis + decision to proceed to Layer 2

    Typical cost: ~0 (free tier covers ~500 analyses/day)
    """

    MODEL = "llama-3.3-70b-versatile"
    # Alternative models (uncomment to switch):
    # MODEL = "mixtral-8x7b-32768"        # Slightly faster, less accurate
    # MODEL = "llama-3.1-8b-instant"      # Fastest, lighter quality
    # MODEL = "gemma2-9b-it"              # Good for structured output

    def __init__(self, config: dict = None):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.client  = None
        self.config  = config or {}
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            logger.warning("GROQ_API_KEY not set. Layer 1 analysis will be skipped.")
            return
        if not GROQ_AVAILABLE:
            return
        try:
            self.client = Groq(api_key=self.api_key)
            logger.debug("Groq client initialized")
        except Exception as e:
            logger.error(f"Groq init failed: {e}")

    def is_available(self) -> bool:
        return bool(self.client and self.api_key)

    # ─────────────────────────────────────────────────────────────────────────
    #  Main triage method
    # ─────────────────────────────────────────────────────────────────────────

    def triage(self, screener_result: dict, news_data: dict, market_context: dict) -> dict:
        """
        Run fast Groq triage on a screener result.

        Args:
            screener_result : Full result dict from screener.scorer.Scorer.build_result()
            news_data       : Output from MarketNewsFetcher.fetch_all_news()
            market_context  : Dict with fii_net, india_vix, global_sentiment etc.

        Returns:
            dict with groq analysis + metadata
        """
        if not self.is_available():
            return self._fallback_triage(screener_result)

        prompt = self._build_prompt(screener_result, news_data, market_context)

        try:
            start = time.time()
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": GROQ_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.15,          # Low temperature for consistent JSON
                max_tokens=800,
                response_format={"type": "json_object"},  # Force JSON mode
            )
            elapsed = round(time.time() - start, 2)

            raw_json = response.choices[0].message.content
            result   = json.loads(raw_json)
            result["_layer"] = "groq"
            result["_model"] = self.MODEL
            result["_latency_sec"] = elapsed
            result["_tokens_used"] = response.usage.total_tokens

            logger.info(
                f"Groq [{screener_result['symbol']}]: "
                f"{result.get('quick_verdict')} conviction={result.get('conviction')}/10 "
                f"proceed={result.get('proceed_to_deep_research')} ({elapsed}s)"
            )
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Groq returned invalid JSON for {screener_result['symbol']}: {e}")
            return self._fallback_triage(screener_result)
        except Exception as e:
            logger.error(f"Groq triage failed for {screener_result['symbol']}: {e}")
            return self._fallback_triage(screener_result)

    # ─────────────────────────────────────────────────────────────────────────
    #  Prompt construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_prompt(self, r: dict, news: dict, ctx: dict) -> str:
        scores    = r.get("scores", {})
        tech      = r.get("technical", {})
        vol       = r.get("volume", {})
        smart     = r.get("smart_money", {})
        fund_det  = r.get("all_details", {})

        # Format news headlines
        headlines = "\n".join(
            f"  • {h}" for h in news.get("top_headlines", ["No recent headlines found"])
        )

        # Smart money details
        sm_detail = " | ".join(filter(None, [
            smart.get("fii_positive", ""),
            smart.get("bulk_deals", ""),
            smart.get("insider", ""),
        ])) or "No notable smart money activity"

        # Fundamental detail
        fund_detail = " | ".join(filter(None, [
            fund_det.get("roe", ""),
            fund_det.get("pe", ""),
            fund_det.get("announcement", ""),
        ])) or "Standard fundamentals"

        return GROQ_TRIAGE_PROMPT.format(
            symbol             = r.get("symbol", ""),
            ltp                = r.get("ltp", 0),
            composite_score    = r.get("composite_score", 0),
            setup_type         = r.get("setup_type", ""),
            sector             = r.get("sector", "Unknown"),
            smart_money_score  = scores.get("smart_money", 0),
            smart_money_detail = sm_detail,
            volume_score       = scores.get("volume", 0),
            vol_spike          = vol.get("spike_ratio", 0),
            delivery_pct       = vol.get("delivery_pct", 0),
            technical_score    = scores.get("technical", 0),
            rsi                = tech.get("rsi", "—"),
            supertrend         = "BUY ✅" if tech.get("supertrend_buy") else "SELL ❌",
            ema_aligned        = "Yes ✅" if tech.get("ema_aligned") else "No",
            news_score         = scores.get("news", 0),
            avg_sentiment      = round(news.get("avg_sentiment", 0), 3),
            fund_score         = scores.get("fundamental", 0),
            fundamental_detail = fund_detail,
            news_headlines     = headlines,
            fii_activity       = ctx.get("fii_activity", "Unknown"),
            india_vix          = ctx.get("india_vix", "Unknown"),
            global_sentiment   = ctx.get("global_sentiment", "Neutral"),
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Fallback (no API key)
    # ─────────────────────────────────────────────────────────────────────────

    def _fallback_triage(self, r: dict) -> dict:
        """
        Rule-based fallback when Groq is unavailable.
        Uses the composite score to make a simple decision.
        """
        score = r.get("composite_score", 0)
        tech  = r.get("technical", {})
        vol   = r.get("volume", {})

        verdict = (
            "STRONG BUY" if score >= 75 else
            "BUY"        if score >= 62 else
            "NEUTRAL"    if score >= 50 else
            "AVOID"
        )
        return {
            "proceed_to_deep_research":  score >= 60,
            "quick_verdict":             verdict,
            "conviction":                min(10, score // 10),
            "bull_thesis":               f"Score {score}/100 with technical alignment and volume support.",
            "bear_thesis":               "No AI analysis — using rule-based fallback.",
            "key_catalysts":             [],
            "key_risks":                 ["Groq API not configured"],
            "best_setup_type":           r.get("setup_type", "BTST"),
            "suggested_holding_period":  "1-2 days",
            "entry_strategy":            "Enter near current LTP with tight SL.",
            "institutional_interest":    "MEDIUM",
            "groq_reasoning":            "Fallback — configure GROQ_API_KEY for AI triage.",
            "_layer":                    "fallback",
            "_model":                    "rule-based",
            "_latency_sec":              0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Batch triage
    # ─────────────────────────────────────────────────────────────────────────

    def batch_triage(
        self,
        screener_results: list,
        news_map: dict,
        market_context: dict,
        rate_limit_delay: float = 0.5,
    ) -> list:
        """
        Triage multiple screener results.
        Returns list of (screener_result, groq_result) tuples,
        filtered to only those where proceed_to_deep_research = True.

        Args:
            screener_results : List of screener result dicts
            news_map         : {symbol: news_data} from MarketNewsFetcher
            market_context   : Shared market context dict
            rate_limit_delay : Seconds between Groq calls (free tier: 500 req/day)
        """
        qualified = []
        for r in screener_results:
            symbol   = r.get("symbol", "")
            news     = news_map.get(symbol, {})
            groq_out = self.triage(r, news, market_context)

            if groq_out.get("proceed_to_deep_research", False):
                qualified.append((r, groq_out))
                logger.info(f"✅ Groq qualified {symbol} for deep research")
            else:
                logger.debug(f"⏭️ Groq skipped {symbol}: {groq_out.get('quick_verdict')}")

            time.sleep(rate_limit_delay)

        logger.info(f"Groq triage: {len(qualified)}/{len(screener_results)} qualified for deep research")
        return qualified
