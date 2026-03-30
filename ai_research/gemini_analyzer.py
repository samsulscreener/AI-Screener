import os
import json
import time
from datetime import datetime
from loguru import logger

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Gemini not installed")

from .prompts import (
    GEMINI_SYSTEM,
    GEMINI_DEEP_RESEARCH_PROMPT,
    GEMINI_MARKET_BRIEFING_PROMPT,
)


class GeminiAnalyzer:

    MODEL = "gemini-1.5-flash"

    GENERATION_CONFIG = {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_output_tokens": 1500,
    }

    def __init__(self, config=None):
        self.api_key = os.environ.get("GOOGLE_API_KEY", "")
        self.model = None
        self.config = config or {}
        self._init_client()

    # -------------------------------------------------- #
    # INIT
    # -------------------------------------------------- #

    def _init_client(self):
        if not self.api_key or not GEMINI_AVAILABLE:
            logger.warning("Gemini not available or API key missing")
            return

        try:
            genai.configure(api_key=self.api_key)

            self.model = genai.GenerativeModel(
                model_name=self.MODEL,
                system_instruction=GEMINI_SYSTEM,
                generation_config=self.GENERATION_CONFIG,
            )

            logger.debug(f"Gemini ready: {self.MODEL}")

        except Exception as e:
            logger.error(f"Gemini init failed: {e}")
            self.model = None

    def is_available(self):
        return self.model is not None

    # -------------------------------------------------- #
    # MAIN ANALYSIS
    # -------------------------------------------------- #

    def deep_research(self, screener, groq, news, ctx, fund=None):

        # If Gemini not available → fallback
        if not self.is_available():
            return self._fallback(screener, groq)

        prompt = self._build_prompt(screener, groq, ctx)

        for _ in range(2):
            try:
                start = time.time()

                resp = self.model.generate_content(prompt)

                if not resp or not hasattr(resp, "text"):
                    raise ValueError("Empty response")

                raw = resp.text.strip()

                parsed = self._parse_response(raw)

                parsed.update({
                    "symbol": screener.get("symbol"),
                    "setup_type": screener.get("setup_type"),
                    "_ltp": screener.get("ltp", 0),
                    "_latency": round(time.time() - start, 2),
                    "_layer": "gemini",
                })

                logger.info(
                    f"{screener.get('symbol')} → {parsed['final_recommendation']} "
                    f"({parsed['conviction_score']}/10)"
                )

                return parsed

            except Exception as e:
                logger.warning(f"Gemini retry: {e}")
                time.sleep(2)

        return self._fallback(screener, groq)

    # -------------------------------------------------- #
    # PARSER
    # -------------------------------------------------- #

    def _parse_response(self, text):

        try:
            data = json.loads(text)
        except:
            data = {}

        rec = str(data.get("final_recommendation", "")).upper()
        if rec not in ["BUY", "SELL", "HOLD"]:
            rec = "HOLD"

        try:
            conviction = int(data.get("conviction_score", 5))
        except:
            conviction = 5

        conviction = max(1, min(10, conviction))

        return {
            "final_recommendation": rec,
            "conviction_score": conviction,
            "entry": data.get("entry"),
            "target": data.get("target"),
            "stop_loss": data.get("stop_loss"),
            "summary": text[:300],
        }

    # -------------------------------------------------- #
    # PROMPT BUILDER
    # -------------------------------------------------- #

    def _build_prompt(self, r, groq, ctx):

        risks = "\n".join(f"- {c}" for c in groq.get("key_risks", []))
        cats = "\n".join(f"- {c}" for c in groq.get("key_catalysts", []))

        return GEMINI_DEEP_RESEARCH_PROMPT.format(
            symbol=r.get("symbol"),
            ltp=r.get("ltp"),
            score=r.get("composite_score"),
            setup=r.get("setup_type"),
            rsi=r.get("technical", {}).get("rsi"),
            risks=risks or "None",
            catalysts=cats or "None",
            global_markets=ctx.get("global_markets", "Neutral"),
        )

    # -------------------------------------------------- #
    # MARKET BRIEFING
    # -------------------------------------------------- #

    def generate_market_briefing(self, market_data, news):

        if not self.is_available():
            return {"market_mood": "NEUTRAL"}

        try:
            prompt = GEMINI_MARKET_BRIEFING_PROMPT.format(
                date=datetime.now().strftime("%d %b"),
                nifty_ltp=market_data.get("nifty_ltp", "NA"),
                nifty_change=market_data.get("nifty_change_pct", 0),
                india_vix=market_data.get("india_vix", "NA"),
            )

            resp = self.model.generate_content(prompt)

            return self._parse_response(resp.text)

        except Exception as e:
            logger.warning(f"Briefing failed: {e}")
            return {"market_mood": "NEUTRAL"}

    # -------------------------------------------------- #
    # FALLBACK (CORE)
    # -------------------------------------------------- #

    def _fallback(self, r, groq):

        ltp = r.get("ltp", 0)

        return {
            "symbol": r.get("symbol"),
            "final_recommendation": groq.get("quick_verdict", "HOLD"),
            "conviction_score": groq.get("conviction", 5),
            "setup_type": r.get("setup_type", "NO_SIGNAL"),
            "entry": ltp,
            "target": round(ltp * 1.03, 2) if ltp else None,
            "stop_loss": round(ltp * 0.97, 2) if ltp else None,
            "trade_setup": r.get("trade_setup", {}),
            "summary": groq.get("bull_thesis", "Fallback analysis"),
            "_ltp": ltp,
            "_layer": "fallback",
        }

    # -------------------------------------------------- #
    # FALLBACK REPORT (CRITICAL FIX)
    # -------------------------------------------------- #

    def _fallback_report(self, screener_result, groq_result):
        return self._fallback(screener_result, groq_result)
