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

from .prompts import (
    GEMINI_SYSTEM,
    GEMINI_DEEP_RESEARCH_PROMPT,
    GEMINI_MARKET_BRIEFING_PROMPT,
)


class GeminiAnalyzer:
    """
    Layer 2 — Deep research using Google Gemini (stable version)
    """

    # ✅ FIXED MODELS (stable + supported)
    PRIMARY_MODEL = "gemini-1.5-flash"
    FALLBACK_MODEL = "gemini-1.5-flash"

    SAFETY_SETTINGS = {
        HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    } if GEMINI_AVAILABLE else {}

    GENERATION_CONFIG = {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
        # ❌ Removed strict JSON mode (unstable)
    }

    def __init__(self, config: dict = None):
        self.api_key = os.environ.get("GOOGLE_API_KEY", "")
        self.model = None
        self.config = config or {}
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not set. Layer 2 skipped.")
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

            logger.debug(f"Gemini initialized: {self.PRIMARY_MODEL}")

        except Exception as e:
            logger.error(f"Gemini init failed: {e}")
            self.model = None

    def is_available(self) -> bool:
        return bool(self.model and self.api_key)

    # ─────────────────────────────────────────────
    # Deep research
    # ─────────────────────────────────────────────

    def deep_research(
        self,
        screener_result: dict,
        groq_result: dict,
        news_data: dict,
        market_context: dict,
        extra_fundamental: dict = None,
    ) -> dict:

        if not self.is_available():
            return self._fallback_report(screener_result, groq_result)

        prompt = self._build_research_prompt(
            screener_result,
            groq_result,
            news_data,
            market_context,
            extra_fundamental or {},
        )

        for attempt in range(3):
            try:
                start = time.time()

                response = self.model.generate_content(prompt)

                if not response or not hasattr(response, "text"):
                    raise ValueError("Empty response from Gemini")

                raw_text = response.text.strip()
                elapsed = round(time.time() - start, 2)

                # ✅ SAFE JSON PARSE
                try:
                    result = json.loads(raw_text)
                except:
                    logger.warning("Gemini returned non-JSON, using fallback parse")
                    result = {
                        "final_recommendation": "HOLD",
                        "conviction_score": 5,
                        "raw_output": raw_text,
                    }

                result["_layer"] = "gemini"
                result["_model"] = self.PRIMARY_MODEL
                result["_latency_sec"] = elapsed

                logger.info(
                    f"Gemini [{screener_result.get('symbol')}]: "
                    f"{result.get('final_recommendation')} "
                    f"conviction={result.get('conviction_score')}/10 ({elapsed}s)"
                )

                return result

            except Exception as e:
                err = str(e)

                if "429" in err:
                    logger.warning("Rate limit hit. Waiting 30s...")
                    time.sleep(30)

                elif "503" in err:
                    logger.warning("Model overloaded. Retrying...")
                    time.sleep(5)

                else:
                    logger.error(f"Gemini failed: {e}")
                    return self._fallback_report(screener_result, groq_result)

        return self._fallback_report(screener_result, groq_result)

    # ─────────────────────────────────────────────
    # Market briefing
    # ─────────────────────────────────────────────

    def generate_market_briefing(self, market_data: dict, news_data: dict) -> dict:

        if not self.is_available():
            return {
                "market_mood": "NEUTRAL",
                "trader_guidance": "Gemini not configured.",
            }

        try:
            prompt = GEMINI_MARKET_BRIEFING_PROMPT.format(
                date=datetime.now().strftime("%d %b %Y"),
                nifty_ltp=market_data.get("nifty_ltp"),
                nifty_change=market_data.get("nifty_change_pct"),
                india_vix=market_data.get("india_vix"),
            )

            response = self.model.generate_content(prompt)

            return json.loads(response.text)

        except Exception as e:
            logger.error(f"Market briefing failed: {e}")
            return {
                "market_mood": "NEUTRAL",
                "trader_guidance": "Fallback mode",
            }

    # ─────────────────────────────────────────────
    # Prompt builder
    # ─────────────────────────────────────────────

    def _build_research_prompt(self, r, groq, news, ctx, fund):

        # ✅ FIXED BUG (important)
        risks = "\n".join(f"- {c}" for c in groq.get("key_risks", []))
        catalysts = "\n".join(f"- {c}" for c in groq.get("key_catalysts", []))

        return GEMINI_DEEP_RESEARCH_PROMPT.format(
            symbol=r.get("symbol"),
            ltp=r.get("ltp"),
            composite_score=r.get("composite_score"),
            setup_type=r.get("setup_type"),
            rsi=r.get("technical", {}).get("rsi"),
            catalysts=catalysts or "None",
            risks=risks or "None",
            global_markets=ctx.get("global_markets"),
        )

    # ─────────────────────────────────────────────
    # Fallback
    # ─────────────────────────────────────────────

    def _fallback_report(self, r: dict, groq: dict) -> dict:

        ltp = r.get("ltp", 0)

        return {
            "symbol": r.get("symbol"),
            "final_recommendation": groq.get("quick_verdict", "HOLD"),
            "conviction_score": groq.get("conviction", 5),
            "entry": ltp,
            "target": ltp * 1.03,
            "stop_loss": ltp * 0.97,
            "_layer": "fallback",
        }
