"""
tests/test_ai_research.py
--------------------------
Unit tests for the AI research layer.
Run: pytest tests/test_ai_research.py -v

These tests use mocking — no real API calls are made.
"""

import pytest
import json
from unittest.mock import MagicMock, patch


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_screener_result():
    return {
        "symbol":          "TATAPOWER",
        "ltp":             412.50,
        "composite_score": 82,
        "setup_type":      "BTST",
        "sector":          "Energy",
        "scores": {
            "smart_money": 85, "volume": 80,
            "technical": 90, "news": 70, "fundamental": 65,
        },
        "technical": {
            "rsi": 58.3, "supertrend_buy": True,
            "ema_aligned": True, "patterns": ["Hammer"],
        },
        "volume": {
            "spike_ratio": 3.2, "delivery_pct": 68.0,
            "oi": {"pcr": 0.95, "max_pain": 410},
        },
        "smart_money": {
            "fii_positive": "FII net buy ₹250Cr ✅",
            "bulk_deals":   "Net bulk buy ₹45Cr ✅",
            "insider":      None,
        },
        "trade_setup": {
            "entry_low": 408.0, "entry_high": 415.0,
            "target":    440.0, "stop_loss":  398.0, "rr_ratio": 2.8,
        },
        "all_details": {
            "roe": "ROE 18.5% ✅", "pe": "PE 24.2 ✅",
            "fii": "FII net buy ₹250Cr ✅",
        },
        "news": {"avg_sentiment": 0.35, "headlines": []},
    }


@pytest.fixture
def sample_news_data():
    return {
        "articles": [
            {
                "source": "marketaux", "title": "Tata Power wins 500MW solar bid",
                "description": "Tata Power secures major renewable energy order.",
                "url": "https://example.com/1", "published_at": "2025-01-15T10:00:00Z",
                "source_name": "Economic Times", "sentiment": 0.72, "relevance": 0.95,
            },
            {
                "source": "alphavantage", "title": "Power sector outlook remains strong",
                "description": "Analysts bullish on renewable energy stocks in India.",
                "url": "https://example.com/2", "published_at": "2025-01-15T08:30:00Z",
                "source_name": "Business Standard", "sentiment": 0.45, "relevance": 0.70,
            },
        ],
        "avg_sentiment":      0.62,
        "article_count":      2,
        "top_headlines":      ["[Economic Times] Tata Power wins 500MW solar bid (sentiment: +0.72)"],
        "full_articles_text": "SOURCE: Economic Times\nTITLE: Tata Power wins 500MW solar bid\n",
        "corporate_actions":  [],
    }


@pytest.fixture
def sample_market_context():
    return {
        "fii_net_cr":      250.0,
        "dii_net_cr":      180.0,
        "fii_positive":    True,
        "fii_activity":    "+₹250Cr",
        "india_vix":       13.5,
        "vix_bullish":     True,
        "global_sentiment":"Positive",
        "crude_price":     82.5,
        "inr_usd":         83.45,
    }


@pytest.fixture
def sample_groq_result():
    return {
        "proceed_to_deep_research": True,
        "quick_verdict":            "STRONG BUY",
        "conviction":               8,
        "bull_thesis":              "Strong FII buying + renewable energy tailwind + solid technical breakout.",
        "bear_thesis":              "High PE for the sector; any regulatory change could hurt margins.",
        "key_catalysts":            ["500MW solar order win", "FII accumulation", "Supertrend BUY"],
        "key_risks":                ["Regulatory risk", "High PE multiple"],
        "best_setup_type":          "BTST",
        "suggested_holding_period": "1-2 days",
        "entry_strategy":           "Buy on dips to 408-410 zone.",
        "institutional_interest":   "HIGH",
        "groq_reasoning":           "Multiple confluences: smart money, volume, news, technical all aligned.",
        "_layer":                   "groq",
        "_model":                   "llama-3.3-70b-versatile",
        "_latency_sec":             1.3,
    }


# ── MarketNewsFetcher Tests ───────────────────────────────────────────────────

class TestMarketNewsFetcher:
    def test_deduplication(self):
        from ai_research.market_news_fetcher import MarketNewsFetcher
        fetcher = MarketNewsFetcher()
        articles = [
            {"title": "Tata Power wins solar bid", "sentiment": 0.7, "relevance": 0.9,
             "source": "a", "description": "", "url": "", "published_at": "", "source_name": "ET"},
            {"title": "Tata Power wins solar bid",  "sentiment": 0.6, "relevance": 0.8,
             "source": "b", "description": "", "url": "", "published_at": "", "source_name": "MC"},
            {"title": "Power sector outlook positive", "sentiment": 0.4, "relevance": 0.7,
             "source": "c", "description": "", "url": "", "published_at": "", "source_name": "BS"},
        ]
        seen = set()
        unique = []
        for art in articles:
            k = art["title"][:50].lower().strip()
            if k not in seen and art["title"]:
                seen.add(k)
                unique.append(art)
        assert len(unique) == 2  # Duplicate removed

    def test_weighted_sentiment_calculation(self):
        articles = [
            {"sentiment": 0.8, "relevance": 1.0},
            {"sentiment": 0.2, "relevance": 0.5},
        ]
        total_weight = sum(a["relevance"] for a in articles)
        weighted = sum(a["sentiment"] * a["relevance"] for a in articles) / total_weight
        assert abs(weighted - (0.8 + 0.1) / 1.5) < 0.001

    @patch("ai_research.market_news_fetcher.requests.get")
    def test_marketaux_fetch_success(self, mock_get):
        from ai_research.market_news_fetcher import MarketNewsFetcher
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "data": [
                {
                    "title": "Tata Power Q3 profit surges",
                    "description": "Strong quarterly results",
                    "url": "https://example.com",
                    "published_at": "2025-01-15T10:00:00Z",
                    "source": "ET",
                    "relevance_score": 0.9,
                    "entities": [{"symbol": "TATAPOWER", "sentiment_score": 0.65}],
                    "highlights": [],
                }
            ]
        }
        fetcher = MarketNewsFetcher()
        fetcher.marketaux_key = "test_key"
        articles = fetcher.fetch_marketaux("TATAPOWER")
        assert len(articles) == 1
        assert articles[0]["sentiment"] == 0.65
        assert articles[0]["source"] == "marketaux"

    @patch("ai_research.market_news_fetcher.requests.get")
    def test_marketaux_api_error_handled(self, mock_get):
        from ai_research.market_news_fetcher import MarketNewsFetcher
        mock_get.return_value.status_code = 429
        mock_get.return_value.json.return_value = {"error": {"message": "Rate limit"}}
        fetcher = MarketNewsFetcher()
        fetcher.marketaux_key = "test_key"
        result = fetcher.fetch_marketaux("INFY")
        assert result == []

    def test_no_api_key_returns_empty(self):
        from ai_research.market_news_fetcher import MarketNewsFetcher
        fetcher = MarketNewsFetcher()
        fetcher.marketaux_key = ""
        result = fetcher.fetch_marketaux("RELIANCE")
        assert result == []

    def test_company_name_lookup(self):
        from ai_research.market_news_fetcher import NSE_COMPANY_NAMES
        assert "RELIANCE" in NSE_COMPANY_NAMES
        assert NSE_COMPANY_NAMES["TCS"] == "Tata Consultancy Services"
        assert NSE_COMPANY_NAMES["HDFCBANK"] == "HDFC Bank"


# ── GroqAnalyzer Tests ────────────────────────────────────────────────────────

class TestGroqAnalyzer:
    def test_fallback_triage_no_key(self, sample_screener_result):
        from ai_research.groq_analyzer import GroqAnalyzer
        groq = GroqAnalyzer()
        groq.api_key = ""
        groq.client  = None
        result = groq.triage(sample_screener_result, {}, {})
        assert result["_layer"] == "fallback"
        assert "proceed_to_deep_research" in result
        assert "quick_verdict" in result

    def test_fallback_verdict_logic(self, sample_screener_result):
        from ai_research.groq_analyzer import GroqAnalyzer
        groq = GroqAnalyzer()
        r75 = {**sample_screener_result, "composite_score": 75}
        assert groq._fallback_triage(r75)["quick_verdict"] == "STRONG BUY"
        r62 = {**sample_screener_result, "composite_score": 62}
        assert groq._fallback_triage(r62)["quick_verdict"] == "BUY"
        r45 = {**sample_screener_result, "composite_score": 45}
        assert groq._fallback_triage(r45)["quick_verdict"] == "AVOID"

    def test_prompt_builds_without_error(self, sample_screener_result, sample_news_data, sample_market_context):
        from ai_research.groq_analyzer import GroqAnalyzer
        groq = GroqAnalyzer()
        prompt = groq._build_prompt(sample_screener_result, sample_news_data, sample_market_context)
        assert "TATAPOWER" in prompt
        assert "412.5" in prompt
        assert "3.2" in prompt   # vol spike
        assert "58.3" in prompt  # RSI

    @patch("ai_research.groq_analyzer.Groq")
    def test_groq_api_success(self, MockGroq, sample_screener_result, sample_news_data, sample_market_context):
        from ai_research.groq_analyzer import GroqAnalyzer
        mock_client = MagicMock()
        MockGroq.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices[0].message.content = json.dumps({
            "proceed_to_deep_research": True,
            "quick_verdict": "BUY",
            "conviction": 7,
            "bull_thesis": "Good setup",
            "bear_thesis": "Some risk",
            "key_catalysts": ["Volume spike"],
            "key_risks": ["Market risk"],
            "best_setup_type": "BTST",
            "suggested_holding_period": "1-2 days",
            "entry_strategy": "Buy near LTP",
            "institutional_interest": "HIGH",
            "groq_reasoning": "Strong signals",
        })
        mock_client.chat.completions.create.return_value.usage.total_tokens = 500
        groq = GroqAnalyzer()
        groq.api_key = "test_key"
        groq.client  = mock_client
        result = groq.triage(sample_screener_result, sample_news_data, sample_market_context)
        assert result["quick_verdict"] == "BUY"
        assert result["proceed_to_deep_research"] is True
        assert result["_layer"] == "groq"


# ── GeminiAnalyzer Tests ──────────────────────────────────────────────────────

class TestGeminiAnalyzer:
    def test_fallback_report_structure(self, sample_screener_result, sample_groq_result):
        from ai_research.gemini_analyzer import GeminiAnalyzer
        gemini = GeminiAnalyzer()
        gemini.api_key = ""
        gemini.model   = None
        result = gemini._fallback_report(sample_screener_result, sample_groq_result)
        assert "trade_setup" in result
        assert result["trade_setup"]["entry_zone_low"] > 0
        assert result["trade_setup"]["target_1"] > 0
        assert result["trade_setup"]["stop_loss"] > 0
        assert result["_layer"] == "fallback"

    def test_prompt_builds_without_error(self, sample_screener_result, sample_groq_result,
                                          sample_news_data, sample_market_context):
        from ai_research.gemini_analyzer import GeminiAnalyzer
        gemini = GeminiAnalyzer()
        prompt = gemini._build_research_prompt(
            sample_screener_result, sample_groq_result,
            sample_news_data, sample_market_context, {}
        )
        assert "TATAPOWER" in prompt
        assert "STRONG BUY" in prompt
        assert "Hammer" in prompt
        assert len(prompt) > 500


# ── ResearchEngine Integration ────────────────────────────────────────────────

class TestResearchEngine:
    @patch("ai_research.research_engine.GeminiAnalyzer")
    @patch("ai_research.research_engine.GroqAnalyzer")
    @patch("ai_research.research_engine.MarketNewsFetcher")
    def test_research_flow(self, MockNews, MockGroq, MockGemini,
                           sample_screener_result, sample_news_data,
                           sample_groq_result):
        from ai_research.research_engine import ResearchEngine

        # Mock all three external deps
        mock_news = MagicMock()
        mock_news.fetch_all_news.return_value = sample_news_data
        mock_news.fetch_global_macro_news.return_value = []
        MockNews.return_value = mock_news

        mock_groq = MagicMock()
        mock_groq.is_available.return_value = True
        mock_groq.triage.return_value = sample_groq_result
        MockGroq.return_value = mock_groq

        mock_gemini = MagicMock()
        mock_gemini.is_available.return_value = True
        mock_gemini.deep_research.return_value = {
            "symbol": "TATAPOWER",
            "final_recommendation": "STRONG BUY",
            "conviction_score": 9,
            "setup_type": "BTST",
            "time_horizon": "1-2 days",
            "trade_setup": {
                "entry_zone_low": 408.0, "entry_zone_high": 415.0,
                "target_1": 440.0, "target_2": 460.0,
                "stop_loss": 398.0, "risk_reward": 2.8,
                "position_sizing_note": "Risk max 2%",
            },
            "executive_summary": "Strong setup with multiple confluences.",
            "bull_case": {"thesis": "Good", "key_drivers": []},
            "bear_case": {"thesis": "Risk", "key_risks": []},
            "smart_money_analysis": {"summary": "FII buying"},
            "technical_analysis": {},
            "news_catalyst_analysis": {},
            "fundamental_snapshot": {},
            "risk_management": {},
            "gemini_confidence_note": "High quality data",
            "_layer": "gemini", "_model": "gemini-2.0-flash-exp",
        }
        MockGemini.return_value = mock_gemini

        engine = ResearchEngine(config={"output": {"db_path": "/tmp/test_screener.db"}})
        engine._save_report = MagicMock()  # Skip DB write in test

        report = engine.research("TATAPOWER", sample_screener_result, force_deep=True)

        assert report["final_recommendation"] == "STRONG BUY"
        assert report["_proceeded_to_deep"] is True
        assert report["trade_setup"]["target_1"] == 440.0
        mock_groq.triage.assert_called_once()
        mock_gemini.deep_research.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
