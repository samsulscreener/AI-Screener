"""
tests/test_screener.py
----------------------
Unit tests for the India Smart Stock Screener.
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Sample config fixture ─────────────────────────────────────────────────────

@pytest.fixture
def config():
    return {
        "screening": {
            "mode": "all",
            "universe": "nifty50",
            "custom_symbols": [],
            "min_price": 20,
            "max_price": 15000,
            "min_volume": 100000,
            "min_market_cap_cr": 100,
        },
        "signals": {
            "weights": {
                "smart_money": 0.25,
                "volume": 0.20,
                "technical": 0.25,
                "news": 0.15,
                "fundamental": 0.15,
            },
            "volume": {
                "spike_multiplier": 2.5,
                "delivery_pct_min": 55,
                "oi_buildup_pct": 15,
            },
            "technical": {
                "rsi_period": 14,
                "rsi_buy_zone": [35, 65],
                "rsi_oversold": 30,
                "rsi_overbought": 75,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "supertrend_period": 10,
                "supertrend_multiplier": 3.0,
                "ema_periods": [9, 21, 50, 200],
                "bb_period": 20,
                "bb_std": 2,
            },
            "smart_money": {
                "fii_net_buy_threshold_cr": 100,
                "bulk_deal_min_value_cr": 10,
                "insider_window_days": 30,
            },
            "news": {
                "lookback_hours": 24,
                "positive_threshold": 0.25,
                "negative_threshold": -0.25,
            },
            "fundamental": {
                "min_roe": 12,
                "max_pe": 60,
                "max_debt_equity": 2.0,
                "earnings_surprise_pct": 5,
            },
        },
        "scoring": {
            "strong_buy_threshold": 70,
            "watch_threshold": 55,
        },
        "output": {
            "save_to_csv": False,
            "save_to_db": False,
        },
        "alerts": {
            "telegram": {"enabled": False, "min_score": 70},
            "email": {"enabled": False},
        },
        "logging": {"level": "WARNING"},
    }


# ── Sample OHLCV fixture ──────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv():
    """Generate 250 days of synthetic OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=250, freq="B")
    close = 500.0 + np.cumsum(np.random.randn(250) * 3)
    close = np.maximum(close, 10)
    high  = close * (1 + np.abs(np.random.randn(250) * 0.01))
    low   = close * (1 - np.abs(np.random.randn(250) * 0.01))
    open_ = close * (1 + np.random.randn(250) * 0.005)
    volume = np.random.randint(500_000, 3_000_000, size=250).astype(float)
    # Inject a volume spike on last day
    volume[-1] = volume[:-1].mean() * 3.5

    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates)


# ── TechnicalAnalyzer Tests ───────────────────────────────────────────────────

class TestTechnicalAnalyzer:
    def test_rsi_range(self, config, sample_ohlcv):
        from screener.technical_analyzer import TechnicalAnalyzer
        ta = TechnicalAnalyzer(config)
        rsi = ta.rsi(sample_ohlcv["Close"].values)
        assert all(0 <= v <= 100 for v in rsi), "RSI values must be in [0, 100]"

    def test_macd_shape(self, config, sample_ohlcv):
        from screener.technical_analyzer import TechnicalAnalyzer
        ta = TechnicalAnalyzer(config)
        close = sample_ohlcv["Close"].values
        macd = ta.macd(close)
        assert "macd" in macd and "signal" in macd and "histogram" in macd
        assert len(macd["macd"]) == len(close)

    def test_supertrend_direction(self, config, sample_ohlcv):
        from screener.technical_analyzer import TechnicalAnalyzer
        ta = TechnicalAnalyzer(config)
        df = ta.supertrend(sample_ohlcv)
        assert "st_direction" in df.columns
        assert set(df["st_direction"].unique()).issubset({1, -1, 0})

    def test_score_returns_dict(self, config, sample_ohlcv):
        from screener.technical_analyzer import TechnicalAnalyzer
        ta = TechnicalAnalyzer(config)
        result = ta.score("TEST", sample_ohlcv)
        assert isinstance(result, dict)
        assert "score" in result
        assert 0 <= result["score"] <= 100

    def test_bollinger_bands(self, config, sample_ohlcv):
        from screener.technical_analyzer import TechnicalAnalyzer
        ta = TechnicalAnalyzer(config)
        bb = ta.bollinger_bands(sample_ohlcv["Close"].values)
        assert "upper" in bb and "lower" in bb
        # Upper must always >= lower (after warmup period)
        mask = ~np.isnan(bb["upper"])
        assert all(bb["upper"][mask] >= bb["lower"][mask])

    def test_pattern_detection(self, config):
        from screener.technical_analyzer import TechnicalAnalyzer
        ta = TechnicalAnalyzer(config)

        # Construct a clear bullish engulfing
        df = pd.DataFrame({
            "Open":  [110, 105, 100, 102, 105],
            "High":  [115, 112, 107, 108, 115],
            "Low":   [105, 100,  96, 100, 104],
            "Close": [108, 102,  98, 106, 114],
            "Volume": [100_000] * 5,
        })
        result = ta.detect_patterns(df)
        assert isinstance(result["patterns"], list)
        assert "bullish_pattern" in result


# ── VolumeAnalyzer Tests ──────────────────────────────────────────────────────

class TestVolumeAnalyzer:
    def test_volume_spike_detected(self, config, sample_ohlcv):
        from screener.volume_analyzer import VolumeAnalyzer
        session = MagicMock()
        va = VolumeAnalyzer(session, config)
        result = va.analyze_volume("TEST", sample_ohlcv)
        # Last bar has 3.5x spike → should detect
        assert result["spike_ratio"] >= 2.5
        assert result["score"] > 0

    def test_score_structure(self, config, sample_ohlcv):
        from screener.volume_analyzer import VolumeAnalyzer
        session = MagicMock()
        session.get.return_value.json.return_value = {}
        va = VolumeAnalyzer(session, config)

        with patch.object(va, "get_delivery_pct", return_value=65.0), \
             patch.object(va, "get_oi_data", return_value={}):
            result = va.score("TEST", sample_ohlcv)
        assert 0 <= result["score"] <= 100

    def test_max_pain_calculation(self, config):
        from screener.volume_analyzer import VolumeAnalyzer
        session = MagicMock()
        va = VolumeAnalyzer(session, config)
        strikes = {
            100: {"ce_oi": 5000, "pe_oi": 1000},
            105: {"ce_oi": 8000, "pe_oi": 3000},
            110: {"ce_oi": 3000, "pe_oi": 6000},
        }
        mp = va._calc_max_pain(strikes)
        assert mp in strikes


# ── Scorer Tests ──────────────────────────────────────────────────────────────

class TestScorer:
    def test_composite_score_bounds(self, config):
        from screener.scorer import Scorer
        sc = Scorer(config)
        for _ in range(20):
            s = sc.composite_score(
                sm_score=np.random.randint(0, 100),
                vol_score=np.random.randint(0, 100),
                tech_score=np.random.randint(0, 100),
                news_score=np.random.randint(0, 100),
                fund_score=np.random.randint(0, 100),
            )
            assert 0 <= s <= 100

    def test_trade_setup_generation(self, config):
        from screener.scorer import Scorer
        sc = Scorer(config)
        tech = {"rsi": 55, "supertrend_buy": True, "ema_aligned": True, "patterns": ["Hammer"]}
        vol  = {"spike_ratio": 3.2}
        setup = sc.generate_trade_setup("RELIANCE", 2800.0, tech, vol)
        assert setup["target"] > 2800.0
        assert setup["stop_loss"] < 2800.0
        assert setup["rr_ratio"] > 0

    def test_build_result_complete(self, config):
        from screener.scorer import Scorer
        sc = Scorer(config)
        result = sc.build_result(
            symbol="TATAPOWER",
            ltp=412.0,
            sm_result={"score": 75, "details": {"fii": "FII net buy ₹250Cr ✅"}},
            vol_result={"score": 80, "spike_ratio": 3.2, "delivery_pct": 68, "oi": {}, "details": {}},
            tech_result={"score": 85, "rsi": 58.0, "supertrend_buy": True,
                         "ema_aligned": True, "patterns": ["Hammer"], "details": {}},
            news_result={"score": 70, "avg_sentiment": 0.35, "headlines": [], "details": {}},
            fund_result={"score": 65, "ratios": {}, "details": {}},
        )
        assert result["symbol"] == "TATAPOWER"
        assert result["composite_score"] >= 55
        assert "trade_setup" in result
        assert result["trade_setup"]["target"] > 412.0

    def test_classify_setups(self, config):
        from screener.scorer import Scorer
        sc = Scorer(config)
        # High volume spike → INTRADAY
        setup = sc.classify_setup(75, {"ema_aligned": True, "patterns": []}, {"spike_ratio": 4.0})
        assert setup == "INTRADAY"

    def test_to_dataframe(self, config):
        from screener.scorer import Scorer
        sc = Scorer(config)
        results = [
            sc.build_result(
                "TESTCO", 100.0,
                {"score": 70, "details": {}},
                {"score": 60, "spike_ratio": 2.0, "delivery_pct": 55, "oi": {}, "details": {}},
                {"score": 80, "rsi": 50.0, "supertrend_buy": True, "ema_aligned": True, "patterns": [], "details": {}},
                {"score": 65, "avg_sentiment": 0.2, "headlines": [], "details": {}},
                {"score": 60, "ratios": {}, "details": {}},
            )
        ]
        df = sc.to_dataframe(results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "Symbol" in df.columns
        assert "Score" in df.columns


# ── NewsAnalyzer Tests ────────────────────────────────────────────────────────

class TestNewsAnalyzer:
    def test_vader_score_range(self, config):
        from screener.news_analyzer import NewsAnalyzer
        na = NewsAnalyzer(config)
        assert -1.0 <= na._vader_score("Stock hits record high on strong earnings") <= 1.0
        assert -1.0 <= na._vader_score("Company faces fraud allegations, CEO resigns") <= 1.0

    def test_symbol_matching(self, config):
        from screener.news_analyzer import NewsAnalyzer
        na = NewsAnalyzer(config)
        articles = [
            {"title": "Reliance Industries Q3 profit up 18%", "summary": "", "source": "ET"},
            {"title": "TCS wins $200M cloud deal", "summary": "", "source": "BS"},
            {"title": "Gold prices surge on Fed uncertainty", "summary": "", "source": "Reuters"},
        ]
        reliance_arts = na._filter_articles_for_symbol("RELIANCE", articles)
        assert len(reliance_arts) == 1
        tcs_arts = na._filter_articles_for_symbol("TCS", articles)
        assert len(tcs_arts) == 1

    def test_score_structure(self, config):
        from screener.news_analyzer import NewsAnalyzer
        na = NewsAnalyzer(config)
        with patch.object(na, "_fetch_rss_articles", return_value=[
            {"title": "Infosys beats earnings estimate by 12%", "summary": "Strong quarter",
             "source": "ET", "published": None},
        ]):
            result = na.score("INFY")
        assert 0 <= result["score"] <= 100
        assert "avg_sentiment" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
