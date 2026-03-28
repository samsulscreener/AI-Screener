"""
scorer.py
---------
Combines all signal scores into a final composite score,
generates trade setup (entry, target, SL), and categorizes
the setup as Intraday / BTST / Swing.
"""

import math
import pandas as pd
from loguru import logger
from typing import Optional
from datetime import datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")


class Scorer:
    def __init__(self, config: dict):
        w = config["signals"]["weights"]
        self.weights = {
            "smart_money": w.get("smart_money", 0.25),
            "volume":      w.get("volume", 0.20),
            "technical":   w.get("technical", 0.25),
            "news":        w.get("news", 0.15),
            "fundamental": w.get("fundamental", 0.15),
        }
        self.strong_buy_thresh = config["scoring"]["strong_buy_threshold"]
        self.watch_thresh      = config["scoring"]["watch_threshold"]

    def composite_score(
        self,
        sm_score: int,
        vol_score: int,
        tech_score: int,
        news_score: int,
        fund_score: int,
    ) -> int:
        """Weighted composite score (0–100)."""
        score = (
            sm_score   * self.weights["smart_money"] +
            vol_score  * self.weights["volume"] +
            tech_score * self.weights["technical"] +
            news_score * self.weights["news"] +
            fund_score * self.weights["fundamental"]
        )
        return int(round(score))

    def classify_setup(self, score: int, tech: dict, vol: dict) -> str:
        """
        Classify the trade type based on signals.
        Intraday: high volume spike + technical in momentum
        BTST: moderate volume + reversal pattern detected
        Swing: fundamentals + EMA aligned + moderate volume
        """
        spike_ratio = vol.get("spike_ratio", 1)
        has_pattern = len(tech.get("patterns", [])) > 0
        ema_aligned = tech.get("ema_aligned", False)
        supertrend  = tech.get("supertrend_buy", False)

        if spike_ratio >= 3.0 and score >= 65:
            return "INTRADAY"
        elif has_pattern and score >= 60:
            return "BTST"
        elif ema_aligned and supertrend and score >= 55:
            return "SWING"
        elif score >= self.watch_thresh:
            return "WATCH"
        return "NO_SIGNAL"

    def generate_trade_setup(self, symbol: str, ltp: float, tech: dict, vol: dict) -> dict:
        """
        Generate entry zone, target, and stop-loss.

        Strategy:
          - Entry: LTP or slight pullback (0.3–0.5% below)
          - Target: Based on ATR or nearest resistance (approx 3–6%)
          - SL: ATR-based (approx 1.5–2% below entry)
        """
        if ltp <= 0:
            return {}

        rsi = tech.get("rsi", 50)
        trade_type = self.classify_setup(0, tech, vol)

        # Risk/reward multipliers by trade type
        multipliers = {
            "INTRADAY": (0.003, 0.015, 0.008),   # entry_discount, target_pct, sl_pct
            "BTST":     (0.005, 0.030, 0.015),
            "SWING":    (0.008, 0.060, 0.025),
        }
        ed, tp, sp = multipliers.get(trade_type, (0.005, 0.030, 0.015))

        entry_low  = round(ltp * (1 - ed * 1.5), 2)
        entry_high = round(ltp * (1 + ed * 0.5), 2)
        target     = round(ltp * (1 + tp), 2)
        stop_loss  = round(ltp * (1 - sp), 2)
        risk       = round(ltp - stop_loss, 2)
        reward     = round(target - ltp, 2)
        rr_ratio   = round(reward / risk, 1) if risk > 0 else 0

        return {
            "entry_low":  entry_low,
            "entry_high": entry_high,
            "target":     target,
            "stop_loss":  stop_loss,
            "risk_pts":   risk,
            "reward_pts": reward,
            "rr_ratio":   rr_ratio,
        }

    def build_result(
        self,
        symbol: str,
        ltp: float,
        sm_result:   dict,
        vol_result:  dict,
        tech_result: dict,
        news_result: dict,
        fund_result: dict,
        sector:      str = "",
        market_cap_cr: float = 0,
    ) -> dict:
        """Assemble the full screener result for one symbol."""

        comp = self.composite_score(
            sm_result.get("score", 0),
            vol_result.get("score", 0),
            tech_result.get("score", 0),
            news_result.get("score", 0),
            fund_result.get("score", 0),
        )

        setup_type = self.classify_setup(comp, tech_result, vol_result)
        trade      = self.generate_trade_setup(symbol, ltp, tech_result, vol_result)

        # Signal classification
        if comp >= self.strong_buy_thresh:
            signal = "STRONG BUY"
            emoji  = "🔴"
        elif comp >= self.watch_thresh:
            signal = "WATCH"
            emoji  = "🟡"
        else:
            signal = "NO SIGNAL"
            emoji  = "⚪"

        # Collate all detail strings
        all_details = {}
        all_details.update(sm_result.get("details", {}))
        all_details.update(vol_result.get("details", {}))
        all_details.update(tech_result.get("details", {}))
        all_details.update(news_result.get("details", {}))
        all_details.update(fund_result.get("details", {}))

        return {
            "symbol":        symbol,
            "ltp":           ltp,
            "sector":        sector,
            "market_cap_cr": market_cap_cr,
            "composite_score": comp,
            "signal":        signal,
            "emoji":         emoji,
            "setup_type":    setup_type,
            "scores": {
                "smart_money": sm_result.get("score", 0),
                "volume":      vol_result.get("score", 0),
                "technical":   tech_result.get("score", 0),
                "news":        news_result.get("score", 0),
                "fundamental": fund_result.get("score", 0),
            },
            "technical": {
                "rsi":            tech_result.get("rsi"),
                "supertrend_buy": tech_result.get("supertrend_buy"),
                "ema_aligned":    tech_result.get("ema_aligned"),
                "patterns":       tech_result.get("patterns", []),
            },
            "volume": {
                "spike_ratio":   vol_result.get("spike_ratio"),
                "delivery_pct":  vol_result.get("delivery_pct"),
                "oi":            vol_result.get("oi", {}),
            },
            "smart_money": {
                "fii_positive":  sm_result.get("details", {}).get("fii"),
                "bulk_deals":    sm_result.get("details", {}).get("bulk"),
                "insider":       sm_result.get("details", {}).get("insider"),
            },
            "news": {
                "avg_sentiment": news_result.get("avg_sentiment"),
                "headlines":     news_result.get("headlines", [])[:3],
            },
            "trade_setup": trade,
            "all_details": all_details,
            "timestamp": datetime.now(IST).isoformat(),
        }

    def to_dataframe(self, results: list) -> pd.DataFrame:
        """Convert list of results to a flat DataFrame."""
        rows = []
        for r in results:
            rows.append({
                "Symbol":        r["symbol"],
                "LTP":           r["ltp"],
                "Score":         r["composite_score"],
                "Signal":        r["signal"],
                "Setup":         r["setup_type"],
                "RSI":           r["technical"].get("rsi"),
                "ST_BUY":        r["technical"].get("supertrend_buy"),
                "Vol_Spike":     r["volume"].get("spike_ratio"),
                "Delivery%":     r["volume"].get("delivery_pct"),
                "SM_Score":      r["scores"]["smart_money"],
                "Vol_Score":     r["scores"]["volume"],
                "Tech_Score":    r["scores"]["technical"],
                "News_Score":    r["scores"]["news"],
                "Fund_Score":    r["scores"]["fundamental"],
                "Entry_Low":     r["trade_setup"].get("entry_low"),
                "Entry_High":    r["trade_setup"].get("entry_high"),
                "Target":        r["trade_setup"].get("target"),
                "SL":            r["trade_setup"].get("stop_loss"),
                "RR":            r["trade_setup"].get("rr_ratio"),
                "Sector":        r["sector"],
                "Timestamp":     r["timestamp"],
            })
        return pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
