import pandas as pd
from datetime import datetime
from loguru import logger
import pytz

IST = pytz.timezone("Asia/Kolkata")


class Scorer:

    def __init__(self, config):

        w = config["signals"]["weights"]

        self.weights = {
            "smart_money": w.get("smart_money", 0.25),
            "volume": w.get("volume", 0.20),
            "technical": w.get("technical", 0.25),
            "news": w.get("news", 0.15),
            "fundamental": w.get("fundamental", 0.15),
        }

        self.strong_buy = config["scoring"]["strong_buy_threshold"]
        self.watch = config["scoring"]["watch_threshold"]

    # ------------------------------------------------ #
    # COMPOSITE SCORE
    # ------------------------------------------------ #

    def composite_score(self, sm, vol, tech, news, fund):

        try:
            score = (
                sm * self.weights["smart_money"] +
                vol * self.weights["volume"] +
                tech * self.weights["technical"] +
                news * self.weights["news"] +
                fund * self.weights["fundamental"]
            )
            return int(round(score))
        except:
            return 0

    # ------------------------------------------------ #
    # CLASSIFY
    # ------------------------------------------------ #

    def classify_setup(self, score, tech, vol):

        spike = vol.get("spike_ratio", 1) or 1
        patterns = tech.get("patterns", []) or []
        ema = tech.get("ema_aligned", False)
        st = tech.get("supertrend_buy", False)

        if spike >= 2.5 and score >= 60:
            return "INTRADAY"

        if patterns and score >= 55:
            return "BTST"

        if ema and st and score >= 50:
            return "SWING"

        if score >= self.watch:
            return "WATCH"

        return "NO_SIGNAL"

    # ------------------------------------------------ #
    # TRADE SETUP (SMART)
    # ------------------------------------------------ #

    def generate_trade_setup(self, ltp, score, setup):

        if ltp <= 0:
            return {}

        # dynamic risk model
        if score >= 75:
            tp, sl = 0.05, 0.02
        elif score >= 60:
            tp, sl = 0.04, 0.02
        else:
            tp, sl = 0.03, 0.015

        entry_low = round(ltp * 0.995, 2)
        entry_high = round(ltp * 1.005, 2)

        target = round(ltp * (1 + tp), 2)
        stop = round(ltp * (1 - sl), 2)

        risk = ltp - stop
        reward = target - ltp

        rr = round(reward / risk, 1) if risk > 0 else 0

        return {
            "entry_low": entry_low,
            "entry_high": entry_high,
            "target": target,
            "stop_loss": stop,
            "rr_ratio": rr,
        }

    # ------------------------------------------------ #
    # BUILD RESULT
    # ------------------------------------------------ #

    def build_result(
        self,
        symbol,
        ltp,
        sm_result,
        vol_result,
        tech_result,
        news_result,
        fund_result,
    ):

        sm = sm_result.get("score", 0)
        vol = vol_result.get("score", 0)
        tech = tech_result.get("score", 0)
        news = news_result.get("score", 0)
        fund = fund_result.get("score", 0)

        comp = self.composite_score(sm, vol, tech, news, fund)

        setup = self.classify_setup(comp, tech_result, vol_result)

        trade = self.generate_trade_setup(ltp, comp, setup)

        # SIGNAL LOGIC (FIXED)
        if comp >= self.strong_buy:
            signal = "BUY"
        elif comp >= self.watch:
            signal = "WATCH"
        else:
            signal = "IGNORE"

        return {
            "symbol": symbol,
            "ltp": ltp,
            "composite_score": comp,
            "signal": signal,
            "setup_type": setup,
            "scores": {
                "smart_money": sm,
                "volume": vol,
                "technical": tech,
                "news": news,
                "fundamental": fund,
            },
            "technical": tech_result or {},
            "volume": vol_result or {},
            "trade_setup": trade,
            "timestamp": datetime.now(IST).isoformat(),
        }

    # ------------------------------------------------ #
    # DATAFRAME
    # ------------------------------------------------ #

    def to_dataframe(self, results):

        rows = []

        for r in results:
            try:
                rows.append({
                    "Symbol": r["symbol"],
                    "Score": r["composite_score"],
                    "Signal": r["signal"],
                    "Setup": r["setup_type"],
                    "Entry": r["trade_setup"].get("entry_low"),
                    "Target": r["trade_setup"].get("target"),
                    "SL": r["trade_setup"].get("stop_loss"),
                })
            except:
                continue

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).sort_values("Score", ascending=False)
