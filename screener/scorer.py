import pandas as pd
from datetime import datetime
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

    # ---------------- SAFE VALUE ---------------- #

    def _safe_val(self, x, default=0):
        try:
            if hasattr(x, "iloc"):
                return float(x.iloc[-1])
            return float(x)
        except:
            return default

    # ---------------- SCORE ---------------- #

    def composite_score(self, sm, vol, tech, news, fund):

        return int(round(
            self._safe_val(sm) * self.weights["smart_money"] +
            self._safe_val(vol) * self.weights["volume"] +
            self._safe_val(tech) * self.weights["technical"] +
            self._safe_val(news) * self.weights["news"] +
            self._safe_val(fund) * self.weights["fundamental"]
        ))

    # ---------------- CLASSIFY ---------------- #

    def classify_setup(self, score, tech, vol):

        spike = self._safe_val(vol.get("spike_ratio", 1), 1)

        patterns = tech.get("patterns", [])
        if not isinstance(patterns, list):
            patterns = []

        ema = bool(tech.get("ema_aligned", False))
        st = bool(tech.get("supertrend_buy", False))

        if spike >= 2.5 and score >= 60:
            return "INTRADAY"

        if len(patterns) > 0 and score >= 55:
            return "BTST"

        if ema and st and score >= 50:
            return "SWING"

        if score >= self.watch:
            return "WATCH"

        return "NO_SIGNAL"

    # ---------------- TRADE ---------------- #

    def generate_trade_setup(self, ltp, score):

        if ltp <= 0:
            return {}

        if score >= 75:
            tp, sl = 0.05, 0.02
        elif score >= 60:
            tp, sl = 0.04, 0.02
        else:
            tp, sl = 0.03, 0.015

        entry = ltp
        target = round(ltp * (1 + tp), 2)
        stop = round(ltp * (1 - sl), 2)

        risk = entry - stop
        reward = target - entry

        rr = round(reward / risk, 1) if risk > 0 else 0

        return {
            "entry_low": entry,
            "target": target,
            "stop_loss": stop,
            "rr_ratio": rr,
        }

    # ---------------- BUILD ---------------- #

    def build_result(self, symbol, ltp, sm_result, vol_result, tech_result, news_result, fund_result):

        sm = sm_result.get("score", 0)
        vol = vol_result.get("score", 0)
        tech = tech_result.get("score", 0)
        news = news_result.get("score", 0)
        fund = fund_result.get("score", 0)

        comp = self.composite_score(sm, vol, tech, news, fund)

        setup = self.classify_setup(comp, tech_result, vol_result)

        trade = self.generate_trade_setup(ltp, comp)

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
            "technical": tech_result or {},
            "volume": vol_result or {},
            "trade_setup": trade,
            "timestamp": datetime.now(IST).isoformat(),
        }

    # ---------------- DF ---------------- #

    def to_dataframe(self, results):

        rows = []

        for r in results:
            rows.append({
                "Symbol": r.get("symbol"),
                "Score": r.get("composite_score"),
                "Signal": r.get("signal"),
                "Setup": r.get("setup_type"),
                "Target": r.get("trade_setup", {}).get("target"),
            })

        return pd.DataFrame(rows)
