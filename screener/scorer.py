import pandas as pd
from datetime import datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")


class Scorer:

    def __init__(self, config):

        w = config["signals"]["weights"]

        self.weights = {
            "smart_money": w.get("smart_money", 0.20),
            "volume": w.get("volume", 0.20),
            "technical": w.get("technical", 0.40),
            "news": w.get("news", 0.10),
            "fundamental": w.get("fundamental", 0.10),
        }

        self.strong_buy = config["scoring"]["strong_buy_threshold"]
        self.watch = config["scoring"]["watch_threshold"]

    # ---------------- SAFE VALUE ---------------- #

    def _safe_val(self, x, default=0.0):
        try:
            if x is None:
                return float(default)

            if hasattr(x, "iloc"):
                if len(x) == 0:
                    return float(default)
                return float(x.iloc[-1].item())

            if hasattr(x, "item"):
                return float(x.item())

            if isinstance(x, str):
                x = x.strip().replace("%", "")
                if x == "" or x.lower() in ["nan", "none", "na"]:
                    return float(default)
                return float(x)

            return float(x)

        except Exception:
            return float(default)

    # ---------------- SCORE ---------------- #

    def composite_score(self, sm, vol, tech, news, fund):

        sm = self._safe_val(sm)
        vol = self._safe_val(vol)
        tech = self._safe_val(tech)
        news = self._safe_val(news)
        fund = self._safe_val(fund)

        score = (
            sm * self.weights["smart_money"] +
            vol * self.weights["volume"] +
            tech * self.weights["technical"] +
            news * self.weights["news"] +
            fund * self.weights["fundamental"]
        )

        return int(round(score))

    # ---------------- CLASSIFY ---------------- #

    def classify_setup(self, score, tech, vol):

        spike = self._safe_val(vol.get("spike_ratio", 1), 1)

        patterns = tech.get("patterns", [])
        if not isinstance(patterns, list):
            patterns = []

        ema = bool(tech.get("ema_aligned", False))
        st = bool(tech.get("supertrend_buy", False))

        if spike >= 2.5 and score >= 50:
            return "INTRADAY"

        if len(patterns) > 0 and score >= 45:
            return "BTST"

        if ema and st and score >= 40:
            return "SWING"

        if score >= self.watch:
            return "WATCH"

        return "NO_SIGNAL"

    # ---------------- TRADE ---------------- #

    def generate_trade_setup(self, ltp, score):

        if ltp <= 0:
            return {}

        if score >= 70:
            tp, sl = 0.05, 0.02
        elif score >= 50:
            tp, sl = 0.04, 0.02
        else:
            tp, sl = 0.03, 0.015

        entry = round(ltp, 2)
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

        sm = self._safe_val(sm_result.get("score", 0))
        vol = self._safe_val(vol_result.get("score", 0))
        tech = self._safe_val(tech_result.get("score", 0))
        news = self._safe_val(news_result.get("score", 0))
        fund = self._safe_val(fund_result.get("score", 0))

        comp = self.composite_score(sm, vol, tech, news, fund)

        setup = self.classify_setup(comp, tech_result, vol_result)

        trade = self.generate_trade_setup(ltp, comp)

        if comp >= self.strong_buy:
            signal = "BUY"
        elif comp >= self.watch:
            signal = "WATCH"
        elif comp >= 20:
            signal = "WEAK"
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

    # ---------------- DF ---------------- #

    def to_dataframe(self, results):

        rows = []

        for r in results:
            try:
                rows.append({
                    "Symbol": r.get("symbol"),
                    "Score": r.get("composite_score"),
                    "Signal": r.get("signal"),
                    "Setup": r.get("setup_type"),
                    "Entry": r.get("trade_setup", {}).get("entry_low"),
                    "Target": r.get("trade_setup", {}).get("target"),
                    "SL": r.get("trade_setup", {}).get("stop_loss"),
                })
            except:
                continue

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).sort_values("Score", ascending=False)
