"""
volume_analyzer.py  (v2)
------------------------
FIXES vs v1:
  * Score is a genuine 0-100. v1 capped at 30 while the scorer weighted
    it as if it could reach 100 -> its real influence was 6%, not 20%.
  * Partial-bar correction. v1 compared today's INCOMPLETE volume against
    a 20-day full-session average, so the spike ratio measured the clock,
    not the stock (suppressed at 09:45, inflated at 15:15).
  * delivery_df is actually USED. v1 accepted the parameter and ignored it.
  * Reads spike_multiplier / delivery_pct_min from config instead of
    hardcoding thresholds.
  * Adds an accumulation signal (up-volume vs down-volume) — cheap and
    genuinely informative about who is doing the buying.
"""

import pandas as pd
import numpy as np
from loguru import logger

from .technical_analyzer import flatten_columns

# NSE trading session = 09:15 -> 15:30 = 375 minutes
SESSION_MINUTES = 375


class VolumeAnalyzer:

    def __init__(self, session=None, config=None):
        self.session = session
        cfg = (config or {}).get("signals", {}).get("volume", {})
        self.spike_multiplier = float(cfg.get("spike_multiplier", 2.5))
        self.delivery_min     = float(cfg.get("delivery_pct_min", 55))
        self.adjust_partial   = bool(cfg.get("adjust_partial_bar", True))

    # ------------------------------------------------------------------ #

    @staticmethod
    def _series(df, col):
        data = df[col]
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
        return pd.to_numeric(data, errors="coerce").dropna().astype(float)

    @staticmethod
    def lookup_delivery(delivery_df, symbol):
        """
        NSE's delivery payload changes column names between endpoints and
        between years. Try every spelling we've seen, return None if absent
        so the caller can renormalise instead of scoring a fake 0.
        """
        if delivery_df is None or not isinstance(delivery_df, pd.DataFrame) or delivery_df.empty:
            return None

        sym_cols = ["symbol", "SYMBOL", "Symbol", "TckrSymb", "BD_SYMBOL"]
        pct_cols = [
            "deliveryToTradedQuantity", "DELIV_PER", "deliveryPercentage",
            "delivPer", "DelivPer", "delivery_pct", "DELIVERY_PERCENTAGE",
        ]

        sym_col = next((c for c in sym_cols if c in delivery_df.columns), None)
        pct_col = next((c for c in pct_cols if c in delivery_df.columns), None)
        if not sym_col or not pct_col:
            return None

        try:
            match = delivery_df[
                delivery_df[sym_col].astype(str).str.strip().str.upper() == symbol.upper()
            ]
            if match.empty:
                return None
            val = pd.to_numeric(
                str(match.iloc[0][pct_col]).replace("%", "").strip(), errors="coerce"
            )
            if pd.isna(val):
                return None
            return float(val)
        except Exception:
            return None

    @staticmethod
    def session_fraction(now=None):
        """Fraction of the trading session elapsed, in (0, 1]."""
        from datetime import datetime
        import pytz
        IST = pytz.timezone("Asia/Kolkata")
        now = now or datetime.now(IST)
        if now.tzinfo is None:
            now = IST.localize(now)
        open_t  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
        close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
        if now <= open_t:
            return 1.0          # pre-open: last bar is a complete previous session
        if now >= close_t:
            return 1.0
        elapsed = (now - open_t).total_seconds() / 60.0
        return max(elapsed / SESSION_MINUTES, 0.05)   # floor avoids a divide blow-up

    # ------------------------------------------------------------------ #
    #  Score  (genuine 0-100)
    # ------------------------------------------------------------------ #
    #  Relative volume        45
    #  Delivery %             30   (dropped + renormalised if unavailable)
    #  Short-term vol trend   15
    #  Accumulation (up/down) 10
    #                        ----
    #                        100
    # ------------------------------------------------------------------ #

    def score(self, symbol, df, delivery_df=None, intraday=False):
        try:
            if df is None or len(df) == 0:
                return {"score": 0.0, "available": False, "reason": "no data"}

            df = flatten_columns(df)
            missing = [c for c in ("Volume", "Close") if c not in df.columns]
            if missing:
                return {"score": 0.0, "available": False,
                        "reason": f"missing column(s): {', '.join(missing)}"}

            vol = self._series(df, "Volume")
            close = self._series(df, "Close")

            if len(vol) < 21:
                return {"score": 0.0, "available": False, "reason": f"only {len(vol)} bars"}

            latest_vol = float(vol.iloc[-1])
            # exclude the current (possibly partial) bar from its own baseline
            avg_vol = float(vol.iloc[-21:-1].mean())

            if avg_vol <= 0:
                return {"score": 0.0, "available": False, "reason": "zero avg volume"}

            # --- partial-bar correction -----------------------------------
            frac = 1.0
            if intraday and self.adjust_partial:
                frac = self.session_fraction()
            projected = latest_vol / frac
            ratio = projected / avg_vol

            pts = 0.0
            budget = 0.0
            detail = {}

            # --- 1. Relative volume: 45 pts --------------------------------
            budget += 45
            m = self.spike_multiplier
            if ratio >= m:
                pts += 45
            elif ratio >= m * 0.8:
                pts += 36
            elif ratio >= 1.5:
                pts += 27
            elif ratio >= 1.2:
                pts += 16
            elif ratio >= 1.0:
                pts += 8
            detail["spike_ratio"] = round(ratio, 2)
            detail["partial_bar_adj"] = round(frac, 3)

            # --- 2. Delivery %: 30 pts (renormalise if missing) ------------
            deliv = self.lookup_delivery(delivery_df, symbol)
            if deliv is not None:
                budget += 30
                if deliv >= self.delivery_min:
                    pts += 30
                elif deliv >= self.delivery_min - 10:
                    pts += 18
                elif deliv >= self.delivery_min - 20:
                    pts += 8
                detail["delivery_pct"] = round(deliv, 1)

            # --- 3. Short-term volume trend: 15 pts ------------------------
            budget += 15
            avg5  = float(vol.iloc[-6:-1].mean())
            trend = avg5 / avg_vol if avg_vol > 0 else 1.0
            if trend >= 1.5:
                pts += 15
            elif trend >= 1.2:
                pts += 10
            elif trend >= 1.0:
                pts += 5
            detail["vol_trend_5v20"] = round(trend, 2)

            # --- 4. Accumulation: 10 pts -----------------------------------
            # Volume on up-days vs down-days over the last 20 sessions.
            budget += 10
            ret = close.pct_change()
            common = ret.index.intersection(vol.index)
            r20 = ret.loc[common].tail(20)
            v20 = vol.loc[common].tail(20)
            up_vol   = float(v20[r20 > 0].sum())
            down_vol = float(v20[r20 < 0].sum())
            if down_vol > 0:
                acc = up_vol / down_vol
                if acc >= 1.5:
                    pts += 10
                elif acc >= 1.2:
                    pts += 7
                elif acc >= 1.0:
                    pts += 4
                detail["accumulation_ratio"] = round(acc, 2)

            score = round(pts / budget * 100.0, 1) if budget > 0 else 0.0

            return {
                "score": score,
                "available": True,
                "avg_volume_20d": int(avg_vol),
                **detail,
            }

        except Exception as e:
            logger.error(f"{symbol} volume failed: {e}")
            return {"score": 0.0, "available": False, "reason": str(e)}
