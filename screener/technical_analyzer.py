"""
technical_analyzer.py
---------------------
Computes all technical signals:
  - RSI, MACD, Supertrend, EMA stack
  - Bollinger Band squeeze / breakout
  - 52-week high proximity
  - Candlestick patterns (engulfing, hammer, doji)
  - Support/Resistance zone checks
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Optional


class TechnicalAnalyzer:
    def __init__(self, config: dict):
        cfg = config["signals"]["technical"]
        self.rsi_period     = cfg.get("rsi_period", 14)
        self.rsi_buy_zone   = cfg.get("rsi_buy_zone", [35, 65])
        self.rsi_oversold   = cfg.get("rsi_oversold", 30)
        self.rsi_overbought = cfg.get("rsi_overbought", 75)
        self.macd_fast      = cfg.get("macd_fast", 12)
        self.macd_slow      = cfg.get("macd_slow", 26)
        self.macd_signal    = cfg.get("macd_signal", 9)
        self.st_period      = cfg.get("supertrend_period", 10)
        self.st_mult        = cfg.get("supertrend_multiplier", 3.0)
        self.ema_periods    = cfg.get("ema_periods", [9, 21, 50, 200])
        self.bb_period      = cfg.get("bb_period", 20)
        self.bb_std         = cfg.get("bb_std", 2)

    # ------------------------------------------------------------------ #
    #  Indicators
    # ------------------------------------------------------------------ #

    def rsi(self, close: np.ndarray, period: int = None) -> np.ndarray:
        p = period or self.rsi_period
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).ewm(com=p - 1, min_periods=p).mean().values
        avg_loss = pd.Series(loss).ewm(com=p - 1, min_periods=p).mean().values
        rs = avg_gain / (avg_loss + 1e-9)
        rsi_vals = 100 - (100 / (1 + rs))
        return np.concatenate([[50], rsi_vals])

    def macd(self, close: np.ndarray) -> dict:
        s = pd.Series(close)
        fast_ema = s.ewm(span=self.macd_fast, adjust=False).mean()
        slow_ema = s.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            "macd": macd_line.values,
            "signal": signal_line.values,
            "histogram": histogram.values,
        }

    def supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supertrend indicator."""
        high = df["High"].values
        low = df["Low"].values
        close = df["Close"].values
        n = len(close)

        atr = self._atr(high, low, close, self.st_period)
        hl2 = (high + low) / 2.0
        upper = hl2 + self.st_mult * atr
        lower = hl2 - self.st_mult * atr

        supertrend = np.zeros(n)
        direction = np.ones(n)  # 1 = Bullish, -1 = Bearish

        for i in range(1, n):
            # Adjust bands
            if close[i - 1] <= upper[i - 1]:
                upper[i] = min(upper[i], upper[i - 1])
            if close[i - 1] >= lower[i - 1]:
                lower[i] = max(lower[i], lower[i - 1])

            if supertrend[i - 1] == upper[i - 1]:
                if close[i] <= upper[i]:
                    supertrend[i] = upper[i]
                    direction[i] = -1
                else:
                    supertrend[i] = lower[i]
                    direction[i] = 1
            else:
                if close[i] >= lower[i]:
                    supertrend[i] = lower[i]
                    direction[i] = 1
                else:
                    supertrend[i] = upper[i]
                    direction[i] = -1

        df = df.copy()
        df["supertrend"] = supertrend
        df["st_direction"] = direction  # 1 = BUY, -1 = SELL
        return df

    def ema(self, close: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(close).ewm(span=period, adjust=False).mean().values

    def bollinger_bands(self, close: np.ndarray) -> dict:
        s = pd.Series(close)
        mid = s.rolling(self.bb_period).mean()
        std = s.rolling(self.bb_period).std()
        upper = mid + self.bb_std * std
        lower = mid - self.bb_std * std
        bandwidth = ((upper - lower) / mid).values
        pct_b = ((s - lower) / (upper - lower + 1e-9)).values
        return {"upper": upper.values, "lower": lower.values, "mid": mid.values,
                "bandwidth": bandwidth, "pct_b": pct_b}

    def _atr(self, high, low, close, period):
        tr = np.maximum(high[1:] - low[1:],
             np.maximum(abs(high[1:] - close[:-1]),
                        abs(low[1:]  - close[:-1])))
        tr = np.concatenate([[high[0] - low[0]], tr])
        return pd.Series(tr).ewm(span=period, adjust=False).mean().values

    # ------------------------------------------------------------------ #
    #  Candlestick Patterns
    # ------------------------------------------------------------------ #

    def detect_patterns(self, df: pd.DataFrame) -> dict:
        """Detect key bullish reversal candlestick patterns."""
        o = df["Open"].values
        h = df["High"].values
        l = df["Low"].values
        c = df["Close"].values
        patterns = []

        if len(c) < 3:
            return {"patterns": [], "bullish_pattern": False}

        body = abs(c - o)
        candle_range = h - l + 1e-9
        body_ratio = body / candle_range

        # --- Bullish Engulfing (last 2 candles) ---
        if (c[-2] < o[-2]) and (c[-1] > o[-1]) and (c[-1] > o[-2]) and (o[-1] < c[-2]):
            patterns.append("Bullish Engulfing")

        # --- Hammer ---
        lower_wick = o[-1] - l[-1] if c[-1] >= o[-1] else c[-1] - l[-1]
        upper_wick = h[-1] - (max(o[-1], c[-1]))
        if (lower_wick >= 2 * body[-1]) and (upper_wick < 0.3 * body[-1]) and body_ratio[-1] > 0.2:
            patterns.append("Hammer")

        # --- Morning Star (3 candles) ---
        if len(c) >= 3:
            if (c[-3] < o[-3] and                   # Day 1: bearish
                body[-2] < 0.3 * body[-3] and       # Day 2: small body (doji-like)
                c[-1] > o[-1] and                   # Day 3: bullish
                c[-1] > (o[-3] + c[-3]) / 2):       # Day 3 closes above midpoint of Day 1
                patterns.append("Morning Star")

        # --- Bullish Marubozu ---
        if c[-1] > o[-1] and body_ratio[-1] > 0.85:
            patterns.append("Bullish Marubozu")

        # --- Inside Bar Breakout ---
        if h[-1] > h[-2] and l[-1] >= l[-2] and c[-1] > o[-1]:
            patterns.append("Inside Bar Breakout")

        return {"patterns": patterns, "bullish_pattern": len(patterns) > 0}

    # ------------------------------------------------------------------ #
    #  Composite Technical Score
    # ------------------------------------------------------------------ #

    def score(self, symbol: str, df: pd.DataFrame) -> dict:
        """
        Technical score (0-100).

        Breakdown (total 25 pts → normalized to 100):
          RSI in buy zone     : 5 pts
          MACD bullish cross  : 5 pts
          Supertrend BUY      : 5 pts
          EMA stack aligned   : 4 pts
          Breakout / pattern  : 3 pts
          52-wk high prox     : 3 pts
        """
        if df is None or len(df) < 50:
            return {"score": 0, "details": {}}

        close = df["Close"].values
        pts = 0
        details = {}

        # 1. RSI
        rsi_vals = self.rsi(close)
        rsi_now = rsi_vals[-1]
        if self.rsi_buy_zone[0] <= rsi_now <= self.rsi_buy_zone[1]:
            pts += 5
            details["rsi"] = f"RSI {rsi_now:.1f} (buy zone) ✅"
        elif rsi_now <= self.rsi_oversold:
            pts += 3
            details["rsi"] = f"RSI {rsi_now:.1f} (oversold bounce) ⚡"
        else:
            details["rsi"] = f"RSI {rsi_now:.1f}"

        # 2. MACD
        macd_data = self.macd(close)
        macd_h = macd_data["histogram"]
        if macd_h[-1] > 0 and macd_h[-1] > macd_h[-2]:
            pts += 5
            details["macd"] = "MACD histogram rising ✅"
        elif macd_h[-1] > 0:
            pts += 2
            details["macd"] = "MACD positive"

        # 3. Supertrend
        df_st = self.supertrend(df)
        if df_st["st_direction"].iloc[-1] == 1:
            pts += 5
            details["supertrend"] = "Supertrend BUY ✅"

        # 4. EMA Stack
        ema9   = self.ema(close, 9)[-1]
        ema21  = self.ema(close, 21)[-1]
        ema50  = self.ema(close, 50)[-1]
        ema200 = self.ema(close, 200)[-1]
        ltp = close[-1]
        if ema9 > ema21 > ema50 > ema200 and ltp > ema9:
            pts += 4
            details["ema"] = "Full EMA stack aligned ✅"
        elif ltp > ema50:
            pts += 2
            details["ema"] = "Price above 50 EMA"

        # 5. Candlestick Patterns
        pat = self.detect_patterns(df)
        if pat["bullish_pattern"]:
            pts += 3
            details["pattern"] = ", ".join(pat["patterns"]) + " ✅"

        # 6. 52-week high proximity
        high_52w = df["High"].tail(252).max()
        if ltp >= high_52w * 0.97:
            pts += 3
            details["52w"] = f"Near 52-week high ({ltp/high_52w*100:.1f}%) ✅"

        # 7. Bollinger Band
        bb = self.bollinger_bands(close)
        if 0.8 <= bb["pct_b"][-1] <= 1.0:
            pts += 0  # Approaching upper band — momentum but also overbought
        if bb["bandwidth"][-1] < bb["bandwidth"][-20:-1].mean() * 0.5:
            pts += 2
            details["bb"] = "Bollinger squeeze (breakout imminent) ⚡"

        score = min(100, int(pts / 25 * 100))
        return {
            "score": score,
            "raw_pts": pts,
            "rsi": round(rsi_now, 1),
            "macd_histogram": round(float(macd_h[-1]), 4),
            "supertrend_buy": bool(df_st["st_direction"].iloc[-1] == 1),
            "ema_aligned": ema9 > ema21 > ema50,
            "patterns": pat["patterns"],
            "details": details,
        }
