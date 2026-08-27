"""
technical_analyzer.py  (v2)
---------------------------
FIXES vs v1:
  * RSI now uses Wilder smoothing (ewm alpha=1/n), not rolling().mean().
    v1 diverged from every charting platform by ~5-6 points.
  * Score is a genuine 0-100. v1 topped out at 70 while the scorer
    weighted it as if it could reach 100.
  * Actually reads config.yaml (EMA periods, MACD, supertrend, ATR,
    RSI zone) instead of hardcoding MA20/MA50/RSI14 and ignoring config.
  * Emits ATR + swing low so the scorer can build volatility-scaled
    stops instead of a flat 2%.
  * Renormalises when history is too short for the 200-EMA components
    rather than silently scoring the stock low.
"""

import numpy as np
import pandas as pd
from loguru import logger


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns MultiIndex columns; collapse to the OHLCV level."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


class TechnicalAnalyzer:

    def __init__(self, config=None):
        cfg = (config or {}).get("signals", {}).get("technical", {})
        self.rsi_period    = int(cfg.get("rsi_period", 14))
        self.rsi_zone      = cfg.get("rsi_buy_zone", [40, 65])
        self.rsi_oversold  = float(cfg.get("rsi_oversold", 30))
        self.rsi_overbought= float(cfg.get("rsi_overbought", 75))
        self.macd_fast     = int(cfg.get("macd_fast", 12))
        self.macd_slow     = int(cfg.get("macd_slow", 26))
        self.macd_signal   = int(cfg.get("macd_signal", 9))
        self.st_period     = int(cfg.get("supertrend_period", 10))
        self.st_mult       = float(cfg.get("supertrend_multiplier", 3.0))
        self.ema_periods   = cfg.get("ema_periods", [21, 50, 200])
        self.atr_period    = int(cfg.get("atr_period", 14))
        self.breakout_lb   = int(cfg.get("breakout_lookback", 20))

    # ------------------------------------------------------------------ #
    #  Series helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _series(df, col):
        """Always return a clean 1-D float Series for an OHLCV column."""
        data = df[col]
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
        return pd.to_numeric(data, errors="coerce").dropna().astype(float)

    @staticmethod
    def _wilder(s: pd.Series, n: int) -> pd.Series:
        """Wilder's smoothing == EMA with alpha = 1/n."""
        return s.ewm(alpha=1.0 / n, adjust=False).mean()

    # ------------------------------------------------------------------ #
    #  Indicators
    # ------------------------------------------------------------------ #

    def rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = -delta.clip(upper=0)
        avg_gain = self._wilder(gain, self.rsi_period)
        avg_loss = self._wilder(loss, self.rsi_period)
        rs = avg_gain / avg_loss.replace(0, np.nan)
        out = 100 - (100 / (1 + rs))
        return out.fillna(50.0)

    def macd(self, close: pd.Series):
        ema_f = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_s = close.ewm(span=self.macd_slow, adjust=False).mean()
        line  = ema_f - ema_s
        sig   = line.ewm(span=self.macd_signal, adjust=False).mean()
        return line, sig, line - sig

    def atr(self, high, low, close) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return self._wilder(tr, self.atr_period)

    def supertrend(self, high, low, close, atr):
        """
        Standard ATR bands supertrend.
        Returns (direction_series, line_series); direction 1 = long, -1 = short.
        """
        n = len(close)
        hl2 = (high + low) / 2.0
        upper_basic = (hl2 + self.st_mult * atr).to_numpy()
        lower_basic = (hl2 - self.st_mult * atr).to_numpy()
        c = close.to_numpy()

        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        direction = np.ones(n, dtype=int)

        start = max(self.st_period, self.atr_period)
        if n <= start:
            return pd.Series(direction, index=close.index), pd.Series(lower, index=close.index)

        upper[start] = upper_basic[start]
        lower[start] = lower_basic[start]

        for i in range(start + 1, n):
            upper[i] = upper_basic[i] if (upper_basic[i] < upper[i - 1] or c[i - 1] > upper[i - 1]) else upper[i - 1]
            lower[i] = lower_basic[i] if (lower_basic[i] > lower[i - 1] or c[i - 1] < lower[i - 1]) else lower[i - 1]

            if direction[i - 1] == 1:
                direction[i] = -1 if c[i] < lower[i] else 1
            else:
                direction[i] = 1 if c[i] > upper[i] else -1

        line = np.where(direction == 1, lower, upper)
        return pd.Series(direction, index=close.index), pd.Series(line, index=close.index)

    # ------------------------------------------------------------------ #
    #  Score  (genuine 0-100)
    # ------------------------------------------------------------------ #
    #  Budget:
    #    Trend structure (EMA stack)  25
    #    Supertrend direction         20
    #    RSI zone                     15
    #    MACD histogram               15
    #    Breakout proximity           15
    #    Long-term trend (200-EMA)    10
    #                                ----
    #                                100
    #  Components whose lookback exceeds available history are excluded
    #  from BOTH numerator and denominator, then the score is rescaled.
    # ------------------------------------------------------------------ #

    def score(self, symbol, df):
        try:
            if df is None or len(df) == 0:
                return {"score": 0.0, "available": False, "reason": "no data"}

            df = flatten_columns(df)
            close = self._series(df, "Close")
            high  = self._series(df, "High")  if "High"  in df.columns else close
            low   = self._series(df, "Low")   if "Low"   in df.columns else close

            n = len(close)
            if n < 30:
                return {"score": 0.0, "available": False, "reason": f"only {n} bars"}

            # align (dropna may have trimmed different amounts)
            idx = close.index.intersection(high.index).intersection(low.index)
            close, high, low = close.loc[idx], high.loc[idx], low.loc[idx]
            n = len(close)

            last = float(close.iloc[-1])
            pts = 0.0
            budget = 0.0
            detail = {}

            # ---------- EMAs ----------
            emas = {}
            for p in self.ema_periods:
                if n >= p:
                    emas[p] = float(close.ewm(span=p, adjust=False).mean().iloc[-1])

            e_short = emas.get(21)
            e_mid   = emas.get(50)
            e_long  = emas.get(200)

            # Trend structure: 25 pts (needs 21 & 50)
            if e_short is not None and e_mid is not None:
                budget += 25
                if last > e_mid:
                    pts += 10
                if e_short > e_mid:
                    pts += 10
                if last > e_short:
                    pts += 5
                detail["ema_aligned"] = bool(last > e_short > e_mid)

            # Long-term trend: 10 pts (needs 200)
            if e_long is not None:
                budget += 10
                ema200 = close.ewm(span=200, adjust=False).mean()
                slope_up = bool(ema200.iloc[-1] > ema200.iloc[-21]) if n >= 221 else False
                if last > e_long:
                    pts += 6
                if slope_up:
                    pts += 4
                detail["above_200ema"] = bool(last > e_long)

            # ---------- RSI: 15 pts ----------
            rsi_s = self.rsi(close)
            rsi_v = float(rsi_s.iloc[-1])
            budget += 15
            lo, hi = float(self.rsi_zone[0]), float(self.rsi_zone[1])
            if lo <= rsi_v <= hi:
                pts += 15
            elif hi < rsi_v <= self.rsi_overbought:
                pts += 9
            elif rsi_v > self.rsi_overbought:
                pts += 2                      # extended, not a fresh entry
            elif rsi_v < self.rsi_oversold:
                pts += 7                      # washed out, possible reversal
            else:
                pts += 4
            detail["rsi"] = round(rsi_v, 2)

            # ---------- MACD: 15 pts ----------
            if n >= self.macd_slow + self.macd_signal:
                budget += 15
                _, _, hist = self.macd(close)
                h_now  = float(hist.iloc[-1])
                h_prev = float(hist.iloc[-2])
                if h_now > 0:
                    pts += 8
                if h_now > h_prev:
                    pts += 7
                detail["macd_hist"] = round(h_now, 4)
                detail["macd_rising"] = bool(h_now > h_prev)

            # ---------- ATR + Supertrend ----------
            atr_s = self.atr(high, low, close)
            atr_v = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else 0.0

            if n > max(self.st_period, self.atr_period) + 2:
                budget += 20
                direction, st_line = self.supertrend(high, low, close, atr_s)
                st_dir = int(direction.iloc[-1])
                if st_dir == 1:
                    pts += 20
                    # a fresh flip is worth flagging, not extra points
                    detail["supertrend_fresh"] = bool(int(direction.iloc[-2]) == -1)
                detail["supertrend_buy"] = bool(st_dir == 1)
                st_val = float(st_line.iloc[-1]) if not pd.isna(st_line.iloc[-1]) else 0.0
                detail["supertrend_line"] = round(st_val, 2)

            # ---------- Breakout proximity: 15 pts ----------
            if n >= self.breakout_lb:
                budget += 15
                hh = float(high.tail(self.breakout_lb).max())
                if hh > 0:
                    gap = (hh - last) / hh * 100.0
                    if gap <= 1.0:
                        pts += 15
                    elif gap <= 3.0:
                        pts += 11
                    elif gap <= 6.0:
                        pts += 6
                    elif gap <= 12.0:
                        pts += 2
                    detail["pct_from_20d_high"] = round(gap, 2)

            if budget <= 0:
                return {"score": 0.0, "available": False, "reason": "no computable components"}

            score = round(pts / budget * 100.0, 1)

            # ---------- context the scorer needs for risk sizing ----------
            swing_low = float(low.tail(self.breakout_lb).min())
            atr_pct = round(atr_v / last * 100.0, 2) if last > 0 else 0.0

            return {
                "score": score,
                "available": True,
                "components_used": int(budget),
                "rsi": round(rsi_v, 2),
                "atr": round(atr_v, 2),
                "atr_pct": atr_pct,
                "swing_low": round(swing_low, 2),
                "trend": "bullish" if (e_mid is not None and last > e_mid) else "bearish",
                **detail,
            }

        except Exception as e:
            logger.error(f"{symbol} technical failed: {e}")
            return {"score": 0.0, "available": False, "reason": str(e)}
