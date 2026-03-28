import numpy as np
import pandas as pd
from loguru import logger


class TechnicalAnalyzer:

    def __init__(self, config: dict):
        cfg = config["signals"]["technical"]

        self.rsi_period = cfg.get("rsi_period", 14)
        self.rsi_buy_zone = cfg.get("rsi_buy_zone", [35, 65])
        self.rsi_oversold = cfg.get("rsi_oversold", 30)

        self.macd_fast = cfg.get("macd_fast", 12)
        self.macd_slow = cfg.get("macd_slow", 26)
        self.macd_signal = cfg.get("macd_signal", 9)

        self.st_period = cfg.get("supertrend_period", 10)
        self.st_mult = cfg.get("supertrend_multiplier", 3.0)

    # ---------------- RSI ---------------- #

    def rsi(self, close):
        if len(close) < self.rsi_period:
            return np.array([50] * len(close))

        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        avg_gain = pd.Series(gain).ewm(com=self.rsi_period - 1).mean()
        avg_loss = pd.Series(loss).ewm(com=self.rsi_period - 1).mean()

        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))

        return np.concatenate([[50], rsi.values])

    # ---------------- MACD ---------------- #

    def macd(self, close):
        s = pd.Series(close)

        fast = s.ewm(span=self.macd_fast, adjust=False).mean()
        slow = s.ewm(span=self.macd_slow, adjust=False).mean()

        macd = fast - slow
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        hist = macd - signal

        return macd.values, signal.values, hist.values

    # ---------------- Supertrend ---------------- #

    def supertrend(self, df):
        if len(df) < 20:
            df["st_direction"] = 1
            return df

        high = df["High"].values
        low = df["Low"].values
        close = df["Close"].values

        atr = self._atr(high, low, close)

        hl2 = (high + low) / 2
        upper = hl2 + self.st_mult * atr
        lower = hl2 - self.st_mult * atr

        st = np.zeros(len(close))
        direction = np.ones(len(close))

        for i in range(1, len(close)):
            if close[i] > upper[i - 1]:
                direction[i] = 1
            elif close[i] < lower[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]

            st[i] = lower[i] if direction[i] == 1 else upper[i]

        df["st_direction"] = direction
        return df

    def _atr(self, high, low, close):
        tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
        tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
        tr = np.concatenate([[high[0] - low[0]], tr])

        return pd.Series(tr).ewm(span=self.st_period).mean().values

    # ---------------- EMA ---------------- #

    def ema(self, close, period):
        return pd.Series(close).ewm(span=period, adjust=False).mean().values

    # ---------------- Patterns ---------------- #

    def detect_patterns(self, df):
        if len(df) < 3:
            return []

        o = df["Open"].values
        c = df["Close"].values

        patterns = []

        if c[-2] < o[-2] and c[-1] > o[-1]:
            patterns.append("Bullish Engulfing")

        return patterns

    # ---------------- MAIN SCORE ---------------- #

    def score(self, symbol, df):

        if df is None or len(df) < 50:
            return {"score": 0}

        try:
            close = df["Close"].values
            ltp = float(close[-1])

            pts = 0
            details = {}

            # RSI
            rsi_vals = self.rsi(close)
            rsi_now = float(rsi_vals[-1])

            if self.rsi_buy_zone[0] <= rsi_now <= self.rsi_buy_zone[1]:
                pts += 5
                details["rsi"] = f"{rsi_now:.1f}"

            elif rsi_now < self.rsi_oversold:
                pts += 3

            # MACD
            macd, signal, hist = self.macd(close)

            if len(hist) >= 2:
                if hist[-1] > 0 and hist[-1] > hist[-2]:
                    pts += 5
                elif hist[-1] > 0:
                    pts += 2

            # Supertrend
            df = self.supertrend(df.copy())

            if df["st_direction"].iloc[-1] == 1:
                pts += 5

            # EMA
            ema9 = self.ema(close, 9)[-1]
            ema21 = self.ema(close, 21)[-1]
            ema50 = self.ema(close, 50)[-1]

            if ema9 > ema21 > ema50:
                pts += 4
            elif ltp > ema50:
                pts += 2

            # Patterns
            patterns = self.detect_patterns(df)
            if patterns:
                pts += 3

            # 52 week high
            high_52 = df["High"].tail(252).max()
            if high_52 > 0 and ltp >= high_52 * 0.97:
                pts += 3

            score = int(min(100, pts / 25 * 100))

            return {
                "score": score,
                "rsi": round(rsi_now, 1),
                "patterns": patterns,
                "details": details,
            }

        except Exception as e:
            logger.error(f"{symbol} tech failed: {e}")
            return {"score": 0}
