import pandas as pd
from loguru import logger


class TechnicalAnalyzer:

    def __init__(self, config=None):
        self.config = config or {}

    def score(self, symbol, df):

        try:
            if df is None or df.empty or len(df) < 50:
                return {"score": 0}

            close_series = df["Close"]

            # ✅ FORCE SCALAR VALUES
            last_close = float(close_series.iloc[-1])
            ma20 = float(close_series.rolling(20).mean().iloc[-1])
            ma50 = float(close_series.rolling(50).mean().iloc[-1])

            # ---------------- RSI ---------------- #
            delta = close_series.diff()

            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)

            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()

            rs = avg_gain / (avg_loss + 1e-9)
            rsi = 100 - (100 / (1 + rs))

            rsi_val = float(rsi.iloc[-1])

            # ---------------- SCORING ---------------- #
            score = 0

            # ✅ NOW SAFE (all scalars)
            if last_close > ma20:
                score += 10

            if last_close > ma50:
                score += 10

            if 50 < rsi_val < 70:
                score += 10

            elif rsi_val < 30:
                score += 5

            return {
                "score": score,
                "rsi": round(rsi_val, 2),
                "trend": "bullish" if last_close > ma50 else "bearish"
            }

        except Exception as e:
            logger.error(f"{symbol} tech failed: {e}")
            return {"score": 0}
