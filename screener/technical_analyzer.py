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

            # ✅ SAFE SCALAR VALUES
            last_close = float(close_series.iloc[-1].item())
            ma20 = float(close_series.rolling(20).mean().iloc[-1].item())
            ma50 = float(close_series.rolling(50).mean().iloc[-1].item())

            # ---------------- RSI ---------------- #
            delta = close_series.diff()

            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)

            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()

            rs = avg_gain / (avg_loss + 1e-9)
            rsi = 100 - (100 / (1 + rs))

            rsi_val = float(rsi.iloc[-1].item())

            # ---------------- SCORING ---------------- #
            score = 0

            # 🔥 STRONG TREND
            if last_close > ma20:
                score += 20

            if last_close > ma50:
                score += 20

            # 🔥 RSI STRENGTH
            if 50 < rsi_val < 65:
                score += 20

            elif 65 <= rsi_val < 75:
                score += 15

            elif rsi_val < 30:
                score += 10

            return {
                "score": score,
                "rsi": round(rsi_val, 2),
                "trend": "bullish" if last_close > ma50 else "bearish"
            }

        except Exception as e:
            logger.error(f"{symbol} tech failed: {e}")
            return {"score": 0}
