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
            last_close = float(close_series.iloc[-1]) if not pd.isna(close_series.iloc[-1]) else 0
            ma20_series = close_series.rolling(20).mean()
            ma50_series = close_series.rolling(50).mean()

            ma20 = float(ma20_series.iloc[-1]) if not pd.isna(ma20_series.iloc[-1]) else last_close
            ma50 = float(ma50_series.iloc[-1]) if not pd.isna(ma50_series.iloc[-1]) else last_close

            # ---------------- RSI ---------------- #
            delta = close_series.diff()

            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)

            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()

            rs = avg_gain / (avg_loss + 1e-9)
            rsi = 100 - (100 / (1 + rs))

            rsi_val = rsi.iloc[-1]
            if pd.isna(rsi_val):
                rsi_val = 50
            rsi_val = float(rsi_val)

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
