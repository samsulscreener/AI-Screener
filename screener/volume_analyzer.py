from loguru import logger


class VolumeAnalyzer:

    def __init__(self, session=None, config=None):
        self.session = session
        self.config = config or {}

    def score(self, symbol, df, delivery_df=None):

        try:
            if df is None or df.empty:
                return {"score": 0}

            # ✅ FIXED (no warnings, no Series issues)
            latest_vol = float(df["Volume"].iloc[-1])
            avg_vol = float(df["Volume"].tail(20).mean())

            if avg_vol == 0:
                return {"score": 0}

            ratio = latest_vol / avg_vol

            if ratio > 2:
                score = 20
            elif ratio > 1.5:
                score = 15
            elif ratio > 1.2:
                score = 10
            else:
                score = 5

            return {
                "score": score,
                "volume_ratio": round(ratio, 2)
            }

        except Exception as e:
            logger.error(f"{symbol} volume error: {e}")
            return {"score": 0}
