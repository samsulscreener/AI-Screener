from loguru import logger


class VolumeAnalyzer:

    def __init__(self, session=None, config=None):
        self.session = session
        self.config = config or {}

    def score(self, symbol, df, delivery_df=None):  # ✅ FIXED SIGNATURE

        try:
            if df is None or df.empty:
                return {"score": 0}

            if "Volume" not in df.columns:
                return {"score": 0}

            vol_series = df["Volume"].dropna()

            if len(vol_series) < 20:
                return {"score": 0}

            # ✅ SAFE SCALAR
            latest_vol = float(vol_series.iloc[-1].item())
            avg_vol = float(vol_series.tail(20).mean())

            if avg_vol == 0:
                return {"score": 0}

            ratio = latest_vol / avg_vol

            # 🔥 STRONGER SCORING
            if ratio >= 2.5:
                score = 30
            elif ratio >= 2.0:
                score = 25
            elif ratio >= 1.5:
                score = 20
            elif ratio >= 1.2:
                score = 10
            else:
                score = 5

            return {
                "score": score,
                "spike_ratio": round(ratio, 2)
            }

        except Exception as e:
            logger.error(f"{symbol} volume error: {e}")
            return {"score": 0}
