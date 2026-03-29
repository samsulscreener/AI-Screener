from loguru import logger


class VolumeAnalyzer:

    def __init__(self, config=None):
        self.config = config or {}

    def score(self, symbol, df):

        try:
            # ---------------- SAFETY ---------------- #
            if df is None or df.empty:
                return {"score": 0}

            if "Volume" not in df.columns:
                return {"score": 0}

            vol_series = df["Volume"].dropna()

            if len(vol_series) < 20:
                return {"score": 0}

            # ---------------- SAFE SCALAR ---------------- #
            latest_vol = float(vol_series.iloc[-1].item())
            avg_vol_20 = float(vol_series.tail(20).mean())

            if avg_vol_20 == 0:
                return {"score": 0}

            ratio = latest_vol / avg_vol_20

            # ---------------- SCORING ---------------- #
            score = 0

            if ratio >= 2.5:
                score = 30   # strong breakout volume

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
                "spike_ratio": round(ratio, 2),
                "avg_volume_20": int(avg_vol_20),
                "latest_volume": int(latest_vol)
            }

        except Exception as e:
            logger.error(f"{symbol} volume error: {e}")
            return {"score": 0}
