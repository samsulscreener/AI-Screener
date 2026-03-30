from loguru import logger
import pandas as pd


class VolumeAnalyzer:

    def __init__(self, session=None, config=None):
        self.session = session
        self.config = config or {}

    def _get_series(self, df, col):
        data = df[col]

        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]

        return data.dropna()

    def score(self, symbol, df, delivery_df=None):

        try:
            if df is None or df.empty:
                return {"score": 0}

            vol = self._get_series(df, "Volume")

            if len(vol) < 20:
                return {"score": 0}

            latest_vol = float(vol.iloc[-1])
            avg_vol = float(vol.tail(20).mean())

            if avg_vol == 0:
                return {"score": 0}

            ratio = latest_vol / avg_vol

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
