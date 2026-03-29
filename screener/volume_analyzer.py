def score(self, symbol, df):

    try:
        if df is None or df.empty:
            return {"score": 0}

        vol_series = df["Volume"].dropna()

        if len(vol_series) < 20:
            return {"score": 0}

        latest_vol = float(vol_series.iloc[-1])
        avg_vol = float(vol_series.tail(20).mean())

        ratio = latest_vol / (avg_vol + 1e-9)

        if ratio > 2:
            score = 30
        elif ratio > 1.5:
            score = 20
        elif ratio > 1.2:
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
