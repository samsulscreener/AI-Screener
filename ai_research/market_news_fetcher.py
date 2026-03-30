def _safe_float(self, x, default=0.0):
    try:
        if x is None:
            return float(default)
        return float(x)
    except:
        return float(default)


def _compute_weighted_sentiment(self, articles):

    if not articles:
        return {
            "score": 0,
            "sentiment": "neutral",
            "confidence": 0
        }

    cleaned = []

    for a in articles:
        sentiment = self._safe_float(a.get("sentiment", 0), 0)
        relevance = self._safe_float(a.get("relevance", 0.5), 0.5)

        cleaned.append({
            "sentiment": sentiment,
            "relevance": relevance
        })

    # 🔥 total weight (safe)
    total_weight = sum(a["relevance"] for a in cleaned)

    if total_weight == 0:
        return {
            "score": 0,
            "sentiment": "neutral",
            "confidence": 0
        }

    # 🔥 weighted sentiment (safe)
    weighted_sentiment = sum(
        a["sentiment"] * a["relevance"]
        for a in cleaned
    ) / total_weight

    # 🔥 classification
    if weighted_sentiment > 0.2:
        label = "positive"
    elif weighted_sentiment < -0.2:
        label = "negative"
    else:
        label = "neutral"

    return {
        "score": round(weighted_sentiment, 3),
        "sentiment": label,
        "confidence": round(min(total_weight, 1.0), 2)
    }
