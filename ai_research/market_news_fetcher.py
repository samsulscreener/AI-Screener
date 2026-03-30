from loguru import logger
import requests


class MarketNewsFetcher:

    def __init__(self, config=None):
        self.config = config or {}
        self.api_key = self.config.get("marketaux_api_key", "")

    # ---------------- SAFE FLOAT ---------------- #

    def _safe_float(self, x, default=0.0):
        try:
            if x is None:
                return float(default)

            if isinstance(x, str):
                x = x.strip()
                if x == "":
                    return float(default)

            return float(x)
        except:
            return float(default)

    # ---------------- FETCH ---------------- #

    def fetch_marketaux(self, symbol):

        if not self.api_key:
            return []

        try:
            url = "https://api.marketaux.com/v1/news/all"

            params = {
                "symbols": symbol,
                "language": "en",
                "api_token": self.api_key,
                "limit": 10,
            }

            res = requests.get(url, params=params, timeout=10)

            if res.status_code != 200:
                logger.warning(f"{symbol} news API failed: {res.status_code}")
                return []

            data = res.json()

            articles = []

            for item in data.get("data", []):
                articles.append({
                    "title": item.get("title"),
                    "sentiment": self._safe_float(item.get("sentiment", 0)),
                    "relevance": self._safe_float(item.get("relevance_score", 0.5), 0.5),
                })

            return articles

        except Exception as e:
            logger.warning(f"News fetch failed for {symbol}: {e}")
            return []

    # ---------------- SENTIMENT ---------------- #

    def _compute_weighted_sentiment(self, articles):

        if not articles:
            return {
                "score": 0.0,
                "sentiment": "neutral",
                "confidence": 0.0
            }

        total_weight = sum(
            self._safe_float(a.get("relevance", 0.5), 0.5)
            for a in articles
        )

        if total_weight == 0:
            return {
                "score": 0.0,
                "sentiment": "neutral",
                "confidence": 0.0
            }

        weighted_sentiment = sum(
            self._safe_float(a.get("sentiment", 0), 0.0) *
            self._safe_float(a.get("relevance", 0.5), 0.5)
            for a in articles
        ) / total_weight

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

    # ---------------- PUBLIC API ---------------- #

    def fetch_all_news(self, symbol):

        articles = self.fetch_marketaux(symbol)

        result = self._compute_weighted_sentiment(articles)

        # 🔥 CRITICAL FIX: add article_count + safe fields
        return {
            "articles": articles,
            "article_count": len(articles),   # ✅ FIXED (your crash)
            "score": float(result.get("score", 0)),
            "sentiment": result.get("sentiment", "neutral"),
            "confidence": float(result.get("confidence", 0)),
        }
