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
            }

            res = requests.get(url, params=params, timeout=10)

            if res.status_code != 200:
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
                "score": 0,
                "sentiment": "neutral",
                "confidence": 0
            }

        total_weight = sum(
            self._safe_float(a.get("relevance", 0.5), 0.5)
            for a in articles
        )

        if total_weight == 0:
            return {
                "score": 0,
                "sentiment": "neutral",
                "confidence": 0
            }

        weighted_sentiment = sum(
            self._safe_float(a.get("sentiment", 0), 0) *
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

        return {
            "articles": articles,
            "score": result["score"],
            "sentiment": result["sentiment"],
            "confidence": result["confidence"]
        }
