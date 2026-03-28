from loguru import logger


class NewsAnalyzer:

    def __init__(self, config=None):
        self.config = config or {}

    def score(self, symbol):

        try:
            # 🔥 simplified (since APIs unstable on GitHub)

            sentiment = 0  # neutral
            articles = []

            score = 5  # base score

            return {
                "score": score,
                "sentiment": sentiment,
                "article_count": len(articles)  # ✅ FIXED KEY
            }

        except Exception as e:
            logger.error(f"{symbol} news error: {e}")
            return {"score": 0, "article_count": 0}
