"""
news_analyzer.py
----------------
Fetches and scores news sentiment from multiple sources:
  - NewsAPI (international + domestic)
  - RSS feeds (ET, Moneycontrol, Business Standard)
  - NLP using VADER + optional HuggingFace FinBERT
"""

import os
import re
import time
import feedparser
import requests
from datetime import datetime, timedelta
from loguru import logger
from typing import List, Dict, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

VADER = SentimentIntensityAnalyzer()

# Finance-specific RSS feeds (no auth required)
RSS_FEEDS = {
    "economic_times":    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "moneycontrol":      "https://www.moneycontrol.com/rss/business.xml",
    "business_standard": "https://www.business-standard.com/rss/markets-106.rss",
    "livemint":          "https://www.livemint.com/rss/markets",
    "bloomberg_india":   "https://feeds.bloomberg.com/markets/news.rss",
    "reuters_markets":   "https://feeds.reuters.com/reuters/businessNews",
}

# Symbols that often appear as aliases in news
SYMBOL_ALIASES = {
    "RELIANCE":    ["reliance industries", "reliance jio", "ril"],
    "TCS":         ["tata consultancy", "tcs"],
    "HDFCBANK":    ["hdfc bank", "hdfc"],
    "ICICIBANK":   ["icici bank", "icici"],
    "SBIN":        ["state bank", "sbi"],
    "TATAMOTORS":  ["tata motors", "jaguar land rover", "jlr"],
    "INFY":        ["infosys"],
    "WIPRO":       ["wipro"],
    "BHARTIARTL":  ["bharti airtel", "airtel"],
    "BAJFINANCE":  ["bajaj finance"],
}


class NewsAnalyzer:
    def __init__(self, config: dict):
        self.cfg = config["signals"]["news"]
        self.api_key = os.getenv("NEWS_API_KEY", "")
        self.lookback_hours = self.cfg.get("lookback_hours", 24)
        self.pos_threshold = self.cfg.get("positive_threshold", 0.25)
        self.neg_threshold = self.cfg.get("negative_threshold", -0.25)
        self._news_cache: Dict[str, list] = {}
        self._last_fetch: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    #  Feed Fetching
    # ------------------------------------------------------------------ #

    def _fetch_rss_articles(self) -> List[dict]:
        """Fetch articles from all RSS feeds. Cached per run."""
        if self._last_fetch and (datetime.now() - self._last_fetch).seconds < 600:
            return list(self._news_cache.get("rss", []))

        articles = []
        cutoff = datetime.now() - timedelta(hours=self.lookback_hours)

        for source, url in RSS_FEEDS.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    pub = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        pub = datetime(*entry.published_parsed[:6])
                    if pub and pub < cutoff:
                        continue
                    articles.append({
                        "source": source,
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", ""),
                        "published": pub,
                    })
            except Exception as e:
                logger.warning(f"RSS fetch failed ({source}): {e}")

        self._news_cache["rss"] = articles
        self._last_fetch = datetime.now()
        logger.info(f"Fetched {len(articles)} RSS articles from {len(RSS_FEEDS)} feeds")
        return articles

    def _fetch_newsapi_articles(self, query: str) -> List[dict]:
        """Fetch from NewsAPI for a specific query."""
        if not self.api_key:
            return []
        try:
            from_date = (datetime.now() - timedelta(hours=self.lookback_hours)).strftime("%Y-%m-%dT%H:%M:%S")
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "from": from_date,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 20,
                "apiKey": self.api_key,
            }
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            articles = []
            for art in data.get("articles", []):
                articles.append({
                    "source": art.get("source", {}).get("name", "newsapi"),
                    "title": art.get("title", ""),
                    "summary": art.get("description", ""),
                    "published": art.get("publishedAt"),
                })
            return articles
        except Exception as e:
            logger.warning(f"NewsAPI fetch failed: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  Symbol Matching
    # ------------------------------------------------------------------ #

    def _get_search_terms(self, symbol: str) -> List[str]:
        """Return all text aliases for a symbol."""
        terms = [symbol.lower()]
        aliases = SYMBOL_ALIASES.get(symbol.upper(), [])
        terms.extend(aliases)
        return terms

    def _filter_articles_for_symbol(self, symbol: str, articles: List[dict]) -> List[dict]:
        """Return articles that mention the symbol or its aliases."""
        terms = self._get_search_terms(symbol)
        matched = []
        for art in articles:
            text = (art["title"] + " " + art.get("summary", "")).lower()
            if any(term in text for term in terms):
                matched.append(art)
        return matched

    # ------------------------------------------------------------------ #
    #  Sentiment Scoring
    # ------------------------------------------------------------------ #

    def _vader_score(self, text: str) -> float:
        """VADER compound sentiment score (-1 to +1)."""
        return VADER.polarity_scores(text)["compound"]

    def _score_articles(self, articles: List[dict]) -> dict:
        """Score a list of articles and return aggregated sentiment."""
        if not articles:
            return {"avg_score": 0.0, "positive": 0, "negative": 0, "neutral": 0, "headlines": []}

        scores = []
        breakdown = {"positive": 0, "negative": 0, "neutral": 0}
        headlines = []

        for art in articles[:20]:  # Cap at 20 articles
            text = art["title"] + ". " + art.get("summary", "")
            score = self._vader_score(text)
            scores.append(score)
            if score >= self.pos_threshold:
                breakdown["positive"] += 1
            elif score <= self.neg_threshold:
                breakdown["negative"] += 1
            else:
                breakdown["neutral"] += 1
            headlines.append({
                "headline": art["title"][:120],
                "source": art["source"],
                "score": round(score, 3),
            })

        avg = sum(scores) / len(scores) if scores else 0.0
        return {
            "avg_score": round(avg, 3),
            "article_count": len(articles),
            **breakdown,
            "headlines": sorted(headlines, key=lambda x: abs(x["score"]), reverse=True)[:5],
        }

    # ------------------------------------------------------------------ #
    #  Global Market Sentiment
    # ------------------------------------------------------------------ #

    def get_global_market_sentiment(self) -> dict:
        """
        Analyze global macro news that affects Indian markets:
          - Fed policy / US rates
          - China economic data
          - Oil prices / commodities
          - Global risk-on / risk-off
        """
        global_queries = [
            "Federal Reserve interest rates", "US inflation CPI",
            "China economic growth", "crude oil prices",
            "global stock market rally", "emerging markets FII",
        ]
        all_articles = []
        for q in global_queries[:3]:  # Limit API calls
            all_articles.extend(self._fetch_newsapi_articles(q))

        result = self._score_articles(all_articles)
        result["risk_on"] = result["avg_score"] > 0.1
        return result

    # ------------------------------------------------------------------ #
    #  Per-Symbol Scoring
    # ------------------------------------------------------------------ #

    def score(self, symbol: str) -> dict:
        """
        News sentiment score (0-100) for a given symbol.
        Higher = more positive news sentiment.
        """
        rss_articles = self._fetch_rss_articles()
        sym_articles = self._filter_articles_for_symbol(symbol, rss_articles)

        # Supplement with NewsAPI if key available
        if self.api_key and len(sym_articles) < 3:
            company_name = " ".join(SYMBOL_ALIASES.get(symbol, [symbol.lower()])[:1])
            api_arts = self._fetch_newsapi_articles(f"{company_name} stock India")
            sym_articles.extend(api_arts)

        sentiment = self._score_articles(sym_articles)
        avg = sentiment["avg_score"]

        # Convert to 0–100 score
        # avg in [-1, +1] → map to [0, 100], centered at 50
        raw_score = int((avg + 1) / 2 * 100)
        score = max(0, min(100, raw_score))

        details = {}
        if avg >= self.pos_threshold:
            details["sentiment"] = f"Positive sentiment ({avg:.2f}) ✅"
        elif avg <= self.neg_threshold:
            details["sentiment"] = f"Negative sentiment ({avg:.2f}) ❌"
        else:
            details["sentiment"] = f"Neutral sentiment ({avg:.2f})"

        if sentiment["headlines"]:
            details["top_headline"] = sentiment["headlines"][0]["headline"]

        return {
            "score": score,
            "avg_sentiment": avg,
            "article_count": sentiment["article_count"],
            "positive_count": sentiment["positive"],
            "negative_count": sentiment["negative"],
            "headlines": sentiment["headlines"],
            "details": details,
        }
