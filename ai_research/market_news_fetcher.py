"""
market_news_fetcher.py
-----------------------
Fetches rich, ticker-specific news with built-in sentiment scores from:

  1. MarketAux API (marketaux.com)
     - Best free API for stock-specific news
     - Returns relevance score + sentiment per article
     - Free: 100 requests/day | Endpoint: /v1/news/all

  2. Alpha Vantage News Sentiment
     - Tied to fundamentals data already in the stack
     - Free: 25 req/day | Endpoint: NEWS_SENTIMENT

  3. NSE Corporate Actions (no API key needed)
     - Dividends, buybacks, board meetings, results dates

Sign up for MarketAux: https://www.marketaux.com  (free tier, no CC required)
"""

import os
import time
import requests
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from loguru import logger

# ── NSE symbol → company name map for better search queries ──────────────────
NSE_COMPANY_NAMES = {
    "RELIANCE":   "Reliance Industries",
    "TCS":        "Tata Consultancy Services",
    "HDFCBANK":   "HDFC Bank",
    "ICICIBANK":  "ICICI Bank",
    "INFY":       "Infosys",
    "SBIN":       "State Bank of India",
    "BHARTIARTL": "Bharti Airtel",
    "ITC":        "ITC Limited",
    "KOTAKBANK":  "Kotak Mahindra Bank",
    "LT":         "Larsen Toubro",
    "AXISBANK":   "Axis Bank",
    "BAJFINANCE": "Bajaj Finance",
    "HCLTECH":    "HCL Technologies",
    "WIPRO":      "Wipro",
    "TATAMOTORS": "Tata Motors",
    "MARUTI":     "Maruti Suzuki",
    "SUNPHARMA":  "Sun Pharmaceutical",
    "TITAN":      "Titan Company",
    "ULTRACEMCO": "UltraTech Cement",
    "NESTLEIND":  "Nestle India",
    "NTPC":       "NTPC",
    "ONGC":       "ONGC",
    "DRREDDY":    "Dr Reddys Laboratories",
    "POWERGRID":  "Power Grid Corporation",
    "M&M":        "Mahindra Mahindra",
    "ADANIENT":   "Adani Enterprises",
    "ADANIPORTS": "Adani Ports",
    "TATAPOWER":  "Tata Power",
    "TATASTEEL":  "Tata Steel",
    "HINDALCO":   "Hindalco Industries",
    "JSWSTEEL":   "JSW Steel",
    "COALINDIA":  "Coal India",
    "GRASIM":     "Grasim Industries",
    "DIVISLAB":   "Divi's Laboratories",
    "BPCL":       "Bharat Petroleum",
    "EICHERMOT":  "Eicher Motors",
    "HEROMOTOCO": "Hero MotoCorp",
    "APOLLOHOSP": "Apollo Hospitals",
    "BAJAJFINSV": "Bajaj Finserv",
    "INDUSINDBK": "IndusInd Bank",
    "CIPLA":      "Cipla",
}

# ── Global macro queries that affect Indian markets ───────────────────────────
GLOBAL_MACRO_QUERIES = [
    "Federal Reserve interest rates",
    "US inflation CPI jobs",
    "China economic data PMI",
    "crude oil OPEC prices",
    "dollar index DXY emerging markets",
    "India GDP RBI monetary policy",
    "FII FPI India equity flows",
]


class MarketNewsFetcher:
    def __init__(self, config: dict = None):
        self.marketaux_key  = os.getenv("MARKETAUX_API_KEY", "")
        self.newsapi_key    = os.getenv("NEWS_API_KEY", "")
        self.av_key         = os.getenv("ALPHA_VANTAGE_KEY", "")
        self._article_cache: Dict[str, dict] = {}
        self._cache_ttl_min = 30

    # ─────────────────────────────────────────────────────────────────────────
    #  MarketAux API  (Primary)
    # ─────────────────────────────────────────────────────────────────────────

    def fetch_marketaux(self, symbol: str, days_back: int = 2, limit: int = 15) -> List[dict]:
        """
        Fetch news from MarketAux API for a specific NSE stock.

        API docs: https://www.marketaux.com/documentation
        Endpoint: GET /v1/news/all
        Free tier: 100 req/day

        Returns list of articles with:
          - title, description, url, published_at
          - relevance_score (0-1) for this entity
          - sentiment_score (-1 to +1)  <-- built-in, no NLP needed!
          - entities: [{symbol, name, country}]
        """
        if not self.marketaux_key:
            logger.debug("MARKETAUX_API_KEY not set — skipping MarketAux fetch")
            return []

        cache_key = f"mx_{symbol}_{days_back}"
        if self._is_cached(cache_key):
            return self._article_cache[cache_key]["data"]

        # Build search query: try NSE symbol + company name
        company = NSE_COMPANY_NAMES.get(symbol.upper(), symbol)
        # MarketAux accepts ticker symbols for major global exchanges;
        # for NSE we search by entity name text
        published_after = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M")

        params = {
            "api_token":       self.marketaux_key,
            "search":          company,
            "language":        "en",
            "published_after": published_after,
            "limit":           limit,
            "sort":            "published_at",
            "sort_order":      "desc",
            # Filter for finance & markets topics
            "filter_entities": "true",
            "must_have_entities": "false",
        }

        try:
            resp = requests.get(
                "https://api.marketaux.com/v1/news/all",
                params=params,
                timeout=15,
            )
            data = resp.json()

            if resp.status_code != 200:
                logger.warning(f"MarketAux error {resp.status_code}: {data.get('error', {}).get('message', '')}")
                return []

            articles = []
            for art in data.get("data", []):
                # Extract sentiment from entities if available
                entity_sentiment = 0.0
                for ent in art.get("entities", []):
                    if symbol.upper() in str(ent.get("symbol", "")).upper() or \
                       company.lower() in str(ent.get("name", "")).lower():
                        entity_sentiment = ent.get("sentiment_score", 0.0)
                        break

                # Fall back to article-level sentiment
                highlights = art.get("highlights", [])
                hl_sentiment = highlights[0].get("sentiment", 0.0) if highlights else 0.0
                final_sentiment = entity_sentiment if entity_sentiment != 0 else hl_sentiment

                articles.append({
                    "source":         "marketaux",
                    "title":          art.get("title", ""),
                    "description":    art.get("description", ""),
                    "url":            art.get("url", ""),
                    "published_at":   art.get("published_at", ""),
                    "source_name":    art.get("source", ""),
                    "sentiment":      round(float(final_sentiment), 3),
                    "relevance":      art.get("relevance_score", 0.5),
                    "image_url":      art.get("image_url", ""),
                })

            logger.info(f"MarketAux: {len(articles)} articles for {symbol}")
            self._cache_set(cache_key, articles)
            return articles

        except Exception as e:
            logger.error(f"MarketAux fetch failed for {symbol}: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    #  Alpha Vantage News Sentiment  (Secondary)
    # ─────────────────────────────────────────────────────────────────────────

    def fetch_alpha_vantage_news(self, symbol: str, limit: int = 10) -> List[dict]:
        """
        Fetch news + sentiment from Alpha Vantage.
        Endpoint: TIME_SERIES_INTRADAY / NEWS_SENTIMENT

        Free: 25 req/day
        AV uses NSE: prefix for Indian stocks e.g. NSE:RELIANCE
        """
        if not self.av_key:
            return []

        cache_key = f"av_news_{symbol}"
        if self._is_cached(cache_key):
            return self._article_cache[cache_key]["data"]

        params = {
            "function":  "NEWS_SENTIMENT",
            "tickers":   f"NSE:{symbol}",
            "limit":     limit,
            "sort":      "LATEST",
            "apikey":    self.av_key,
        }
        try:
            resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=15)
            data = resp.json()
            articles = []
            for item in data.get("feed", []):
                # Find per-ticker sentiment
                ticker_sentiment = 0.0
                for ts in item.get("ticker_sentiment", []):
                    if f"NSE:{symbol}" in ts.get("ticker", "") or symbol in ts.get("ticker", ""):
                        ticker_sentiment = float(ts.get("ticker_sentiment_score", 0))
                        break

                articles.append({
                    "source":       "alphavantage",
                    "title":        item.get("title", ""),
                    "description":  item.get("summary", ""),
                    "url":          item.get("url", ""),
                    "published_at": item.get("time_published", ""),
                    "source_name":  item.get("source", ""),
                    "sentiment":    ticker_sentiment,
                    "relevance":    0.6,
                })
            logger.info(f"Alpha Vantage news: {len(articles)} articles for {symbol}")
            self._cache_set(cache_key, articles)
            return articles
        except Exception as e:
            logger.warning(f"AV news fetch failed for {symbol}: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    #  Global Macro News (MarketAux)
    # ─────────────────────────────────────────────────────────────────────────

    def fetch_global_macro_news(self, hours_back: int = 12) -> List[dict]:
        """Fetch macro news affecting Indian markets from MarketAux."""
        if not self.marketaux_key:
            return self._fetch_macro_rss()

        cache_key = f"macro_{hours_back}"
        if self._is_cached(cache_key, ttl_min=60):
            return self._article_cache[cache_key]["data"]

        published_after = (datetime.utcnow() - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M")
        params = {
            "api_token":       self.marketaux_key,
            "search":          "Federal Reserve India market FII crude oil",
            "language":        "en",
            "published_after": published_after,
            "limit":           20,
            "sort":            "relevance_score",
        }
        try:
            resp = requests.get("https://api.marketaux.com/v1/news/all", params=params, timeout=15)
            data = resp.json()
            articles = []
            for art in data.get("data", []):
                articles.append({
                    "source":       "marketaux",
                    "title":        art.get("title", ""),
                    "description":  art.get("description", ""),
                    "url":          art.get("url", ""),
                    "published_at": art.get("published_at", ""),
                    "source_name":  art.get("source", ""),
                    "sentiment":    0.0,
                    "relevance":    art.get("relevance_score", 0.5),
                })
            logger.info(f"Global macro news: {len(articles)} articles")
            self._cache_set(cache_key, articles)
            return articles
        except Exception as e:
            logger.warning(f"Macro news fetch failed: {e}")
            return self._fetch_macro_rss()

    def _fetch_macro_rss(self) -> List[dict]:
        """RSS fallback for macro news (no API key needed)."""
        feeds = {
            "Reuters Business":  "https://feeds.reuters.com/reuters/businessNews",
            "Bloomberg Markets": "https://feeds.bloomberg.com/markets/news.rss",
            "ET Markets":        "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
        }
        articles = []
        cutoff = datetime.utcnow() - timedelta(hours=24)
        for source, url in feeds.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]:
                    articles.append({
                        "source":       "rss",
                        "title":        entry.get("title", ""),
                        "description":  entry.get("summary", "")[:300],
                        "url":          entry.get("link", ""),
                        "published_at": str(getattr(entry, "published", "")),
                        "source_name":  source,
                        "sentiment":    0.0,
                        "relevance":    0.5,
                    })
            except Exception:
                pass
        return articles

    # ─────────────────────────────────────────────────────────────────────────
    #  NSE Corporate Actions
    # ─────────────────────────────────────────────────────────────────────────

    def get_nse_corporate_actions(self, symbol: str) -> List[dict]:
        """Fetch upcoming corporate actions from NSE (no API key required)."""
        url = "https://www.nseindia.com/api/corporates-corporateActions"
        params = {"index": "equities", "symbol": symbol}
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com"}
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            data = resp.json()
            actions = []
            for item in data[:5]:
                actions.append({
                    "symbol":       symbol,
                    "action":       item.get("subject", ""),
                    "record_date":  item.get("recordDate", ""),
                    "ex_date":      item.get("exDate", ""),
                    "purpose":      item.get("purpose", ""),
                })
            return actions
        except Exception as e:
            logger.warning(f"NSE corp actions fetch failed for {symbol}: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    #  Combined fetch for a symbol
    # ─────────────────────────────────────────────────────────────────────────

    def fetch_all_news(self, symbol: str) -> dict:
        """
        Aggregate all news sources for a symbol.
        Returns a dict with:
          - articles: list of all articles (deduped, sorted by date)
          - avg_sentiment: weighted average sentiment
          - article_count: total articles found
          - top_headlines: top 5 most relevant/recent
          - corporate_actions: upcoming events
        """
        articles = []

        # Primary: MarketAux (has built-in sentiment)
        mx_articles = self.fetch_marketaux(symbol, days_back=2)
        articles.extend(mx_articles)

        # Secondary: Alpha Vantage (if MarketAux is empty)
        if len(articles) < 3:
            av_articles = self.fetch_alpha_vantage_news(symbol)
            articles.extend(av_articles)

        # Deduplicate by title similarity
        seen_titles = set()
        unique_articles = []
        for art in articles:
            title_key = art["title"][:50].lower().strip()
            if title_key not in seen_titles and art["title"]:
                seen_titles.add(title_key)
                unique_articles.append(art)

        # Sort by published date (newest first)
        unique_articles.sort(key=lambda x: x.get("published_at", ""), reverse=True)

        # Weighted avg sentiment (weight by relevance)
        if unique_articles:
            total_weight = sum(
              float(a.get("relevance", 0.5) or 0.5)
              for a in unique_articles
            )
            weighted_sentiment = sum(
                a.get("sentiment", 0) * a.get("relevance", 0.5)
                for a in unique_articles
            ) / total_weight if total_weight > 0 else 0.0
        else:
            weighted_sentiment = 0.0

        # Corporate actions
        corp_actions = self.get_nse_corporate_actions(symbol)

        # Top headlines for Groq triage
        top_headlines = [
            f"[{a['source_name']}] {a['title']} (sentiment: {a['sentiment']:+.2f})"
            for a in unique_articles[:8]
        ]

        # Full article text for Gemini deep research
        full_articles_text = "\n\n".join([
            f"SOURCE: {a['source_name']}\nDATE: {a['published_at'][:16]}\n"
            f"TITLE: {a['title']}\nSUMMARY: {a['description'][:400]}\n"
            f"SENTIMENT: {a['sentiment']:+.3f} | URL: {a['url']}"
            for a in unique_articles[:12]
        ])

        return {
            "articles":            unique_articles,
            "avg_sentiment":       round(weighted_sentiment, 3),
            "article_count":       len(unique_articles),
            "top_headlines":       top_headlines,
            "full_articles_text":  full_articles_text,
            "corporate_actions":   corp_actions,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Cache helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _is_cached(self, key: str, ttl_min: int = None) -> bool:
        ttl = ttl_min or self._cache_ttl_min
        if key in self._article_cache:
            age_min = (datetime.utcnow() - self._article_cache[key]["ts"]).seconds / 60
            return age_min < ttl
        return False

    def _cache_set(self, key: str, data: list):
        self._article_cache[key] = {"data": data, "ts": datetime.utcnow()}

    def get_marketaux_remaining_quota(self) -> str:
        """Check remaining API quota for debugging."""
        if not self.marketaux_key:
            return "No key configured"
        try:
            resp = requests.get(
                "https://api.marketaux.com/v1/news/all",
                params={"api_token": self.marketaux_key, "limit": 1},
                timeout=10,
            )
            headers = resp.headers
            remaining = headers.get("X-RateLimit-Remaining", "?")
            limit = headers.get("X-RateLimit-Limit", "?")
            return f"{remaining}/{limit} requests remaining today"
        except Exception:
            return "Unable to check quota"
