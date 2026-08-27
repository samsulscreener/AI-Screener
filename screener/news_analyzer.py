"""
news_analyzer.py  (v2)
----------------------
FIXES vs v1:
  * v1 returned `score = 5` for EVERY symbol, ALWAYS. That constant was
    weighted at 15% — a sixth of the scoring budget spent on a number that
    added the same 1.5 points to every stock and changed no rankings.
  * v2 either (a) computes a real headline sentiment from Google News RSS,
    or (b) reports available=False so the scorer DROPS the component and
    renormalises the remaining weights. A missing signal must never be
    faked as a constant.
  * Uses only the standard library for parsing (no feedparser/vader
    dependency), with a finance-specific lexicon. This is deliberately a
    weak signal — keep its weight low (0.05) until your Phase 1 backtest
    proves it earns more.
"""

import re
import time
import html
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import requests
from loguru import logger

# Finance-specific lexicon. Deliberately small and high-precision:
# a generic sentiment model mislabels ordinary market vocabulary.
POSITIVE = {
    "beats": 2.0, "beat": 2.0, "surges": 2.0, "surge": 1.5, "jumps": 1.5,
    "record high": 2.0, "all-time high": 2.0, "upgrade": 2.0, "upgrades": 2.0,
    "buy rating": 2.0, "target raised": 2.5, "wins order": 2.5, "bags order": 2.5,
    "new order": 2.0, "contract win": 2.5, "buyback": 2.0, "dividend": 1.0,
    "expansion": 1.5, "capacity addition": 1.5, "acquisition": 1.0,
    "profit rises": 2.0, "profit jumps": 2.5, "revenue growth": 1.5,
    "margin expansion": 2.0, "stake buy": 1.5, "bonus issue": 1.0,
    "outperform": 1.5, "multibagger": 0.5, "turnaround": 1.5,
}
NEGATIVE = {
    "falls": -1.5, "plunges": -2.5, "slumps": -2.0, "crashes": -2.5,
    "downgrade": -2.0, "downgrades": -2.0, "sell rating": -2.0,
    "target cut": -2.5, "profit falls": -2.0, "loss": -2.0, "losses": -2.0,
    "fraud": -3.0, "sebi": -1.5, "probe": -2.0, "investigation": -2.5,
    "raid": -2.5, "default": -3.0, "insolvency": -3.0, "nclt": -2.5,
    "pledge": -2.0, "pledged": -2.0, "resigns": -1.5, "resignation": -1.5,
    "auditor resigns": -3.0, "qualified opinion": -3.0, "penalty": -2.0,
    "recall": -2.0, "stake sale": -1.0, "block deal": -0.5, "downgraded": -2.0,
}

RSS_URL = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"


class NewsAnalyzer:

    def __init__(self, config=None):
        cfg = (config or {}).get("signals", {}).get("news", {})
        self.enabled       = bool(cfg.get("enabled", True))
        self.lookback_h    = int(cfg.get("lookback_hours", 48))
        self.max_articles  = int(cfg.get("max_articles", 12))
        self.pos_threshold = float(cfg.get("positive_threshold", 0.15))
        self.neg_threshold = float(cfg.get("negative_threshold", -0.15))
        self._cache = {}          # symbol -> (timestamp, result)
        self._cache_ttl = 1800    # 30 minutes; headlines don't change faster

    # ------------------------------------------------------------------ #

    def fetch_headlines(self, symbol, company_name=None):
        query = f'"{company_name or symbol}" NSE stock'
        url = RSS_URL.format(q=quote_plus(query))
        try:
            r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                return None
            root = ET.fromstring(r.content)
        except Exception as e:
            logger.debug(f"{symbol} news RSS failed: {e}")
            return None

        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.lookback_h)
        items = []
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            if not title:
                continue
            pub = item.findtext("pubDate")
            when = None
            if pub:
                for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"):
                    try:
                        when = datetime.strptime(pub, fmt)
                        if when.tzinfo is None:
                            when = when.replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        continue
            if when and when < cutoff:
                continue
            items.append({"title": html.unescape(title), "published": pub})
            if len(items) >= self.max_articles:
                break
        return items

    @staticmethod
    def score_headline(text):
        t = " " + re.sub(r"\s+", " ", text.lower()) + " "
        s = 0.0
        hits = []
        for phrase, w in POSITIVE.items():
            if phrase in t:
                s += w
                hits.append(phrase)
        for phrase, w in NEGATIVE.items():
            if phrase in t:
                s += w
                hits.append(phrase)
        # squash to [-1, 1] so one shouty headline can't dominate
        return max(-1.0, min(1.0, s / 4.0)), hits

    # ------------------------------------------------------------------ #

    def score(self, symbol, company_name=None):
        if not self.enabled:
            return {"score": 0.0, "available": False, "reason": "news disabled"}

        cached = self._cache.get(symbol)
        if cached and time.time() - cached[0] < self._cache_ttl:
            return cached[1]

        try:
            items = self.fetch_headlines(symbol, company_name)

            if items is None:
                # fetch failed -> unknown, NOT neutral. Drop the component.
                result = {"score": 0.0, "available": False, "reason": "news fetch failed",
                          "article_count": 0, "headlines": []}
                self._cache[symbol] = (time.time(), result)
                return result

            if not items:
                # genuinely no news. That IS information: no catalyst.
                result = {"score": 50.0, "available": True, "sentiment": 0.0,
                          "article_count": 0, "headlines": [],
                          "label": "no recent coverage"}
                self._cache[symbol] = (time.time(), result)
                return result

            scored = []
            for it in items:
                s, hits = self.score_headline(it["title"])
                scored.append({"headline": it["title"], "sentiment": round(s, 3), "hits": hits})

            # weight recent headlines slightly higher (they're returned newest-first)
            weights = [1.0 / (1 + 0.15 * i) for i in range(len(scored))]
            total_w = sum(weights)
            avg = sum(a["sentiment"] * w for a, w in zip(scored, weights)) / total_w

            # map [-1, 1] -> [0, 100], neutral = 50
            score = round((avg + 1.0) * 50.0, 1)

            if avg >= self.pos_threshold:
                label = "positive"
            elif avg <= self.neg_threshold:
                label = "negative"
            else:
                label = "neutral"

            result = {
                "score": score,
                "available": True,
                "sentiment": round(avg, 3),
                "label": label,
                "article_count": len(scored),
                "headlines": scored[:5],
            }
            self._cache[symbol] = (time.time(), result)
            return result

        except Exception as e:
            logger.error(f"{symbol} news failed: {e}")
            return {"score": 0.0, "available": False, "reason": str(e),
                    "article_count": 0, "headlines": []}
