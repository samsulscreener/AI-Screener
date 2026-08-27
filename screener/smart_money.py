"""
fundamental_analyzer.py  (v2)
-----------------------------
FIXES vs v1:
  * SQLite cache with a TTL. v1 scraped Screener.in once per symbol per run:
    7 runs/day x 500 symbols = 3,500 requests/day to a site whose data changes
    once a quarter. You get rate-limited, the scrape returns {}, and 15-20% of
    your weighting budget silently zeroes out. Same for the two NSE endpoints.
  * Score is a genuine 0-100 with an explicit point budget, and components
    with missing data are dropped and renormalised rather than scored 0.
  * HARD REJECT on promoter pledge. In Indian smallcaps a pledged promoter
    holding is the most reliable predictor of permanent capital loss. It must
    remove the stock from the universe, not deduct a few points.
  * Growth (sales/profit CAGR) is now scored — v1 had no growth term at all,
    which is the single most important fundamental input for anything beyond
    a 2-week horizon.
  * Polite rate limiting so you don't get your IP banned.
"""

import os
import re
import json
import time
import sqlite3
import threading
import requests
import pandas as pd
from loguru import logger
from datetime import datetime, timedelta

NSE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Referer": "https://www.nseindia.com/",
}
SCREENER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml",
}

_throttle_lock = threading.Lock()
_last_call = [0.0]
MIN_INTERVAL = 0.7          # seconds between Screener.in requests, globally


def _throttle():
    with _throttle_lock:
        wait = MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()


class FundamentalAnalyzer:

    def __init__(self, session: requests.Session, config: dict):
        self.session = session
        self.cfg = config.get("signals", {}).get("fundamental", {})
        self.enabled = bool(self.cfg.get("enabled", True))
        self.cache_days = int(self.cfg.get("cache_days", 7))
        self.db_path = config.get("output", {}).get("db_path", "data/screener.db")
        self.av_key = os.getenv("ALPHA_VANTAGE_KEY", "")
        self._init_cache()

    # ------------------------------------------------------------------ #
    #  Cache
    # ------------------------------------------------------------------ #

    def _init_cache(self):
        d = os.path.dirname(self.db_path)
        if d:
            os.makedirs(d, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals_cache (
                symbol      TEXT PRIMARY KEY,
                payload     TEXT,
                fetched_at  TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _cache_get(self, symbol):
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            row = conn.execute(
                "SELECT payload, fetched_at FROM fundamentals_cache WHERE symbol=?",
                (symbol,)).fetchone()
            conn.close()
            if not row:
                return None
            fetched = datetime.fromisoformat(row[1])
            if datetime.now() - fetched > timedelta(days=self.cache_days):
                return None
            return json.loads(row[0])
        except Exception:
            return None

    def _cache_put(self, symbol, payload):
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute(
                "INSERT OR REPLACE INTO fundamentals_cache VALUES (?,?,?)",
                (symbol, json.dumps(payload), datetime.now().isoformat()))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"fundamentals cache write failed for {symbol}: {e}")

    # ------------------------------------------------------------------ #
    #  Screener.in
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_num(text):
        if text is None:
            return None
        t = str(text).replace(",", "").replace("%", "").replace("₹", "").strip()
        m = re.search(r"-?\d+\.?\d*", t)
        return float(m.group()) if m else None

    def _scrape_screener(self, symbol):
        """Scrape ratios + growth tables from Screener.in. Returns {} on failure."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("beautifulsoup4 not installed; fundamentals unavailable")
            return {}

        out = {}
        for path in ("consolidated/", ""):
            url = f"https://www.screener.in/company/{symbol}/{path}"
            try:
                _throttle()
                resp = requests.get(url, timeout=15, headers=SCREENER_HEADERS)
                if resp.status_code != 200:
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")

                ratios = {}
                for li in soup.select("#top-ratios li"):
                    name_el = li.select_one(".name")
                    val_el = li.select_one(".value")
                    if name_el and val_el:
                        ratios[name_el.get_text(strip=True)] = val_el.get_text(" ", strip=True)

                if not ratios:
                    continue

                out = {
                    "roe":        self._to_num(ratios.get("ROE") or ratios.get("Return on equity")),
                    "roce":       self._to_num(ratios.get("ROCE") or ratios.get("Return on capital employed")),
                    "pe":         self._to_num(ratios.get("Stock P/E") or ratios.get("P/E")),
                    "market_cap": self._to_num(ratios.get("Market Cap")),
                    "book_value": self._to_num(ratios.get("Book Value")),
                    "div_yield":  self._to_num(ratios.get("Dividend Yield")),
                    "face_value": self._to_num(ratios.get("Face Value")),
                }

                # Debt/equity and pledge live in the ranges/quick-ratio sections
                page_text = soup.get_text(" ", strip=True)
                m = re.search(r"Debt to equity\s*([\d.]+)", page_text, re.I)
                if m:
                    out["debt_equity"] = float(m.group(1))
                m = re.search(r"Pledged percentage\s*([\d.]+)", page_text, re.I)
                out["pledge_pct"] = float(m.group(1)) if m else 0.0

                # Growth tables ("Compounded Sales Growth" / "Compounded Profit Growth")
                for label, key in (("Compounded Sales Growth", "sales_cagr"),
                                   ("Compounded Profit Growth", "profit_cagr")):
                    blk = re.search(
                        label + r"(.{0,400}?)(?:Compounded|Stock Price|Return on Equity|$)",
                        page_text, re.I | re.S)
                    if blk:
                        nums = re.findall(r"(-?\d+)%", blk.group(1))
                        if nums:
                            # order on the page: 10Y, 5Y, 3Y, TTM
                            out[key + "_5y"] = float(nums[1]) if len(nums) > 1 else float(nums[0])
                            out[key + "_3y"] = float(nums[2]) if len(nums) > 2 else None
                            out[key + "_ttm"] = float(nums[3]) if len(nums) > 3 else None

                if out.get("roe") is not None or out.get("pe") is not None:
                    break
            except Exception as e:
                logger.debug(f"Screener.in fetch failed for {symbol} ({path or 'standalone'}): {e}")
                continue

        return {k: v for k, v in out.items() if v is not None}

    def get_screener_data(self, symbol):
        cached = self._cache_get(symbol)
        if cached is not None:
            return cached
        data = self._scrape_screener(symbol)
        self._cache_put(symbol, data)      # cache empties too, so we don't retry all day
        return data

    # ------------------------------------------------------------------ #
    #  NSE corporate announcements
    # ------------------------------------------------------------------ #

    POSITIVE_KW = [
        "buyback", "buy back", "bonus issue", "merger", "acquisition", "amalgamation",
        "joint venture", "wins order", "bags order", "order win", "contract",
        "letter of intent", "expansion", "capacity", "capex", "commissioning",
        "record date", "interim dividend", "stake acquisition", "fund raise",
    ]
    NEGATIVE_KW = [
        "pledge", "invocation", "downgrade", "default", "insolvency", "nclt",
        "sebi", "show cause", "investigation", "penalty", "fraud", "litigation",
        "auditor resignation", "resignation of auditor", "qualified opinion",
        "resignation of managing director", "resignation of cfo", "delisting",
    ]

    def get_corporate_announcements(self, symbol, days_back=30):
        try:
            resp = self.session.get(
                f"https://www.nseindia.com/api/top-corp-info?symbol={symbol}&market=equities",
                headers=NSE_HEADERS, timeout=12)
            if resp.status_code != 200:
                return None
            data = resp.json()
        except Exception as e:
            logger.debug(f"announcements failed for {symbol}: {e}")
            return None

        cutoff = datetime.now() - timedelta(days=days_back)
        out = []
        raw = data.get("announcements", {})
        items = raw.get("data", raw) if isinstance(raw, dict) else raw
        for item in (items or []):
            try:
                d = pd.to_datetime(item.get("an_dt", item.get("sort_date", "")), errors="coerce")
                if pd.notna(d) and d.to_pydatetime().replace(tzinfo=None) < cutoff:
                    continue
                out.append({"date": str(d.date()) if pd.notna(d) else "",
                            "subject": item.get("subject", "") or item.get("desc", "")})
            except Exception:
                continue
        return out

    def _announcement_sentiment(self, announcements):
        pos = neg = 0
        flags = []
        for a in announcements:
            t = a["subject"].lower()
            for kw in self.NEGATIVE_KW:
                if kw in t:
                    neg += 1
                    flags.append(a["subject"][:70])
                    break
            else:
                for kw in self.POSITIVE_KW:
                    if kw in t:
                        pos += 1
                        flags.append(a["subject"][:70])
                        break
        return pos, neg, flags

    # ------------------------------------------------------------------ #
    #  Hard rejects
    # ------------------------------------------------------------------ #

    def hard_reject(self, symbol, ratios=None, announcements=None):
        """
        Returns (True, reason) if the stock should be removed from the universe
        entirely. These are not score deductions — the asymmetry justifies it:
        you have hundreds of candidates, and one fraud costs more than ten misses.
        """
        ratios = ratios if ratios is not None else self.get_screener_data(symbol)
        max_pledge = float(self.cfg.get("max_pledge_pct", 10))

        pledge = ratios.get("pledge_pct")
        if pledge is not None and pledge > max_pledge:
            return True, f"promoter pledge {pledge:.1f}% > {max_pledge}%"

        if announcements:
            for a in announcements:
                t = a["subject"].lower()
                if any(k in t for k in ("auditor resignation", "resignation of auditor",
                                        "qualified opinion", "insolvency", "nclt admitted")):
                    return True, f"filing red flag: {a['subject'][:60]}"

        return False, ""

    # ------------------------------------------------------------------ #
    #  Score  (genuine 0-100)
    # ------------------------------------------------------------------ #
    #  Profitability (ROE + ROCE)   30
    #  Growth (sales + profit CAGR) 25
    #  Balance sheet (D/E)          15
    #  Valuation (PE sanity)        10
    #  Corporate announcements      20
    #                              ----
    #                              100
    # ------------------------------------------------------------------ #

    def score(self, symbol):
        if not self.enabled:
            return {"score": 0.0, "available": False, "reason": "fundamentals disabled"}

        pts = 0.0
        budget = 0.0
        details = {}

        ratios = self.get_screener_data(symbol)
        anns = self.get_corporate_announcements(symbol)

        rejected, reason = self.hard_reject(symbol, ratios, anns)
        if rejected:
            return {"score": 0.0, "available": True, "hard_reject": True,
                    "reject_reason": reason, "ratios": ratios, "details": {}}

        # --- Profitability: 30 ---
        roe = ratios.get("roe")
        roce = ratios.get("roce")
        if roe is not None or roce is not None:
            budget += 30
            min_roe = float(self.cfg.get("min_roe", 12))
            min_roce = float(self.cfg.get("min_roce", 15))
            if roe is not None:
                if roe >= min_roe * 1.5:
                    pts += 15
                elif roe >= min_roe:
                    pts += 11
                elif roe > 0:
                    pts += 4
                details["roe"] = roe
            if roce is not None:
                if roce >= min_roce * 1.5:
                    pts += 15
                elif roce >= min_roce:
                    pts += 11
                elif roce > 0:
                    pts += 4
                details["roce"] = roce

        # --- Growth: 25 ---
        s5 = ratios.get("sales_cagr_5y")
        p5 = ratios.get("profit_cagr_5y")
        if s5 is not None or p5 is not None:
            budget += 25
            if s5 is not None:
                if s5 >= 20:
                    pts += 13
                elif s5 >= 12:
                    pts += 9
                elif s5 >= 5:
                    pts += 4
                details["sales_cagr_5y"] = s5
            if p5 is not None:
                if p5 >= 20:
                    pts += 12
                elif p5 >= 12:
                    pts += 8
                elif p5 >= 5:
                    pts += 4
                details["profit_cagr_5y"] = p5

        # --- Balance sheet: 15 ---
        de = ratios.get("debt_equity")
        if de is not None:
            budget += 15
            max_de = float(self.cfg.get("max_debt_equity", 2.0))
            if de <= 0.3:
                pts += 15
            elif de <= max_de / 2:
                pts += 11
            elif de <= max_de:
                pts += 6
            details["debt_equity"] = de

        # --- Valuation sanity: 10 ---
        pe = ratios.get("pe")
        if pe is not None:
            budget += 10
            min_pe = float(self.cfg.get("min_pe", 3))
            max_pe = float(self.cfg.get("max_pe", 60))
            if min_pe <= pe <= max_pe * 0.5:
                pts += 10
            elif min_pe <= pe <= max_pe:
                pts += 6
            details["pe"] = pe

        # --- Announcements: 20 ---
        if anns is not None:
            budget += 20
            pos, neg, flags = self._announcement_sentiment(anns)
            net = pos - neg
            if net >= 2:
                pts += 20
            elif net == 1:
                pts += 14
            elif net == 0:
                pts += 10
            elif net == -1:
                pts += 4
            details["announcements_net"] = net
            if flags:
                details["announcement_sample"] = flags[0]

        if budget == 0:
            return {"score": 0.0, "available": False,
                    "reason": "no fundamental data", "ratios": {}, "details": {}}

        return {
            "score": round(pts / budget * 100.0, 1),
            "available": True,
            "hard_reject": False,
            "components_used": int(budget),
            "ratios": ratios,
            "details": details,
        }
