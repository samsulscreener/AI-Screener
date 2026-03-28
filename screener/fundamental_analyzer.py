"""
fundamental_analyzer.py
-----------------------
Tracks fundamental catalysts:
  - Analyst upgrades / downgrades (broker ratings)
  - Earnings surprise (actual vs estimate)
  - M&A / merger announcements
  - Buybacks & dividend announcements
  - Promoter pledging changes
  - Screener.in ratios (ROE, PE, debt)
"""

import os
import requests
import pandas as pd
from loguru import logger
from typing import Optional
from datetime import datetime, timedelta

NSE_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com"}


class FundamentalAnalyzer:
    def __init__(self, session: requests.Session, config: dict):
        self.session = session
        self.cfg = config["signals"]["fundamental"]
        self.av_key = os.getenv("ALPHA_VANTAGE_KEY", "")

    # ------------------------------------------------------------------ #
    #  Corporate Announcements (NSE)
    # ------------------------------------------------------------------ #

    def get_corporate_announcements(self, symbol: str, days_back: int = 7) -> list:
        """Fetch recent corporate announcements for a symbol from NSE."""
        url = f"https://www.nseindia.com/api/top-corp-info?symbol={symbol}&market=equities"
        try:
            resp = self.session.get(url, headers=NSE_HEADERS, timeout=15)
            data = resp.json()
            announcements = []
            cutoff = datetime.now() - timedelta(days=days_back)

            for item in data.get("announcements", []):
                try:
                    ann_date = pd.to_datetime(item.get("an_dt", ""))
                    if ann_date < cutoff:
                        continue
                    announcements.append({
                        "date": ann_date.date(),
                        "subject": item.get("subject", ""),
                        "desc": item.get("desc", ""),
                        "attachment": item.get("attchmntFile", ""),
                    })
                except Exception:
                    pass
            return announcements
        except Exception as e:
            logger.warning(f"Corporate announcements fetch failed for {symbol}: {e}")
            return []

    def _keyword_score(self, text: str) -> int:
        """Quick keyword-based scoring for announcement subject."""
        text = text.lower()
        positive_keywords = [
            "buyback", "buy back", "dividend", "merger", "acquisition",
            "joint venture", "mou", "order", "contract", "award",
            "profit", "revenue growth", "expansion", "capex", "upgrade",
            "partnership", "collaboration", "listing",
        ]
        negative_keywords = [
            "pledge", "promoter sell", "downgrade", "loss", "fraud",
            "sebi notice", "investigation", "default", "resignation of md",
            "litigation", "penalty",
        ]
        pos = sum(1 for kw in positive_keywords if kw in text)
        neg = sum(1 for kw in negative_keywords if kw in text)
        return pos - neg

    # ------------------------------------------------------------------ #
    #  Earnings Surprise
    # ------------------------------------------------------------------ #

    def get_earnings_surprise(self, symbol: str) -> dict:
        """Check latest quarterly earnings vs estimates via Alpha Vantage."""
        if not self.av_key:
            return {}
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "EARNINGS",
                "symbol": f"NSE:{symbol}",
                "apikey": self.av_key,
            }
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()
            quarterly = data.get("quarterlyEarnings", [])
            if not quarterly:
                return {}

            latest = quarterly[0]
            reported = float(latest.get("reportedEPS", 0) or 0)
            estimated = float(latest.get("estimatedEPS", 0) or 0)
            surprise_pct = ((reported - estimated) / abs(estimated) * 100) if estimated != 0 else 0

            return {
                "reported_eps": reported,
                "estimated_eps": estimated,
                "surprise_pct": round(surprise_pct, 2),
                "beat": surprise_pct >= self.cfg.get("earnings_surprise_pct", 5),
                "fiscal_date": latest.get("fiscalDateEnding", ""),
            }
        except Exception as e:
            logger.warning(f"Earnings fetch failed for {symbol}: {e}")
            return {}

    # ------------------------------------------------------------------ #
    #  Broker Ratings (Screener.in scrape)
    # ------------------------------------------------------------------ #

    def get_screener_data(self, symbol: str) -> dict:
        """
        Fetch key financial ratios from Screener.in (public, no auth).
        Returns ROE, PE, D/E ratio, promoter holding %.
        """
        url = f"https://www.screener.in/company/{symbol}/consolidated/"
        try:
            resp = requests.get(url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/html",
            })
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "lxml")
            ratios = {}

            for li in soup.select("#top-ratios li"):
                name_el = li.select_one(".name")
                val_el  = li.select_one(".value")
                if name_el and val_el:
                    k = name_el.text.strip()
                    v = val_el.text.strip().replace(",", "").replace("%", "")
                    try:
                        ratios[k] = float(v)
                    except Exception:
                        ratios[k] = v

            return {
                "roe": ratios.get("Return on equity", ratios.get("ROE", 0)),
                "pe": ratios.get("Stock P/E", ratios.get("P/E", 0)),
                "debt_equity": ratios.get("Debt / Equity", 0),
                "promoter_holding": ratios.get("Promoter Holding", 0),
                "roce": ratios.get("ROCE", 0),
            }
        except Exception as e:
            logger.warning(f"Screener.in fetch failed for {symbol}: {e}")
            return {}

    # ------------------------------------------------------------------ #
    #  Pledging / Promoter Changes
    # ------------------------------------------------------------------ #

    def get_shareholding_pattern(self, symbol: str) -> dict:
        """Fetch shareholding pattern from NSE."""
        url = f"https://www.nseindia.com/api/shareHoldingPatterns?symbol={symbol}"
        try:
            resp = self.session.get(url, headers=NSE_HEADERS, timeout=15)
            data = resp.json()
            records = data.get("data", [])
            if not records:
                return {}

            latest = records[0]
            return {
                "promoter_pct": float(latest.get("promoter", 0)),
                "fii_pct": float(latest.get("fiis", 0)),
                "dii_pct": float(latest.get("diis", 0)),
                "public_pct": float(latest.get("public", 0)),
                "quarter": latest.get("quarter", ""),
            }
        except Exception as e:
            logger.warning(f"Shareholding fetch failed for {symbol}: {e}")
            return {}

    # ------------------------------------------------------------------ #
    #  Composite Score
    # ------------------------------------------------------------------ #

    def score(self, symbol: str) -> dict:
        """
        Fundamental score (0–100).

        Points:
          Positive announcements:     up to 5 pts
          Earnings beat:              5 pts
          ROE >= threshold:           4 pts
          PE reasonable:              3 pts
          D/E <= threshold:           3 pts
          Promoter holding high:      3 pts
          FII increasing:             2 pts
        """
        pts = 0
        details = {}

        # 1. Corporate announcements
        anns = self.get_corporate_announcements(symbol)
        ann_score = sum(self._keyword_score(a["subject"]) for a in anns)
        if ann_score > 0:
            pts += min(5, ann_score * 2)
            pos_anns = [a["subject"][:60] for a in anns if self._keyword_score(a["subject"]) > 0]
            if pos_anns:
                details["announcement"] = pos_anns[0] + " ✅"

        # 2. Earnings surprise
        earnings = self.get_earnings_surprise(symbol)
        if earnings.get("beat"):
            pts += 5
            details["earnings"] = f"EPS beat by {earnings['surprise_pct']:.1f}% ✅"

        # 3. Screener.in fundamentals
        ratios = self.get_screener_data(symbol)
        if ratios:
            roe = ratios.get("roe", 0)
            pe  = ratios.get("pe", 0)
            de  = ratios.get("debt_equity", 99)

            if roe >= self.cfg.get("min_roe", 12):
                pts += 4
                details["roe"] = f"ROE {roe:.1f}% ✅"

            if 0 < pe <= self.cfg.get("max_pe", 60):
                pts += 3
                details["pe"] = f"PE {pe:.1f} (reasonable) ✅"

            if de <= self.cfg.get("max_debt_equity", 2.0):
                pts += 3
                details["debt"] = f"D/E {de:.2f} (healthy) ✅"

        # 4. Shareholding
        sh = self.get_shareholding_pattern(symbol)
        if sh.get("promoter_pct", 0) >= 50:
            pts += 3
            details["promoter"] = f"Promoter holding {sh['promoter_pct']:.1f}% ✅"
        if sh.get("fii_pct", 0) >= 10:
            pts += 2
            details["fii_holding"] = f"FII holding {sh['fii_pct']:.1f}% ✅"

        score = min(100, int(pts / 25 * 100))
        return {"score": score, "raw_pts": pts, "ratios": ratios, "details": details}
