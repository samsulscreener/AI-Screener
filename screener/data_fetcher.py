"""
data_fetcher.py
---------------
Fetches OHLCV, market breadth, and stock universe data
from NSE, BSE, and Yahoo Finance.
"""

import time
import requests
import pandas as pd
import yfinance as yf
from loguru import logger
from typing import List, Optional
from functools import lru_cache
from datetime import datetime, timedelta
import pytz

IST = pytz.timezone("Asia/Kolkata")

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com",
}

# Pre-defined index constituents (symbols as per Yahoo Finance: SYMBOL.NS)
UNIVERSES = {
    "nifty50":  "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050",
    "nifty200": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20200",
    "nifty500": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500",
}


class DataFetcher:
    def __init__(self, config: dict):
        self.cfg = config["screening"]
        self.session = requests.Session()
        self._init_nse_session()

    # ------------------------------------------------------------------ #
    #  Session & Universe
    # ------------------------------------------------------------------ #

    def _init_nse_session(self):
        """NSE requires a cookie handshake before API calls."""
        try:
            self.session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
            logger.debug("NSE session initialized")
        except Exception as e:
            logger.warning(f"NSE session init failed: {e}. Some data may be unavailable.")

    def get_universe(self) -> List[str]:
        """Return list of NSE symbols based on configured universe."""
        universe = self.cfg.get("universe", "nifty500")
        custom = self.cfg.get("custom_symbols", [])

        if universe == "custom" and custom:
            logger.info(f"Using custom universe: {len(custom)} symbols")
            return [s.upper() for s in custom]

        url = UNIVERSES.get(universe, UNIVERSES["nifty500"])
        try:
            resp = self.session.get(url, headers=NSE_HEADERS, timeout=15)
            data = resp.json()
            symbols = [item["symbol"] for item in data.get("data", []) if item.get("symbol")]
            logger.info(f"Fetched {len(symbols)} symbols from {universe.upper()}")
            return symbols
        except Exception as e:
            logger.error(f"Failed to fetch universe from NSE: {e}")
            return self._fallback_nifty50()

    def _fallback_nifty50(self) -> List[str]:
        """Hardcoded Nifty 50 fallback."""
        return [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
            "LT", "BAJFINANCE", "HCLTECH", "ASIANPAINT", "AXISBANK",
            "MARUTI", "SUNPHARMA", "TITAN", "BAJAJFINSV", "WIPRO",
            "ULTRACEMCO", "NESTLEIND", "TECHM", "POWERGRID", "NTPC",
            "INDUSINDBK", "TATAMOTORS", "ONGC", "DRREDDY", "M&M",
        ]

    # ------------------------------------------------------------------ #
    #  Price & OHLCV Data
    # ------------------------------------------------------------------ #

    def get_ohlcv(self, symbol: str, period: str = "3mo", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Fetch OHLCV data via Yahoo Finance."""
        ticker = f"{symbol}.NS"
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if df.empty:
                logger.warning(f"No data for {symbol}")
                return None
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            logger.error(f"OHLCV fetch failed for {symbol}: {e}")
            return None

    def get_intraday_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch 5-min intraday OHLCV for today."""
        return self.get_ohlcv(symbol, period="5d", interval="5m")

    def get_quote(self, symbol: str) -> dict:
        """Fetch live NSE quote."""
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
        try:
            resp = self.session.get(url, headers=NSE_HEADERS, timeout=10)
            data = resp.json()
            pd_data = data.get("priceInfo", {})
            return {
                "symbol": symbol,
                "ltp": pd_data.get("lastPrice", 0),
                "open": pd_data.get("open", 0),
                "high": pd_data.get("intraDayHighLow", {}).get("max", 0),
                "low": pd_data.get("intraDayHighLow", {}).get("min", 0),
                "prev_close": pd_data.get("previousClose", 0),
                "change_pct": pd_data.get("pChange", 0),
                "volume": data.get("securityInfo", {}).get("tradedVolume", 0),
            }
        except Exception as e:
            logger.error(f"Quote fetch failed for {symbol}: {e}")
            return {}

    # ------------------------------------------------------------------ #
    #  Delivery & Market Breadth
    # ------------------------------------------------------------------ #

    def get_bhav_copy(self) -> Optional[pd.DataFrame]:
        """Download today's NSE Bhavcopy (EOD data with delivery %)."""
        today = datetime.now(IST)
        date_str = today.strftime("%d%b%Y").upper()
        url = f"https://www.nseindia.com/api/reports?archives=%5B%7B%22name%22%3A%22CM%20Bhavcopy%22%2C%22type%22%3A%22daily%22%2C%22category%22%3A%22capital-market%22%2C%22section%22%3A%22equities%22%7D%5D&date={today.strftime('%d-%m-%Y')}&type=equities&mode=single"
        try:
            resp = self.session.get(url, headers=NSE_HEADERS, timeout=20)
            df = pd.read_csv(pd.io.common.StringIO(resp.text))
            logger.info(f"Bhavcopy fetched: {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"Bhavcopy fetch failed: {e}")
            return None

    def get_delivery_data(self) -> Optional[pd.DataFrame]:
        """Get delivery position data from NSE."""
        url = "https://www.nseindia.com/api/deliveryposition"
        try:
            resp = self.session.get(url, headers=NSE_HEADERS, timeout=15)
            data = resp.json()
            df = pd.DataFrame(data.get("data", []))
            return df
        except Exception as e:
            logger.warning(f"Delivery data fetch failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Batch Fetching with Rate Limiting
    # ------------------------------------------------------------------ #

    def batch_ohlcv(self, symbols: List[str], period: str = "3mo") -> dict:
        """Fetch OHLCV for multiple symbols with rate limiting."""
        results = {}
        failed = []
        for i, sym in enumerate(symbols):
            df = self.get_ohlcv(sym, period=period)
            if df is not None and not df.empty:
                results[sym] = df
            else:
                failed.append(sym)
            if i > 0 and i % 20 == 0:
                time.sleep(1)  # Rate limiting courtesy
        logger.info(f"OHLCV batch: {len(results)} success, {len(failed)} failed")
        return results

    def get_market_status(self) -> bool:
        """Check if NSE market is currently open."""
        now = datetime.now(IST)
        if now.weekday() >= 5:
            return False
        market_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_open <= now <= market_close
