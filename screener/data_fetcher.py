"""
data_fetcher.py  (v2)
---------------------
FIXES vs v1:
  * Universe no longer depends on NSE's equity-stockIndices API, which
    rejects datacentre IPs and most un-warmed sessions. v1 therefore fell
    through to _fallback_nifty50 — 30 megacaps — silently, and from GitHub
    Actions it did so essentially always. v2 downloads the official
    constituent CSV from nsearchives, caches it to disk for N days, and
    LOGS LOUDLY when the universe is smaller than expected.
  * MultiIndex columns from yfinance are flattened at the source. This is
    the root cause of the LTP=0 bug in screener.py:127.
  * Thread-local requests.Session. v1 shared one Session (and its mutating
    NSE cookie jar) across every ThreadPoolExecutor worker.
  * Batch OHLCV via a single yfinance call instead of 500 sequential ones.
  * Liquidity filters from config are actually computed here.
"""

import os
import io
import time
import threading
from datetime import datetime, timedelta
from typing import List, Optional

import requests
import pandas as pd
import yfinance as yf
from loguru import logger
import pytz

IST = pytz.timezone("Asia/Kolkata")

NSE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

# Official constituent lists. These are static files, not the API, and they
# are far more reliable from a server.
UNIVERSE_CSV = {
    "nifty50":     "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "nifty200":    "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv",
    "nifty500":    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "smallcap250": "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
    "midcap150":   "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
}
EXPECTED_SIZE = {"nifty50": 50, "nifty200": 200, "nifty500": 500,
                 "smallcap250": 250, "midcap150": 150}

CACHE_DIR = "data/universe"


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse yfinance MultiIndex columns down to plain OHLCV names."""
    if df is not None and isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        lvl0 = set(df.columns.get_level_values(0))
        # yfinance orders as (field, ticker) or (ticker, field) depending on version
        if {"Close", "Open", "High", "Low", "Volume"} & lvl0:
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(-1)
    return df


class DataFetcher:

    def __init__(self, config: dict):
        self.config = config
        self.cfg = config.get("screening", {})
        self._local = threading.local()
        os.makedirs(CACHE_DIR, exist_ok=True)
        # warm one session up front so the cookie jar exists
        self._init_session(self.session)

    # ------------------------------------------------------------------ #
    #  Thread-local sessions
    # ------------------------------------------------------------------ #

    @property
    def session(self) -> requests.Session:
        """One Session per thread. NSE mutates cookies on every call and a
        shared Session across a ThreadPoolExecutor is not safe."""
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update(NSE_HEADERS)
            self._init_session(s)
            self._local.session = s
        return s

    @staticmethod
    def _init_session(s: requests.Session):
        """NSE needs a cookie handshake before any API call will answer."""
        try:
            s.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
            s.get("https://www.nseindia.com/market-data/live-equity-market",
                  headers=NSE_HEADERS, timeout=10)
        except Exception as e:
            logger.debug(f"NSE session warm-up failed: {e}")

    # ------------------------------------------------------------------ #
    #  Universe
    # ------------------------------------------------------------------ #

    def get_universe(self) -> List[str]:
        universe = self.cfg.get("universe", "nifty500")
        custom = self.cfg.get("custom_symbols", [])
        exclude = {s.upper() for s in self.cfg.get("exclude_symbols", [])}

        if universe == "custom" and custom:
            syms = [s.upper() for s in custom]
            logger.info(f"Custom universe: {len(syms)} symbols")
            return [s for s in syms if s not in exclude]

        symbols = self._universe_from_csv(universe)

        if not symbols:
            symbols = self._universe_from_api(universe)

        if not symbols:
            logger.error(
                f"UNIVERSE FETCH FAILED for {universe}. Falling back to 30 hardcoded "
                f"megacaps. Your screen is now covering 6% of the intended universe — "
                f"treat today's results as unrepresentative."
            )
            symbols = self._fallback_nifty50()

        expected = EXPECTED_SIZE.get(universe)
        if expected and len(symbols) < expected * 0.8:
            logger.warning(
                f"Universe {universe} returned {len(symbols)} symbols, expected ~{expected}. "
                f"Results are not comparable to a full run."
            )

        symbols = [s for s in symbols if s not in exclude]
        logger.info(f"Universe {universe}: {len(symbols)} symbols")
        return symbols

    def _universe_from_csv(self, universe) -> List[str]:
        url = UNIVERSE_CSV.get(universe)
        if not url:
            return []

        cache_path = os.path.join(CACHE_DIR, f"{universe}.csv")
        ttl_days = int(self.cfg.get("universe_cache_days", 7))

        if os.path.exists(cache_path):
            age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
            if age < timedelta(days=ttl_days):
                try:
                    df = pd.read_csv(cache_path)
                    syms = df["Symbol"].dropna().astype(str).str.strip().str.upper().tolist()
                    if syms:
                        logger.debug(f"Universe {universe} from cache ({age.days}d old)")
                        return syms
                except Exception:
                    pass

        try:
            resp = self.session.get(url, headers=NSE_HEADERS, timeout=20)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            syms = df["Symbol"].dropna().astype(str).str.strip().str.upper().tolist()
            if syms:
                df.to_csv(cache_path, index=False)
                logger.info(f"Universe {universe} downloaded: {len(syms)} symbols (cached)")
                return syms
        except Exception as e:
            logger.warning(f"Constituent CSV fetch failed for {universe}: {e}")

        # stale cache beats no cache
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path)
                syms = df["Symbol"].dropna().astype(str).str.strip().str.upper().tolist()
                logger.warning(f"Using STALE cached universe for {universe} ({len(syms)} symbols)")
                return syms
            except Exception:
                pass
        return []

    def _universe_from_api(self, universe) -> List[str]:
        names = {"nifty50": "NIFTY%2050", "nifty200": "NIFTY%20200",
                 "nifty500": "NIFTY%20500", "smallcap250": "NIFTY%20SMALLCAP%20250",
                 "midcap150": "NIFTY%20MIDCAP%20150"}
        idx = names.get(universe)
        if not idx:
            return []
        try:
            r = self.session.get(
                f"https://www.nseindia.com/api/equity-stockIndices?index={idx}",
                headers=NSE_HEADERS, timeout=15)
            if r.status_code != 200:
                return []
            data = r.json()
            return [i["symbol"].upper() for i in data.get("data", []) if i.get("symbol")]
        except Exception as e:
            logger.debug(f"Index API fallback failed: {e}")
            return []

    @staticmethod
    def _fallback_nifty50() -> List[str]:
        return [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
            "LT", "BAJFINANCE", "HCLTECH", "ASIANPAINT", "AXISBANK",
            "MARUTI", "SUNPHARMA", "TITAN", "BAJAJFINSV", "WIPRO",
            "ULTRACEMCO", "NESTLEIND", "TECHM", "POWERGRID", "NTPC",
            "INDUSINDBK", "TATAMOTORS", "ONGC", "DRREDDY", "M&M",
        ]

    # ------------------------------------------------------------------ #
    #  OHLCV
    # ------------------------------------------------------------------ #

    @staticmethod
    def to_yahoo(symbol: str) -> str:
        return f"{symbol.replace('&', '%26') if False else symbol}.NS"

    def get_ohlcv(self, symbol: str, period="1y", interval="1d") -> Optional[pd.DataFrame]:
        try:
            df = yf.download(
                self.to_yahoo(symbol), period=period, interval=interval,
                progress=False, auto_adjust=True, threads=False,
            )
            if df is None or df.empty:
                return None
            df = flatten_columns(df)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            logger.debug(f"OHLCV failed for {symbol}: {e}")
            return None

    def batch_ohlcv(self, symbols: List[str], period="1y", interval="1d",
                    chunk=60) -> dict:
        """
        One yfinance call per chunk instead of one per symbol.
        For 500 names this is ~9 requests instead of 500.
        """
        out = {}
        for i in range(0, len(symbols), chunk):
            batch = symbols[i:i + chunk]
            tickers = [self.to_yahoo(s) for s in batch]
            try:
                raw = yf.download(
                    tickers, period=period, interval=interval,
                    progress=False, auto_adjust=True, group_by="ticker",
                    threads=True,
                )
            except Exception as e:
                logger.warning(f"Batch OHLCV failed for chunk {i//chunk}: {e}")
                continue

            for sym, tkr in zip(batch, tickers):
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        if tkr not in raw.columns.get_level_values(0):
                            continue
                        sub = raw[tkr].dropna(how="all")
                    else:
                        sub = raw.dropna(how="all")
                    if not sub.empty:
                        out[sym] = sub
                except Exception:
                    continue
            time.sleep(0.4)

        logger.info(f"OHLCV batch: {len(out)}/{len(symbols)} symbols retrieved")
        return out

    # ------------------------------------------------------------------ #
    #  Liquidity filter  (config keys that v1 never read)
    # ------------------------------------------------------------------ #

    def passes_filters(self, symbol: str, df: pd.DataFrame):
        """Returns (ok: bool, reason: str). Applies min/max price, min volume
        and min turnover from config — none of which v1 ever used."""
        try:
            df = flatten_columns(df)
            close = pd.to_numeric(df["Close"], errors="coerce").dropna()
            vol = pd.to_numeric(df["Volume"], errors="coerce").dropna()
            if close.empty or len(vol) < 20:
                return False, "insufficient history"

            ltp = float(close.iloc[-1])
            avg_vol = float(vol.tail(20).mean())
            turnover_cr = ltp * avg_vol / 1e7

            if ltp < float(self.cfg.get("min_price", 0)):
                return False, f"price {ltp:.1f} below min"
            if ltp > float(self.cfg.get("max_price", 1e9)):
                return False, f"price {ltp:.1f} above max"
            if avg_vol < float(self.cfg.get("min_volume", 0)):
                return False, f"avg vol {avg_vol:,.0f} below min"
            if turnover_cr < float(self.cfg.get("min_turnover_cr", 0)):
                return False, f"turnover Rs {turnover_cr:.2f}Cr below min"
            return True, ""
        except Exception as e:
            return False, f"filter error: {e}"

    # ------------------------------------------------------------------ #
    #  Market context
    # ------------------------------------------------------------------ #

    def get_delivery_data(self) -> Optional[pd.DataFrame]:
        """
        NSE's securitywise delivery data. Endpoint shape changes often, so
        this returns None rather than an empty frame on failure — the volume
        analyzer then renormalises instead of scoring a fake 0.
        """
        for url in (
            "https://www.nseindia.com/api/snapshot-capital-market-ordersbook?type=deliverable",
            "https://www.nseindia.com/api/deliveryposition",
        ):
            try:
                r = self.session.get(url, headers=NSE_HEADERS, timeout=15)
                if r.status_code != 200:
                    continue
                data = r.json()
                rows = data.get("data", data if isinstance(data, list) else [])
                df = pd.DataFrame(rows)
                if not df.empty:
                    logger.info(f"Delivery data: {len(df)} rows")
                    return df
            except Exception:
                continue
        logger.warning("Delivery data unavailable — volume component will renormalise")
        return None

    def get_index_context(self) -> dict:
        """Nifty trend + India VIX, used only by the regime gate."""
        ctx = {"index_above_200dma": None, "india_vix": None}
        try:
            idx = yf.download(self.config.get("regime", {}).get("index_symbol", "^NSEI"),
                              period="2y", interval="1d", progress=False,
                              auto_adjust=True, threads=False)
            idx = flatten_columns(idx)
            if idx is not None and len(idx) > 200:
                c = pd.to_numeric(idx["Close"], errors="coerce").dropna()
                ctx["index_above_200dma"] = bool(c.iloc[-1] > c.rolling(200).mean().iloc[-1])
                ctx["index_ltp"] = round(float(c.iloc[-1]), 2)
        except Exception as e:
            logger.debug(f"Index context failed: {e}")

        try:
            r = self.session.get("https://www.nseindia.com/api/allIndices",
                                 headers=NSE_HEADERS, timeout=10)
            if r.status_code == 200:
                for i in r.json().get("data", []):
                    if "VIX" in str(i.get("index", "")).upper():
                        ctx["india_vix"] = float(i.get("last", 0))
                        break
        except Exception as e:
            logger.debug(f"VIX fetch failed: {e}")

        return ctx

    def get_market_status(self) -> bool:
        now = datetime.now(IST)
        if now.weekday() >= 5:
            return False
        o = now.replace(hour=9, minute=15, second=0, microsecond=0)
        c = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return o <= now <= c
