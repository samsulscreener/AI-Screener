import os
import time
import yaml
import sqlite3
import pandas as pd
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from .data_fetcher import DataFetcher
from .smart_money import SmartMoneyAnalyzer
from .volume_analyzer import VolumeAnalyzer
from .technical_analyzer import TechnicalAnalyzer
from .news_analyzer import NewsAnalyzer
from .fundamental_analyzer import FundamentalAnalyzer
from .scorer import Scorer

load_dotenv()


class IndiaStockScreener:

    def __init__(self, config_path="config.yaml"):

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self._setup_logging()
        self._init_db()

        self.fetcher = DataFetcher(self.config)
        self.tech = TechnicalAnalyzer(self.config)
        self.news = NewsAnalyzer(self.config)
        self.scorer = Scorer(self.config)

        session = self.fetcher.session
        self.smart_money = SmartMoneyAnalyzer(session, self.config)
        self.volume = VolumeAnalyzer(session, self.config)
        self.fundamental = FundamentalAnalyzer(session, self.config)

    # ---------------- LOGGING ---------------- #

    def _setup_logging(self):
        log_dir = self.config.get("logging", {}).get("log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)
        logger.add(f"{log_dir}/screener.log", rotation="1 day", retention="7 days")

    # ---------------- DB ---------------- #

    def _init_db(self):
        path = self.config.get("output", {}).get("db_path", "data/screener.db")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db_path = path

        conn = sqlite3.connect(path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                symbol TEXT, score REAL, signal TEXT, timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()

    # ---------------- SAFE FLOAT ---------------- #

    def _safe_float(self, x, default=0.0):
        try:
            if isinstance(x, pd.Series):
                if x.empty:
                    return default
                val = x.iloc[-1]
                if isinstance(val, pd.Series):
                    val = val.values[-1]
                return float(val)

            if hasattr(x, "__len__") and not isinstance(x, (str, bytes)):
                return float(x[-1])

            return float(x)

        except:
            return default

    # ---------------- SAFE CALL ---------------- #

    def _safe_call(self, fn, *args):
        try:
            out = fn(*args)
            return out if isinstance(out, dict) else {}
        except Exception as e:
            logger.error(f"Analyzer error: {e}")
            return {}

    # ---------------- CLEAN DICT ---------------- #

    def _clean_dict(self, d):
        clean = {}
        if not isinstance(d, dict):
            return clean

        for k, v in d.items():
            try:
                if isinstance(v, pd.Series):
                    clean[k] = self._safe_float(v)
                else:
                    clean[k] = v
            except:
                clean[k] = 0

        return clean

    # ---------------- FILTER ---------------- #

    def _passes_filters(self, df):

    if df is None or df.empty or len(df) < 30:
        return False

    try:
        close = self._safe_float(df["Close"])

        # 🔥 Relaxed filtering (critical)
        return close > 10   # only basic sanity

    except:
        return False

    # ---------------- ANALYZE ---------------- #

    def analyze_symbol(self, symbol, fii_dii, delivery_df, mode="all"):

        try:
            df = self.fetcher.get_ohlcv(symbol, period="6mo")

            if not self._passes_filters(df):
                return None

            ltp = self._safe_float(df["Close"].values)

            # analyzers
            sm = self._clean_dict(self._safe_call(self.smart_money.score, symbol, fii_dii))
            vol = self._clean_dict(self._safe_call(self.volume.score, symbol, df, delivery_df))
            tech = self._clean_dict(self._safe_call(self.tech.score, symbol, df))
            news = self._clean_dict(self._safe_call(self.news.score, symbol))
            fund = self._clean_dict(self._safe_call(self.fundamental.score, symbol))

            result = self.scorer.build_result(
                symbol=symbol,
                ltp=ltp,
                sm_result=sm,
                vol_result=vol,
                tech_result=tech,
                news_result=news,
                fund_result=fund,
            )

            if not result:
                return None

            score = result.get("composite_score", 0)
            setup = result.get("setup_type", "")

            # ✅ DEBUG LOG (correct placement)
            logger.info(f"{symbol} raw score: {score}")

            if mode == "btst" and setup not in ["BTST", "INTRADAY"]:
                return None

            # 🔥 relaxed filter
            if score < 40:
                return None

            logger.info(f"✅ {symbol} | Score={score} | {result.get('signal')}")

            return result

        except Exception as e:
            logger.error(f"{symbol} failed: {e}")
            return None

    # ---------------- RUN ---------------- #

    def run(self, mode="all", max_workers=2):

        logger.info(f"Starting screener: {mode}")
        start = time.time()

        fii_dii = self.smart_money.get_fii_dii_activity()
        delivery_df = self.fetcher.get_delivery_data()

        symbols = self.fetcher.get_universe()

        if not symbols:
            logger.warning("Universe fetch failed, using fallback list")
            symbols = [
                "RELIANCE", "TCS", "INFY", "HDFCBANK",
                "ICICIBANK", "SBIN", "AXISBANK",
                "ITC", "LT", "BHARTIARTL",
                "KOTAKBANK", "HCLTECH", "WIPRO",
                "ASIANPAINT", "MARUTI"
            ]

        logger.info(f"Running for {len(symbols)} symbols")

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.analyze_symbol, s, fii_dii, delivery_df, mode): s
                for s in symbols
            }

            for f in as_completed(futures):
                r = f.result()
                if r:
                    results.append(r)

        logger.info(f"Done in {round(time.time() - start, 2)}s")

        if not results:
            logger.warning("No setups found")
            return pd.DataFrame()

        df = self.scorer.to_dataframe(results)
        self._save(df, results)

        return df

    # ---------------- SAVE ---------------- #

    def _save(self, df, raw):

        if df.empty:
            return

        os.makedirs("data/results", exist_ok=True)
        df.to_csv("data/results/latest.csv", index=False)

        try:
            conn = sqlite3.connect(self.db_path)

            for r in raw:
                conn.execute(
                    "INSERT INTO results VALUES (?,?,?,?)",
                    (
                        r.get("symbol"),
                        r.get("composite_score"),
                        r.get("signal"),
                        r.get("timestamp"),
                    ),
                )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.warning(f"DB save failed: {e}")
