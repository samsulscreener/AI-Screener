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

    # ------------------------------------------------ #
    # Logging
    # ------------------------------------------------ #

    def _setup_logging(self):
        log_dir = self.config.get("logging", {}).get("log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)

        logger.add(f"{log_dir}/screener.log", rotation="1 day", retention="7 days")

    # ------------------------------------------------ #
    # DB
    # ------------------------------------------------ #

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

    # ------------------------------------------------ #
    # SAFE FILTER
    # ------------------------------------------------ #

    def _passes_filters(self, df):

        if df is None or df.empty or len(df) < 50:
            return False

        try:
            close = df["Close"].iloc[-1]
            close = float(close.item() if hasattr(close, "item") else close)

            avg_vol = df["Volume"].tail(20).mean()

            cfg = self.config["screening"]

            return (
                cfg["min_price"] <= close <= cfg["max_price"]
                and avg_vol >= cfg["min_volume"]
            )

        except:
            return False

    # ------------------------------------------------ #
    # CORE ANALYSIS
    # ------------------------------------------------ #

    def analyze_symbol(self, symbol, fii_dii, delivery_df, mode="all"):

        try:
            df = self.fetcher.get_ohlcv(symbol, period="6mo")

            if not self._passes_filters(df):
                return None

            close_val = df["Close"].iloc[-1]
            close_val = close_val.item() if hasattr(close_val, "item") else close_val
            ltp = float(close_val)

            # --- analyzers (safe) ---
            sm = self.smart_money.score(symbol, fii_dii) or {}
            vol = self.volume.score(symbol, df, delivery_df) or {}
            tech = self.tech.score(symbol, df) or {}
            news = self.news.score(symbol) or {}
            fund = self.fundamental.score(symbol) or {}

            # --- result ---
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

            # --- filters ---
            if mode == "btst" and setup not in ["BTST", "INTRADAY"]:
                return None

            if score < self.config["scoring"]["watch_threshold"]:
                return None

            tech_data = result.get("technical", {})
            if not isinstance(tech_data, dict) or tech_data.get("rsi") is None:
                return None

            logger.info(f"{symbol} | Score={score} | {result.get('signal')}")

            return result

        except Exception as e:
            logger.error(f"{symbol} failed: {e}")
            return None

    # ------------------------------------------------ #
    # RUN
    # ------------------------------------------------ #

    def run(self, mode="all", max_workers=5):

        logger.info(f"Starting screener: {mode}")
        start = time.time()

        fii_dii = self.smart_money.get_fii_dii_activity()
        delivery_df = self.fetcher.get_delivery_data()
        symbols = self.fetcher.get_universe()

        if not symbols:
            logger.error("No symbols")
            return pd.DataFrame()

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(self.analyze_symbol, s, fii_dii, delivery_df, mode) for s in symbols]

            for f in as_completed(futures):
                r = f.result()
                if r:
                    results.append(r)

        logger.info(f"Done in {round(time.time()-start,2)}s")

        if not results:
            logger.warning("No setups found")
            return pd.DataFrame()

        df = self.scorer.to_dataframe(results)
        self._save(df, results)

        return df

    # ------------------------------------------------ #
    # SAVE
    # ------------------------------------------------ #

    def _save(self, df, raw):

        if df.empty:
            return

        df.to_csv("data/results/latest.csv", index=False)

        conn = sqlite3.connect(self.db_path)

        for r in raw:
            try:
                conn.execute(
                    "INSERT INTO results VALUES (?,?,?,?)",
                    (
                        r.get("symbol"),
                        r.get("composite_score"),
                        r.get("signal"),
                        r.get("timestamp"),
                    ),
                )
            except:
                pass

        conn.commit()
        conn.close()
