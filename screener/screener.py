import os
import time
import yaml
import sqlite3
import pandas as pd
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
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

    def __init__(self, config_path: str = "config.yaml"):
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

    # ------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------ #

    def _setup_logging(self):
        log_cfg = self.config.get("logging", {})
        log_dir = log_cfg.get("log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)

        logger.add(
            f"{log_dir}/screener_{{time}}.log",
            rotation=log_cfg.get("rotate", "1 day"),
            retention=log_cfg.get("retention", "30 days"),
            level=log_cfg.get("level", "INFO"),
        )

    # ------------------------------------------------------------ #
    # DB
    # ------------------------------------------------------------ #

    def _init_db(self):
        db_path = self.config.get("output", {}).get("db_path", "data/screener.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path

        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT, ltp REAL, score INTEGER, signal TEXT,
                setup_type TEXT, rsi REAL, vol_spike REAL, delivery_pct REAL,
                target REAL, stop_loss REAL, rr_ratio REAL,
                sm_score INTEGER, vol_score INTEGER, tech_score INTEGER,
                news_score INTEGER, fund_score INTEGER,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------ #
    # Filters (SAFE)
    # ------------------------------------------------------------ #

    def _passes_filters(self, df: pd.DataFrame) -> bool:
        if df is None or df.empty or len(df) < 50:
            return False

        try:
            close = float(df["Close"].iloc[-1])
            avg_volume = df["Volume"].tail(20).mean()

            cfg = self.config["screening"]

            return (
                cfg["min_price"] <= close <= cfg["max_price"]
                and avg_volume >= cfg["min_volume"]
            )
        except:
            return False

    # ------------------------------------------------------------ #
    # Symbol analysis (SAFE + ADVANCED)
    # ------------------------------------------------------------ #

    def analyze_symbol(self, symbol, fii_dii, delivery_df, mode="all"):

        try:
            df = self.fetcher.get_ohlcv(symbol, period="6mo")

            if not self._passes_filters(df):
                return None

            ltp = float(df["Close"].iloc[-1])

            # Core analyzers
            sm = self.smart_money.score(symbol, fii_dii)
            vol = self.volume.score(symbol, df, delivery_df)
            tech = self.tech.score(symbol, df)
            news = self.news.score(symbol)
            fund = self.fundamental.score(symbol)

            result = self.scorer.build_result(
                symbol=symbol,
                ltp=ltp,
                sm_result=sm,
                vol_result=vol,
                tech_result=tech,
                news_result=news,
                fund_result=fund,
            )

            # ------------------------------------------------ #
            # Advanced filtering logic
            # ------------------------------------------------ #

            score = result.get("composite_score", 0)
            setup = result.get("setup_type", "")

            if mode == "btst" and setup not in ["BTST", "INTRADAY"]:
                return None

            if score < self.config["scoring"]["watch_threshold"]:
                return None

            # Extra quality filter (ADVANCED)
            if result["technical"].get("rsi") is None:
                return None

            logger.info(f"✅ {symbol} | Score={score} | {result['signal']}")

            return result

        except Exception as e:
            logger.error(f"{symbol} failed: {e}")
            return None

    # ------------------------------------------------------------ #
    # Main run
    # ------------------------------------------------------------ #

    def run(self, mode="all", max_workers=5):

        logger.info(f"🚀 Screener starting | Mode: {mode.upper()}")
        start = time.time()

        fii_dii = self.smart_money.get_fii_dii_activity()
        delivery_df = self.fetcher.get_delivery_data()
        global_news = self.news.get_global_market_sentiment()
        symbols = self.fetcher.get_universe()

        if not symbols:
            logger.error("No symbols fetched")
            return pd.DataFrame()

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

        logger.info(f"Completed in {round(time.time()-start,1)}s")

        if not results:
            logger.warning("No setups found")
            return pd.DataFrame()

        df = self.scorer.to_dataframe(results)

        self._save_results(df, results)

        return df

    # ------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------ #

    def _save_results(self, df, raw):

        if df.empty:
            return

        output_cfg = self.config.get("output", {})

        if output_cfg.get("save_to_csv", True):
            from datetime import datetime
            path = f"data/results/screen_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            os.makedirs("data/results", exist_ok=True)
            df.to_csv(path, index=False)
            logger.info(f"Saved CSV: {path}")

        if output_cfg.get("save_to_db", True):
            conn = sqlite3.connect(self.db_path)

            for r in raw:
                try:
                    ts = r.get("trade_setup", {})

                    conn.execute("""
                        INSERT INTO results VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        r["symbol"], r["ltp"], r["composite_score"],
                        r["signal"], r["setup_type"],
                        r["technical"].get("rsi"),
                        r["volume"].get("spike_ratio"),
                        r["volume"].get("delivery_pct"),
                        ts.get("target"), ts.get("stop_loss"),
                        ts.get("rr_ratio"),
                        r["scores"]["smart_money"],
                        r["scores"]["volume"],
                        r["scores"]["technical"],
                        r["scores"]["news"],
                        r["scores"]["fundamental"],
                        r["timestamp"],
                    ))

                except Exception as e:
                    logger.warning(f"DB insert failed: {e}")

            conn.commit()
            conn.close()
