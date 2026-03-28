"""
screener.py
-----------
Main orchestrator — runs all signal analyzers in parallel
and produces ranked results.
"""

import os
import time
import yaml
import sqlite3
import pandas as pd
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
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

        self.fetcher     = DataFetcher(self.config)
        self.tech        = TechnicalAnalyzer(self.config)
        self.news        = NewsAnalyzer(self.config)
        self.scorer      = Scorer(self.config)

        # Analyzers that need the NSE session
        session = self.fetcher.session
        self.smart_money = SmartMoneyAnalyzer(session, self.config)
        self.volume      = VolumeAnalyzer(session, self.config)
        self.fundamental = FundamentalAnalyzer(session, self.config)

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

    def _init_db(self):
        """Initialize SQLite database for results storage."""
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

    # ------------------------------------------------------------------ #
    #  Pre-Screening Filters
    # ------------------------------------------------------------------ #

    def _passes_filters(self, symbol: str, df: pd.DataFrame) -> bool:
        """Quick pre-filter before running expensive analyzers."""
        cfg = self.config["screening"]
        if df is None or len(df) < 20:
            return False

        close = df["Close"].iloc[-1]
        volume = df["Volume"].iloc[-1]
        avg_volume = df["Volume"].tail(20).mean()

        if not (cfg["min_price"] <= close <= cfg["max_price"]):
            return False
        if avg_volume < cfg["min_volume"]:
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Single Symbol Analysis
    # ------------------------------------------------------------------ #

    def analyze_symbol(
        self,
        symbol: str,
        fii_dii: dict,
        delivery_df: Optional[pd.DataFrame],
        mode: str = "all",
    ) -> Optional[dict]:
        """Run all analyzers on a single symbol. Returns result dict or None."""
        try:
            df = self.fetcher.get_ohlcv(symbol, period="6mo")
            if not self._passes_filters(symbol, df):
                return None

            ltp = df["Close"].iloc[-1]

            # Run analyzers
            sm_result   = self.smart_money.score(symbol, fii_dii)
            vol_result  = self.volume.score(symbol, df, delivery_df)
            tech_result = self.tech.score(symbol, df)
            news_result = self.news.score(symbol)
            fund_result = self.fundamental.score(symbol)

            result = self.scorer.build_result(
                symbol=symbol,
                ltp=float(ltp),
                sm_result=sm_result,
                vol_result=vol_result,
                tech_result=tech_result,
                news_result=news_result,
                fund_result=fund_result,
            )

            # Mode-based filtering
            if mode == "intraday" and result["setup_type"] not in ["INTRADAY", "STRONG BUY"]:
                return None
            elif mode == "btst" and result["setup_type"] not in ["BTST", "INTRADAY"]:
                return None
            elif mode == "swing" and result["setup_type"] not in ["SWING", "BTST"]:
                return None

            score = result["composite_score"]
            if score >= self.config["scoring"]["watch_threshold"]:
                logger.info(f"✅ {symbol}: Score={score} | {result['signal']} | {result['setup_type']}")
                return result

            return None

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Full Screen Run
    # ------------------------------------------------------------------ #

    def run(self, mode: str = "all", max_workers: int = 5) -> pd.DataFrame:
        """
        Run the full screener. Returns a sorted DataFrame of results.
        
        Args:
            mode: 'intraday' | 'btst' | 'swing' | 'all'
            max_workers: Parallel threads for symbol analysis
        """
        logger.info(f"🚀 Screener starting | Mode: {mode.upper()} | Universe: {self.config['screening']['universe']}")
        start = time.time()

        # Pre-fetch global data (single fetch, shared across all symbols)
        fii_dii      = self.smart_money.get_fii_dii_activity()
        delivery_df  = self.fetcher.get_delivery_data()
        global_news  = self.news.get_global_market_sentiment()
        symbols      = self.fetcher.get_universe()

        logger.info(f"Universe: {len(symbols)} symbols | FII: {'🟢' if fii_dii.get('fii_positive') else '🔴'} {fii_dii.get('fii_net_cr', 0)}Cr")
        logger.info(f"Global sentiment: {'Risk-ON 🟢' if global_news.get('risk_on') else 'Risk-OFF 🔴'}")

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.analyze_symbol, sym, fii_dii, delivery_df, mode): sym
                for sym in symbols
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        elapsed = round(time.time() - start, 1)
        logger.info(f"✅ Screener done in {elapsed}s | {len(results)} setups found")

        if not results:
            logger.warning("No qualifying setups found.")
            return pd.DataFrame()

        df_results = self.scorer.to_dataframe(results)

        # Save
        self._save_results(df_results, results)

        return df_results

    # ------------------------------------------------------------------ #
    #  Save Results
    # ------------------------------------------------------------------ #

    def _save_results(self, df: pd.DataFrame, raw: list):
        output_cfg = self.config.get("output", {})

        if output_cfg.get("save_to_csv", True):
            from datetime import datetime
            csv_dir = output_cfg.get("csv_dir", "data/results")
            os.makedirs(csv_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            path = f"{csv_dir}/screen_{ts}.csv"
            df.to_csv(path, index=False)
            logger.info(f"Results saved: {path}")

        if output_cfg.get("save_to_db", True):
            conn = sqlite3.connect(self.db_path)
            for r in raw:
                ts = r.get("trade_setup", {})
                conn.execute("""
                    INSERT INTO results (
                        symbol, ltp, score, signal, setup_type, rsi,
                        vol_spike, delivery_pct, target, stop_loss, rr_ratio,
                        sm_score, vol_score, tech_score, news_score, fund_score, timestamp
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    r["symbol"], r["ltp"], r["composite_score"], r["signal"], r["setup_type"],
                    r["technical"].get("rsi"), r["volume"].get("spike_ratio"),
                    r["volume"].get("delivery_pct"),
                    ts.get("target"), ts.get("stop_loss"), ts.get("rr_ratio"),
                    r["scores"]["smart_money"], r["scores"]["volume"],
                    r["scores"]["technical"], r["scores"]["news"], r["scores"]["fundamental"],
                    r["timestamp"],
                ))
            conn.commit()
            conn.close()
