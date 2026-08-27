"""
screener.py  (v2)
-----------------
FIXES vs v1:
  * LTP BUG (v1 line 127). `self._safe_float(df["Close"].values)` on a
    yfinance MultiIndex frame passed a 2-D array to float(x[-1]), which
    raised TypeError, was swallowed by a bare except, and returned 0.0 —
    so scorer.py hit `if ltp <= 0: return {}` and EVERY entry/target/stop/RR
    came back empty. Columns are now flattened at the fetcher and the LTP is
    read off a guaranteed 1-D Series.
  * Batch OHLCV download (~9 requests for 500 names instead of 500).
  * Market-level data (FII/DII, VIX, index trend) fetched ONCE and turned
    into a regime gate, instead of being scored into every stock identically.
  * EVERY scored symbol is persisted with its full component breakdown and
    a run_id. v1 dropped everything under score 20 at storage time, which
    destroyed the negative class you need to ever backtest this.
  * Raw result dicts are kept on self.last_results so alerts can actually
    be sent (v1's main.py had no way to get them).
  * Liquidity/price filters from config are applied.
  * options_analyzer is wired in (v1 had 264 lines of it, never imported).
"""

import os
import time
import uuid
import json
import yaml
import sqlite3
import pandas as pd
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv
import pytz

from .data_fetcher import DataFetcher, flatten_columns
from .smart_money import SmartMoneyAnalyzer
from .volume_analyzer import VolumeAnalyzer
from .technical_analyzer import TechnicalAnalyzer
from .news_analyzer import NewsAnalyzer
from .fundamental_analyzer import FundamentalAnalyzer
from .options_analyzer import OptionsAnalyzer
from .scorer import Scorer

load_dotenv()
IST = pytz.timezone("Asia/Kolkata")


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
        self.options = OptionsAnalyzer(session, self.config)

        self.last_results = []      # raw dicts, for alerts + research engine
        self.run_id = None
        self.regime = {}

    # ---------------- LOGGING ---------------- #

    def _setup_logging(self):
        log_cfg = self.config.get("logging", {})
        log_dir = log_cfg.get("log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)
        logger.add(f"{log_dir}/screener.log",
                   rotation=log_cfg.get("rotate", "1 day"),
                   retention=log_cfg.get("retention", "30 days"),
                   level=log_cfg.get("level", "INFO"))

    # ---------------- DB ---------------- #

    def _init_db(self):
        path = self.config.get("output", {}).get("db_path", "data/screener.db")
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.db_path = path

        conn = sqlite3.connect(path)
        # The signals table IS your backtest dataset. One row per symbol per
        # run, including the ones that scored badly.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                run_ts TEXT,
                mode TEXT,
                symbol TEXT,
                ltp REAL,
                composite_score REAL,
                percentile REAL,
                signal TEXT,
                setup_type TEXT,
                components_live INTEGER,
                score_smart_money REAL,
                score_volume REAL,
                score_technical REAL,
                score_news REAL,
                score_fundamental REAL,
                rsi REAL,
                atr_pct REAL,
                vol_spike REAL,
                delivery_pct REAL,
                entry REAL,
                target REAL,
                stop_loss REAL,
                rr REAL,
                regime TEXT,
                detail_json TEXT,
                ret_1d REAL, ret_5d REAL, ret_20d REAL, ret_60d REAL,
                nifty_1d REAL, nifty_5d REAL, nifty_20d REAL, nifty_60d REAL,
                labelled_at TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS ix_signals_run ON signals(run_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_signals_sym ON signals(symbol, run_ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_signals_lbl ON signals(labelled_at)")
        conn.commit()
        conn.close()

    # ---------------- HELPERS ---------------- #

    @staticmethod
    def _last_close(df):
        """
        THE v1 BUG FIX. Always returns a scalar float LTP, never 0.0-by-accident.
        """
        try:
            df = flatten_columns(df)
            s = df["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s = pd.to_numeric(s, errors="coerce").dropna()
            return float(s.iloc[-1]) if len(s) else 0.0
        except Exception as e:
            logger.warning(f"LTP extraction failed: {e}")
            return 0.0

    def _safe_call(self, fn, *args, **kwargs):
        try:
            out = fn(*args, **kwargs)
            return out if isinstance(out, dict) else {"score": 0.0, "available": False}
        except Exception as e:
            logger.debug(f"Analyzer {getattr(fn, '__qualname__', fn)} failed: {e}")
            return {"score": 0.0, "available": False, "reason": str(e)}

    # ---------------- ANALYZE ---------------- #

    def analyze_symbol(self, symbol, ohlcv, delivery_df, mode="all"):
        try:
            if ohlcv is None or len(ohlcv) < 30:
                return None

            ok, reason = self.fetcher.passes_filters(symbol, ohlcv)
            if not ok:
                logger.debug(f"{symbol} filtered out: {reason}")
                return None

            ltp = self._last_close(ohlcv)
            if ltp <= 0:
                logger.warning(f"{symbol}: could not read LTP, skipping")
                return None

            intraday = mode == "intraday" or self.fetcher.get_market_status()

            tech = self._safe_call(self.tech.score, symbol, ohlcv)
            vol  = self._safe_call(self.volume.score, symbol, ohlcv, delivery_df,
                                   intraday=intraday)
            sm   = self._safe_call(self.smart_money.score, symbol)
            news = self._safe_call(self.news.score, symbol)
            fund = self._safe_call(self.fundamental.score, symbol)

            extra = {}
            if self.config.get("signals", {}).get("options", {}).get("enabled"):
                opt = self._safe_call(self.options.analyze, symbol)
                if opt:
                    extra["options"] = opt

            result = self.scorer.build_result(
                symbol=symbol, ltp=ltp,
                sm_result=sm, vol_result=vol, tech_result=tech,
                news_result=news, fund_result=fund,
                regime=self.regime, extra=extra,
            )
            logger.debug(f"{symbol} -> {result['composite_score']} "
                         f"({result['components_live']} live) {result['signal']}")
            return result

        except Exception as e:
            logger.error(f"{symbol} failed: {e}")
            return None

    # ---------------- RUN ---------------- #

    def run(self, mode="all", max_workers=6):
        self.run_id = datetime.now(IST).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        logger.info(f"=== Screener run {self.run_id} | mode={mode} ===")
        start = time.time()

        # ---- market-level data: fetched ONCE, used as a regime gate ----
        idx_ctx = self.fetcher.get_index_context()
        self.regime = self.smart_money.get_market_regime(
            vix=idx_ctx.get("india_vix"),
            index_above_200dma=idx_ctx.get("index_above_200dma"),
        )
        logger.info(f"Regime: {self.regime['regime'].upper()} "
                    f"(threshold {self.regime['threshold_delta']:+d}) "
                    f"{'; '.join(self.regime.get('reasons', [])) or 'no strong signal'}")

        delivery_df = self.fetcher.get_delivery_data()
        symbols = self.fetcher.get_universe()
        if not symbols:
            logger.error("Empty universe — aborting run")
            return pd.DataFrame()

        # ---- bulk price download ----
        period = "3mo" if mode == "intraday" else "2y"
        logger.info(f"Downloading {period} OHLCV for {len(symbols)} symbols...")
        ohlcv_map = self.fetcher.batch_ohlcv(symbols, period=period)

        # ---- per-symbol analysis ----
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(self.analyze_symbol, s, ohlcv_map.get(s), delivery_df, mode): s
                for s in symbols if s in ohlcv_map
            }
            for f in as_completed(futures):
                r = f.result()
                if r:
                    results.append(r)

        if not results:
            logger.warning("No symbols scored")
            return pd.DataFrame()

        # ---- cross-sectional ranking across today's universe ----
        results = self.scorer.rank_results(results)
        results.sort(key=lambda r: r["composite_score"], reverse=True)
        self.last_results = results

        # ---- persist EVERYTHING (this is the backtest dataset) ----
        if self.config.get("output", {}).get("log_all_candidates", True):
            self._save_signals(results, mode)

        actionable = [r for r in results if r["signal"] in ("BUY", "WATCH")]
        rejected = [r for r in results if r["signal"] == "REJECTED"]

        logger.info(f"Scored {len(results)} | actionable {len(actionable)} | "
                    f"hard-rejected {len(rejected)} | {round(time.time()-start,1)}s")

        df = self.scorer.to_dataframe(results)
        self._save_csv(df)
        return df

    # ---------------- SAVE ---------------- #

    def _save_signals(self, results, mode):
        ts = datetime.now(IST).isoformat()
        rows = []
        for r in results:
            t = r.get("technical", {}) or {}
            v = r.get("volume", {}) or {}
            ts_ = r.get("trade_setup", {}) or {}
            s = r.get("scores", {})
            rows.append((
                self.run_id, ts, mode, r.get("symbol"), r.get("ltp"),
                r.get("composite_score"), r.get("percentile"),
                r.get("signal"), r.get("setup_type"), r.get("components_live"),
                s.get("smart_money"), s.get("volume"), s.get("technical"),
                s.get("news"), s.get("fundamental"),
                t.get("rsi"), t.get("atr_pct"),
                v.get("spike_ratio"), v.get("delivery_pct"),
                ts_.get("entry"), ts_.get("target_2"), ts_.get("stop_loss"), ts_.get("rr_ratio"),
                r.get("regime"),
                json.dumps({k: val for k, val in r.items()
                            if k not in ("technical", "volume", "smart_money",
                                         "news", "fundamental")}, default=str),
                None, None, None, None, None, None, None, None, None,
            ))
        try:
            conn = sqlite3.connect(self.db_path, timeout=20)
            conn.executemany(
                "INSERT INTO signals ("
                "run_id, run_ts, mode, symbol, ltp, composite_score, percentile, "
                "signal, setup_type, components_live, score_smart_money, score_volume, "
                "score_technical, score_news, score_fundamental, rsi, atr_pct, vol_spike, "
                "delivery_pct, entry, target, stop_loss, rr, regime, detail_json, "
                "ret_1d, ret_5d, ret_20d, ret_60d, nifty_1d, nifty_5d, nifty_20d, "
                "nifty_60d, labelled_at"
                ") VALUES (" + ",".join(["?"] * 34) + ")", rows)
            conn.commit()
            conn.close()
            logger.info(f"Persisted {len(rows)} signal rows (run {self.run_id})")
        except Exception as e:
            logger.error(f"Signal persistence failed: {e}")

    def _save_csv(self, df):
        if df is None or df.empty:
            return
        out_dir = self.config.get("output", {}).get("csv_dir", "data/results")
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, "latest.csv"), index=False)
        df.to_csv(os.path.join(out_dir, f"{self.run_id}.csv"), index=False)
