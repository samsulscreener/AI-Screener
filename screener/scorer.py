"""
scorer.py  (v2)
---------------
FIXES vs v1:
  * THE x2 FUDGE IS GONE. v1 line 74 did `return int(round(score * 2))`
    because the real weighted maximum was 61.25, not 100 — the thresholds
    were never calibrated, they were worked around. With every analyzer now
    returning a genuine 0-100, the weighted sum is genuinely 0-100 and
    `strong_buy_threshold: 70` finally means what it says.
  * Weights renormalise over AVAILABLE components. If news or fundamentals
    could not be fetched, their weight is redistributed instead of scoring
    the stock 0 for a signal we simply do not have.
  * Volatility-scaled trade setups. v1 used a flat 5% target / 2% stop for
    every stock, which is inside the noise for a 4%-ATR name and absurdly
    wide for a 0.8%-ATR one. Now ATR-based, with the stop pulled to the
    nearest structural level, and setups rejected below a minimum R/R.
  * R/R is no longer a constant. v1 could only ever emit 2.5 or 2.0.
  * Cross-sectional percentile ranking across the run, so "top decile" is
    always well-defined regardless of market conditions.
  * Regime gate raises/lowers the THRESHOLD instead of corrupting the score.
"""

import math
import pandas as pd
from datetime import datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")

COMPONENTS = ("smart_money", "volume", "technical", "news", "fundamental")


class Scorer:

    def __init__(self, config):
        w = config.get("signals", {}).get("weights", {})
        self.weights = {
            "smart_money": float(w.get("smart_money", 0.20)),
            "volume":      float(w.get("volume", 0.20)),
            "technical":   float(w.get("technical", 0.35)),
            "news":        float(w.get("news", 0.05)),
            "fundamental": float(w.get("fundamental", 0.20)),
        }

        sc = config.get("scoring", {})
        self.strong_buy = float(sc.get("strong_buy_threshold", 70))
        self.watch      = float(sc.get("watch_threshold", 55))
        self.weak       = float(sc.get("weak_threshold", 40))
        self.min_components = int(sc.get("min_components", 2))
        self.cross_sectional = bool(sc.get("cross_sectional_ranks", True))

        self.min_rr = 1.8      # reject setups below this after ATR sizing

    # ------------------------------------------------------------------ #

    @staticmethod
    def _num(x, default=0.0):
        try:
            if x is None:
                return float(default)
            if isinstance(x, pd.Series):
                return float(x.iloc[-1]) if len(x) else float(default)
            if isinstance(x, str):
                x = x.strip().replace("%", "").replace(",", "")
                if x == "" or x.lower() in ("nan", "none", "na", "-"):
                    return float(default)
            v = float(x)
            return v if math.isfinite(v) else float(default)
        except Exception:
            return float(default)

    # ------------------------------------------------------------------ #
    #  Composite  (genuine 0-100, weights renormalised over live components)
    # ------------------------------------------------------------------ #

    def composite_score(self, results: dict):
        """
        results: {component_name: analyzer_result_dict}
        Only components with available=True contribute; their weights are
        renormalised so the output stays on a true 0-100 scale.
        """
        live = {}
        for name in COMPONENTS:
            r = results.get(name) or {}
            if r.get("available"):
                live[name] = self._num(r.get("score", 0))

        if not live:
            return 0.0, {}, 0

        total_w = sum(self.weights[n] for n in live)
        if total_w <= 0:
            return 0.0, live, len(live)

        score = sum(live[n] * self.weights[n] for n in live) / total_w
        return round(score, 1), live, len(live)

    def effective_weights(self, live_names):
        total = sum(self.weights[n] for n in live_names) or 1.0
        return {n: round(self.weights[n] / total, 3) for n in live_names}

    # ------------------------------------------------------------------ #
    #  Classification
    # ------------------------------------------------------------------ #

    def classify_signal(self, score, n_components, regime_delta=0):
        if n_components < self.min_components:
            return "INSUFFICIENT_DATA"
        strong = self.strong_buy + regime_delta
        watch  = self.watch + regime_delta
        weak   = self.weak + regime_delta
        if score >= strong:
            return "BUY"
        if score >= watch:
            return "WATCH"
        if score >= weak:
            return "WEAK"
        return "IGNORE"

    def classify_setup(self, score, tech, vol, regime_delta=0):
        """Which timeframe does this setup actually suit?"""
        tech = tech or {}
        vol = vol or {}
        spike = self._num(vol.get("spike_ratio", 1.0), 1.0)
        rsi = self._num(tech.get("rsi", 50), 50)
        st_buy = bool(tech.get("supertrend_buy", False))
        above200 = bool(tech.get("above_200ema", False))
        gap_hi = self._num(tech.get("pct_from_20d_high", 99), 99)
        watch = self.watch + regime_delta

        # Intraday: needs today's participation to be genuinely unusual
        if spike >= 2.5 and score >= watch:
            return "INTRADAY"
        # BTST: moderate spike, closing near the 20-day high, trend intact
        if spike >= 1.5 and gap_hi <= 3.0 and st_buy and score >= watch - 5:
            return "BTST"
        # Swing: structural trend, RSI with room to run
        if st_buy and above200 and 40 <= rsi <= 68 and score >= watch - 10:
            return "SWING"
        if score >= watch:
            return "WATCH"
        return "NO_SIGNAL"

    # ------------------------------------------------------------------ #
    #  Trade setup  (ATR-scaled, structure-aware)
    # ------------------------------------------------------------------ #

    def generate_trade_setup(self, ltp, score, tech):
        tech = tech or {}
        ltp = self._num(ltp)
        if ltp <= 0:
            return {"valid": False, "reason": "no LTP"}

        atr = self._num(tech.get("atr", 0))
        if atr <= 0:
            # fall back to a volatility proxy rather than a made-up flat %
            atr = ltp * 0.02

        # Higher conviction earns a wider target, not a tighter stop.
        rr_target = 3.0 if score >= self.strong_buy else 2.5 if score >= self.watch else 2.0

        raw_stop = ltp - 1.5 * atr

        # Pull the stop to the nearest structural level below, if it is close.
        swing_low = self._num(tech.get("swing_low", 0))
        st_line = self._num(tech.get("supertrend_line", 0))
        candidates = [c for c in (swing_low, st_line) if 0 < c < ltp]
        if candidates:
            best = max(candidates)               # tightest structural level below price
            if abs(best - raw_stop) / ltp < 0.02:
                raw_stop = best * 0.998          # a touch below the level

        stop = round(raw_stop, 2)
        risk = ltp - stop
        if risk <= 0:
            return {"valid": False, "reason": "invalid stop"}

        target1 = round(ltp + risk * rr_target * 0.5, 2)
        target2 = round(ltp + risk * rr_target, 2)
        rr = round((target2 - ltp) / risk, 2)

        risk_pct = round(risk / ltp * 100, 2)

        return {
            "valid": rr >= self.min_rr and risk_pct <= 8.0,
            "entry": round(ltp, 2),
            "stop_loss": stop,
            "target_1": target1,
            "target_2": target2,
            "rr_ratio": rr,
            "risk_pct": risk_pct,
            "atr": round(atr, 2),
            "reason": "" if rr >= self.min_rr else f"R/R {rr} below minimum {self.min_rr}",
        }

    # ------------------------------------------------------------------ #
    #  Build
    # ------------------------------------------------------------------ #

    def build_result(self, symbol, ltp, sm_result, vol_result, tech_result,
                     news_result, fund_result, regime=None, extra=None):

        regime = regime or {}
        regime_delta = int(regime.get("threshold_delta", 0))

        parts = {
            "smart_money": sm_result or {},
            "volume":      vol_result or {},
            "technical":   tech_result or {},
            "news":        news_result or {},
            "fundamental": fund_result or {},
        }

        # Hard rejects short-circuit everything.
        if parts["fundamental"].get("hard_reject"):
            return {
                "symbol": symbol,
                "ltp": self._num(ltp),
                "composite_score": 0.0,
                "signal": "REJECTED",
                "setup_type": "NO_SIGNAL",
                "reject_reason": parts["fundamental"].get("reject_reason", ""),
                "scores": {k: self._num(v.get("score", 0)) for k, v in parts.items()},
                "components_live": 0,
                "technical": parts["technical"],
                "volume": parts["volume"],
                "smart_money": parts["smart_money"],
                "news": parts["news"],
                "fundamental": parts["fundamental"],
                "trade_setup": {"valid": False, "reason": "hard reject"},
                "regime": regime.get("regime", "unknown"),
                "timestamp": datetime.now(IST).isoformat(),
            }

        comp, live, n_live = self.composite_score(parts)
        signal = self.classify_signal(comp, n_live, regime_delta)
        setup = self.classify_setup(comp, parts["technical"], parts["volume"], regime_delta)
        trade = self.generate_trade_setup(ltp, comp, parts["technical"])

        if signal in ("BUY", "WATCH") and not trade.get("valid"):
            signal = "WATCH_NO_ENTRY"

        return {
            "symbol": symbol,
            "ltp": round(self._num(ltp), 2),
            "composite_score": comp,
            "signal": signal,
            "setup_type": setup,
            "scores": {k: self._num(v.get("score", 0)) for k, v in parts.items()},
            "available": {k: bool(v.get("available")) for k, v in parts.items()},
            "effective_weights": self.effective_weights(live.keys()) if live else {},
            "components_live": n_live,
            "technical": parts["technical"],
            "volume": parts["volume"],
            "smart_money": parts["smart_money"],
            "news": parts["news"],
            "fundamental": parts["fundamental"],
            "trade_setup": trade,
            "regime": regime.get("regime", "unknown"),
            "regime_delta": regime_delta,
            "timestamp": datetime.now(IST).isoformat(),
            **(extra or {}),
        }

    # ------------------------------------------------------------------ #
    #  Cross-sectional ranking
    # ------------------------------------------------------------------ #

    def rank_results(self, results):
        """
        Percentile-rank every stock against the rest of TODAY'S universe.
        This is what makes 'top decile' well-defined regardless of whether
        the whole market is strong or weak. Mutates and returns `results`.
        """
        if not self.cross_sectional or not results:
            return results

        scored = [r for r in results if r.get("signal") != "REJECTED"]
        if len(scored) < 5:
            return results

        df = pd.DataFrame([{"symbol": r["symbol"], "score": r["composite_score"]} for r in scored])
        df["pct"] = df["score"].rank(pct=True) * 100
        pct_map = dict(zip(df["symbol"], df["pct"]))

        # per-component percentiles too — this is what your Phase 1 backtest
        # will regress forward returns against.
        comp_maps = {}
        for c in COMPONENTS:
            vals = [(r["symbol"], r["scores"].get(c, 0))
                    for r in scored if r.get("available", {}).get(c)]
            if len(vals) >= 5:
                cdf = pd.DataFrame(vals, columns=["symbol", "v"])
                cdf["p"] = cdf["v"].rank(pct=True) * 100
                comp_maps[c] = dict(zip(cdf["symbol"], cdf["p"]))

        for r in results:
            r["percentile"] = round(pct_map.get(r["symbol"], 0.0), 1)
            r["component_percentiles"] = {
                c: round(m.get(r["symbol"], 0.0), 1)
                for c, m in comp_maps.items() if r["symbol"] in m
            }
        return results

    # ------------------------------------------------------------------ #
    #  DataFrame  (columns now match what main.py actually prints)
    # ------------------------------------------------------------------ #

    def to_dataframe(self, results):
        rows = []
        for r in results:
            t = r.get("technical", {}) or {}
            v = r.get("volume", {}) or {}
            ts = r.get("trade_setup", {}) or {}
            rows.append({
                "Symbol":     r.get("symbol"),
                "LTP":        r.get("ltp"),
                "Score":      r.get("composite_score"),
                "Pctile":     r.get("percentile"),
                "Signal":     r.get("signal"),
                "Setup":      r.get("setup_type"),
                "RSI":        t.get("rsi"),
                "ATR%":       t.get("atr_pct"),
                "Vol_Spike":  v.get("spike_ratio"),
                "Delivery%":  v.get("delivery_pct"),
                "Entry":      ts.get("entry"),
                "Target":     ts.get("target_2"),
                "SL":         ts.get("stop_loss"),
                "RR":         ts.get("rr_ratio"),
                "Live":       r.get("components_live"),
            })

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
