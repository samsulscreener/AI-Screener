"""
smart_money.py  (v2)
--------------------
FIXES vs v1:
  * THE BIG ONE: FII/DII daily net flow is MARKET-WIDE. In v1 it contributed
    12 of a possible 22 points — over half the smart-money score — identically
    to every stock in the universe on a given day. That is zero ranking
    information and it floored every stock at ~27 composite on an FII-positive
    day. It now feeds get_market_regime() instead, which adjusts THRESHOLDS.
  * Score is a genuine 0-100 built only from STOCK-SPECIFIC evidence.
  * Cache bug fixed: v1 returned an empty DataFrame on line 86 BEFORE
    assigning self._bulk_cache, so a failing NSE endpoint was re-hit once
    per symbol for the entire run (500 wasted requests).
  * Adds quarter-on-quarter FII/DII *holding change for this stock*, which
    is genuinely stock-specific, unlike the daily market flow.
  * available=False when no smart-money data could be fetched, so the
    scorer renormalises instead of scoring a fake 0.
"""

import requests
import pandas as pd
from loguru import logger
from datetime import datetime, timedelta
import pytz

IST = pytz.timezone("Asia/Kolkata")

NSE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

_SENTINEL = object()


class SmartMoneyAnalyzer:

    def __init__(self, session: requests.Session, config: dict):
        self.session = session
        self.cfg = config.get("signals", {}).get("smart_money", {})
        self.regime_cfg = config.get("regime", {})

        # caches — note every path assigns before returning
        self._bulk_cache  = _SENTINEL
        self._block_cache = _SENTINEL
        self._fii_cache   = _SENTINEL
        self._holding_cache = {}

    # ------------------------------------------------------------------ #

    def _safe_json(self, url):
        try:
            r = self.session.get(url, headers=NSE_HEADERS, timeout=10)
            if r.status_code != 200:
                logger.debug(f"NSE {r.status_code} for {url}")
                return None
            return r.json()
        except Exception as e:
            logger.debug(f"NSE request failed {url}: {e}")
            return None

    @staticmethod
    def _num(x, default=0.0):
        try:
            return float(str(x).replace(",", "").replace("%", "").strip() or default)
        except Exception:
            return float(default)

    # ------------------------------------------------------------------ #
    #  MARKET-LEVEL  (regime gate, NOT part of the stock score)
    # ------------------------------------------------------------------ #

    def get_fii_dii_activity(self):
        if self._fii_cache is not _SENTINEL:
            return self._fii_cache

        data = self._safe_json("https://www.nseindia.com/api/fiidiiTradeReact")
        result = {"fii_net_cr": 0.0, "dii_net_cr": 0.0, "available": False}

        if data:
            rows = data if isinstance(data, list) else data.get("data", [])
            fii = dii = 0.0
            for r in rows[:4]:
                cat = str(r.get("category", "")).upper()
                val = self._num(r.get("netPurSales", r.get("netPurchasesSales", 0)))
                if "FII" in cat or "FPI" in cat:
                    fii += val
                elif "DII" in cat:
                    dii += val
            result = {"fii_net_cr": fii, "dii_net_cr": dii, "available": True}

        self._fii_cache = result          # cache even on failure
        return result

    def get_market_regime(self, vix=None, index_above_200dma=None):
        """
        Returns a regime dict used to ADJUST THRESHOLDS, never to score a stock.

        risk_on   -> normal or slightly easier bar
        neutral   -> normal bar
        risk_off  -> raise the bar; take fewer, better signals
        """
        fii = self.get_fii_dii_activity()
        cfg = self.regime_cfg

        if not cfg.get("enabled", True):
            return {"regime": "neutral", "threshold_delta": 0, "reasons": ["regime gate disabled"]}

        risk_off_flags, risk_on_flags = [], []

        net_flow = fii.get("fii_net_cr", 0) + fii.get("dii_net_cr", 0)
        if fii.get("available"):
            if net_flow < -1500:
                risk_off_flags.append(f"Institutional net selling Rs {net_flow:,.0f} Cr")
            elif net_flow > 1500:
                risk_on_flags.append(f"Institutional net buying Rs {net_flow:,.0f} Cr")

        if vix is not None and vix > 0:
            if vix > float(cfg.get("vix_caution", 20)):
                risk_off_flags.append(f"India VIX {vix:.1f} elevated")
            elif vix < float(cfg.get("vix_calm", 14)):
                risk_on_flags.append(f"India VIX {vix:.1f} calm")

        if index_above_200dma is False:
            risk_off_flags.append("Nifty below its 200-DMA")
        elif index_above_200dma is True:
            risk_on_flags.append("Nifty above its 200-DMA")

        if len(risk_off_flags) >= 2:
            regime, delta = "risk_off", int(cfg.get("threshold_penalty_risk_off", 10))
        elif risk_off_flags:
            regime, delta = "cautious", int(cfg.get("threshold_penalty_risk_off", 10)) // 2
        elif len(risk_on_flags) >= 2:
            regime, delta = "risk_on", -int(cfg.get("threshold_bonus_risk_on", 0))
        else:
            regime, delta = "neutral", 0

        return {
            "regime": regime,
            "threshold_delta": delta,
            "fii_net_cr": fii.get("fii_net_cr", 0),
            "dii_net_cr": fii.get("dii_net_cr", 0),
            "india_vix": vix,
            "reasons": risk_off_flags + risk_on_flags,
        }

    # ------------------------------------------------------------------ #
    #  STOCK-LEVEL evidence
    # ------------------------------------------------------------------ #

    def _deals_frame(self, kind):
        """kind: 'bulk' or 'block'. Cached, including on failure."""
        attr = f"_{kind}_cache"
        cached = getattr(self, attr)
        if cached is not _SENTINEL:
            return cached

        url = f"https://www.nseindia.com/api/{kind}-deals"
        data = self._safe_json(url)
        df = pd.DataFrame()

        if data:
            df = pd.DataFrame(data.get("data", []))
            if not df.empty:
                try:
                    df["_symbol"] = df.get("symbol", df.get("BD_SYMBOL", pd.Series(dtype=str)))
                    df["_qty"]    = pd.to_numeric(
                        df.get("qty", df.get("BD_QTY_TRD", 0)).astype(str).str.replace(",", ""),
                        errors="coerce").fillna(0)
                    df["_price"]  = pd.to_numeric(
                        df.get("watp", df.get("price", df.get("BD_TP_WATP", 0))).astype(str).str.replace(",", ""),
                        errors="coerce").fillna(0)
                    df["_side"]   = df.get("buySell", df.get("side", df.get("BD_BUY_SELL", ""))).astype(str)
                except Exception as e:
                    logger.debug(f"{kind} deals parse failed: {e}")
                    df = pd.DataFrame()

        setattr(self, attr, df)       # <-- cache BEFORE returning, always
        return df

    def get_deals_for_symbol(self, symbol):
        """Net bulk + block deal value in Rs Cr for this symbol today."""
        net = 0.0
        found = False
        for kind in ("bulk", "block"):
            df = self._deals_frame(kind)
            if df.empty or "_symbol" not in df.columns:
                continue
            sym_df = df[df["_symbol"].astype(str).str.strip().str.upper() == symbol.upper()]
            if sym_df.empty:
                continue
            found = True
            for _, r in sym_df.iterrows():
                val_cr = (r["_qty"] * r["_price"]) / 1e7
                side = str(r["_side"]).strip().upper()
                net += val_cr if side.startswith("B") else -val_cr
        return {"net_cr": round(net, 2), "found": found}

    def get_insider_trades(self, symbol):
        """Net promoter/insider acquisition over the configured window."""
        days = int(self.cfg.get("insider_window_days", 90))
        data = self._safe_json(f"https://www.nseindia.com/api/inside-trading?symbol={symbol}")
        if not data:
            return {"available": False, "net_value": 0.0}

        trades = data.get("data", [])
        if not trades:
            # No filings is NOT evidence of accumulation. Scoring this as a
            # neutral 50 would reintroduce exactly the market-wide-constant
            # problem this module was rewritten to remove.
            return {"available": False, "net_value": 0.0, "count": 0,
                    "reason": "no insider filings in window"}

        cutoff = datetime.now() - timedelta(days=days)
        buy = sell = 0.0
        count = 0
        for t in trades[:40]:
            try:
                d = pd.to_datetime(t.get("date", t.get("acqfromDt", "")), errors="coerce")
                if pd.notna(d) and d.to_pydatetime().replace(tzinfo=None) < cutoff:
                    continue
            except Exception:
                pass
            qty = self._num(t.get("secAcq", t.get("acquisitionDisposal", 0)))
            mode = str(t.get("tdpTransactionType", t.get("acqMode", ""))).lower()
            count += 1
            if "acquisition" in mode or "buy" in mode:
                buy += qty
            elif "disposal" in mode or "sell" in mode:
                sell += qty
        return {"available": True, "net_value": buy - sell, "count": count}

    def get_holding_change(self, symbol):
        """
        Quarter-on-quarter change in FII + DII holding for THIS stock.
        Unlike daily market flow, this is genuinely stock-specific.
        """
        if symbol in self._holding_cache:
            return self._holding_cache[symbol]

        data = self._safe_json(
            f"https://www.nseindia.com/api/shareHoldingPatterns?symbol={symbol}")
        result = {"available": False}

        if data:
            records = data.get("data", [])
            if len(records) >= 2:
                def inst(rec):
                    return self._num(rec.get("fiis", 0)) + self._num(rec.get("diis", 0))
                cur, prev = inst(records[0]), inst(records[1])
                result = {
                    "available": True,
                    "inst_pct": round(cur, 2),
                    "inst_change_pp": round(cur - prev, 2),
                    "promoter_pct": self._num(records[0].get("promoter", 0)),
                }
            elif len(records) == 1:
                result = {
                    "available": True,
                    "inst_pct": self._num(records[0].get("fiis", 0)) + self._num(records[0].get("diis", 0)),
                    "inst_change_pp": 0.0,
                    "promoter_pct": self._num(records[0].get("promoter", 0)),
                }

        self._holding_cache[symbol] = result
        return result

    # ------------------------------------------------------------------ #
    #  Score  (genuine 0-100, stock-specific only)
    # ------------------------------------------------------------------ #
    #  Bulk/block net buying   45
    #  Insider net buying      30
    #  Institutional holding   25
    #                         ----
    #                         100
    #  Components with no data are dropped from BOTH numerator and
    #  denominator. If nothing is available at all -> available=False.
    # ------------------------------------------------------------------ #

    def score(self, symbol, fii_dii=None):
        pts = 0.0
        budget = 0.0
        details = {}

        # --- 1. Bulk / block deals: 45 pts ---
        deals = self.get_deals_for_symbol(symbol)
        min_cr = float(self.cfg.get("bulk_deal_min_value_cr", 5))
        if deals["found"]:
            budget += 45
            net = deals["net_cr"]
            if net >= min_cr * 3:
                pts += 45
            elif net >= min_cr:
                pts += 32
            elif net > 0:
                pts += 15
            elif net <= -min_cr:
                pts += 0          # net institutional selling
            else:
                pts += 8
            details["bulk_block_net_cr"] = net

        # --- 2. Insider activity: 30 pts (only when filings actually exist) ---
        insider = self.get_insider_trades(symbol)
        if insider.get("available") and insider.get("count", 0) > 0:
            budget += 30
            nv = insider.get("net_value", 0.0)
            if nv > 0:
                pts += 30
                details["insider"] = "net buying"
            elif nv < 0:
                pts += 0
                details["insider"] = "net selling"
            else:
                pts += 15
                details["insider"] = "offsetting trades"

        # --- 3. Institutional holding change: 25 pts ---
        hold = self.get_holding_change(symbol)
        if hold.get("available"):
            budget += 25
            chg = hold.get("inst_change_pp", 0.0)
            if chg >= 1.0:
                pts += 25
            elif chg > 0:
                pts += 18
            elif chg == 0:
                pts += 10
            details["inst_holding_pct"] = hold.get("inst_pct")
            details["inst_change_pp"] = chg

        if budget == 0:
            return {"score": 0.0, "available": False,
                    "reason": "no stock-specific smart-money data", "details": {}}

        return {
            "score": round(pts / budget * 100.0, 1),
            "available": True,
            "components_used": int(budget),
            "details": details,
        }
