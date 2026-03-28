"""
options_analyzer.py
-------------------
Advanced options flow analysis for institutional footprints:
  - Put/Call Ratio (PCR) per stock and index
  - Open Interest buildup / unwinding detection
  - Max Pain calculation
  - Unusual options activity (large OI jumps)
  - India VIX (fear gauge)
"""

import requests
import pandas as pd
from loguru import logger
from typing import Optional

NSE_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com"}


class OptionsAnalyzer:
    def __init__(self, session: requests.Session, config: dict):
        self.session = session
        self.cfg = config["signals"].get("volume", {})

    # ------------------------------------------------------------------ #
    #  India VIX
    # ------------------------------------------------------------------ #

    def get_india_vix(self) -> dict:
        """
        Fetch India VIX (market fear gauge).
        VIX < 13  → Low fear, bullish bias
        VIX 13–18 → Normal range
        VIX > 20  → High fear / uncertainty
        """
        url = "https://www.nseindia.com/api/allIndices"
        try:
            resp = self.session.get(url, headers=NSE_HEADERS, timeout=10)
            data = resp.json()
            for idx in data.get("data", []):
                if "VIX" in idx.get("index", "").upper():
                    vix = float(idx.get("last", 0))
                    return {
                        "vix": vix,
                        "change_pct": float(idx.get("percentChange", 0)),
                        "bullish_environment": vix < 16,
                        "high_fear": vix > 20,
                    }
        except Exception as e:
            logger.warning(f"VIX fetch failed: {e}")
        return {"vix": 0, "bullish_environment": True, "high_fear": False}

    # ------------------------------------------------------------------ #
    #  Options Chain
    # ------------------------------------------------------------------ #

    def get_options_chain(self, symbol: str) -> Optional[dict]:
        """Fetch full options chain from NSE."""
        is_index = symbol in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
        if is_index:
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
        try:
            resp = self.session.get(url, headers=NSE_HEADERS, timeout=15)
            return resp.json()
        except Exception as e:
            logger.warning(f"Options chain fetch failed for {symbol}: {e}")
            return None

    def parse_options_chain(self, data: dict) -> pd.DataFrame:
        """Parse raw options chain JSON into a clean DataFrame."""
        records = data.get("records", {}).get("data", [])
        rows = []
        for rec in records:
            strike = rec.get("strikePrice", 0)
            expiry = rec.get("expiryDate", "")
            ce = rec.get("CE", {}) or {}
            pe = rec.get("PE", {}) or {}
            rows.append({
                "strike": strike,
                "expiry": expiry,
                "ce_ltp":          ce.get("lastPrice", 0),
                "ce_oi":           ce.get("openInterest", 0),
                "ce_chng_oi":      ce.get("changeinOpenInterest", 0),
                "ce_volume":       ce.get("totalTradedVolume", 0),
                "ce_iv":           ce.get("impliedVolatility", 0),
                "pe_ltp":          pe.get("lastPrice", 0),
                "pe_oi":           pe.get("openInterest", 0),
                "pe_chng_oi":      pe.get("changeinOpenInterest", 0),
                "pe_volume":       pe.get("totalTradedVolume", 0),
                "pe_iv":           pe.get("impliedVolatility", 0),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  PCR, Max Pain, OI Analysis
    # ------------------------------------------------------------------ #

    def compute_pcr(self, chain_df: pd.DataFrame) -> dict:
        """Compute Put/Call ratio by OI and by volume."""
        total_ce_oi = chain_df["ce_oi"].sum()
        total_pe_oi = chain_df["pe_oi"].sum()
        total_ce_vol = chain_df["ce_volume"].sum()
        total_pe_vol = chain_df["pe_volume"].sum()

        pcr_oi  = round(total_pe_oi / total_ce_oi, 3)  if total_ce_oi  > 0 else 1.0
        pcr_vol = round(total_pe_vol / total_ce_vol, 3) if total_ce_vol > 0 else 1.0

        # PCR interpretation
        # < 0.7  → Extreme call writing → Bearish (market expected to stay below)
        # 0.7–1.3 → Balanced / Neutral
        # > 1.3  → Extreme put writing → Bullish (supports are being bought)
        if pcr_oi > 1.3:
            sentiment = "BULLISH"
        elif pcr_oi < 0.7:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"

        return {
            "pcr_oi": pcr_oi,
            "pcr_vol": pcr_vol,
            "total_ce_oi": int(total_ce_oi),
            "total_pe_oi": int(total_pe_oi),
            "sentiment": sentiment,
        }

    def compute_max_pain(self, chain_df: pd.DataFrame) -> float:
        """
        Max Pain = strike where total option buyers lose the most money at expiry.
        Usually acts as a gravitational pull on the underlying near expiry.
        """
        strikes = chain_df["strike"].unique()
        min_pain_val = float("inf")
        max_pain_strike = 0.0

        for s in strikes:
            # Loss for CE holders if price settles at s
            ce_loss = ((chain_df["strike"] - s).clip(lower=0) * chain_df["ce_oi"]).sum()
            # Loss for PE holders if price settles at s
            pe_loss = ((s - chain_df["strike"]).clip(lower=0) * chain_df["pe_oi"]).sum()
            total = ce_loss + pe_loss

            if total < min_pain_val:
                min_pain_val = total
                max_pain_strike = s

        return float(max_pain_strike)

    def detect_unusual_oi(self, chain_df: pd.DataFrame, top_n: int = 5) -> dict:
        """
        Identify strikes with unusually high OI change (institutional positioning).
        Returns top call and put strikes by OI buildup.
        """
        ce_buildup = chain_df[chain_df["ce_chng_oi"] > 0].nlargest(top_n, "ce_chng_oi")
        pe_buildup = chain_df[chain_df["pe_chng_oi"] > 0].nlargest(top_n, "pe_chng_oi")

        # Strong resistance = highest CE OI strike (call writers defending)
        ce_resistance = chain_df.nlargest(1, "ce_oi")["strike"].values[0] if not chain_df.empty else 0
        pe_support     = chain_df.nlargest(1, "pe_oi")["strike"].values[0] if not chain_df.empty else 0

        return {
            "resistance_strike": ce_resistance,
            "support_strike":    pe_support,
            "top_ce_buildup":    ce_buildup[["strike", "ce_oi", "ce_chng_oi"]].to_dict("records"),
            "top_pe_buildup":    pe_buildup[["strike", "pe_oi", "pe_chng_oi"]].to_dict("records"),
        }

    # ------------------------------------------------------------------ #
    #  Full Analysis for a Symbol
    # ------------------------------------------------------------------ #

    def analyze(self, symbol: str) -> dict:
        """
        Full options analysis returning:
          - PCR (OI and Volume)
          - Max Pain
          - Support/Resistance strikes
          - Unusual OI activity
          - India VIX context
        """
        data = self.get_options_chain(symbol)
        if not data:
            return {}

        chain_df = self.parse_options_chain(data)
        if chain_df.empty:
            return {}

        # Use nearest expiry only for accuracy
        if "expiry" in chain_df.columns and not chain_df["expiry"].empty:
            nearest_expiry = chain_df["expiry"].iloc[0]
            chain_df = chain_df[chain_df["expiry"] == nearest_expiry]

        pcr       = self.compute_pcr(chain_df)
        max_pain  = self.compute_max_pain(chain_df)
        unusual   = self.detect_unusual_oi(chain_df)
        vix       = self.get_india_vix()

        result = {
            **pcr,
            "max_pain":          max_pain,
            "resistance_strike": unusual["resistance_strike"],
            "support_strike":    unusual["support_strike"],
            "top_ce_buildup":    unusual["top_ce_buildup"],
            "top_pe_buildup":    unusual["top_pe_buildup"],
            "india_vix":         vix.get("vix"),
            "vix_bullish_env":   vix.get("bullish_environment"),
        }
        logger.debug(f"Options [{symbol}]: PCR={pcr['pcr_oi']} MaxPain={max_pain} VIX={vix.get('vix')}")
        return result

    def score_from_options(self, options: dict, ltp: float) -> dict:
        """
        Derive a score component (0–100) from options data.
        Used optionally inside the main scorer.
        """
        if not options:
            return {"score": 50, "details": {}}  # Neutral if no data

        pts = 0
        details = {}

        # PCR bullish zone (0.8 – 1.5)
        pcr = options.get("pcr_oi", 1.0)
        if 0.8 <= pcr <= 1.5:
            pts += 5
            details["pcr"] = f"PCR {pcr} — supportive ✅"
        elif pcr > 1.5:
            pts += 3
            details["pcr"] = f"PCR {pcr} — put heavy (bounce zone)"
        elif pcr < 0.6:
            details["pcr"] = f"PCR {pcr} — call heavy (caution) ⚠️"

        # LTP vs support/resistance
        support    = options.get("support_strike", 0)
        resistance = options.get("resistance_strike", 0)
        if support and ltp > 0:
            if ltp >= support * 0.98:
                pts += 5
                details["support"] = f"LTP ₹{ltp} near support ₹{support} ✅"
        if resistance and ltp > 0:
            gap_pct = (resistance - ltp) / ltp * 100
            if gap_pct > 3:
                pts += 3
                details["resistance"] = f"Resistance ₹{resistance} ({gap_pct:.1f}% away) ✅"

        # VIX bullish environment
        if options.get("vix_bullish_env"):
            pts += 5
            details["vix"] = f"India VIX {options.get('india_vix')} (calm market) ✅"
        elif options.get("india_vix", 0) > 22:
            details["vix"] = f"India VIX {options.get('india_vix')} (high fear) ⚠️"

        # Max pain proximity (within 1%)
        max_pain = options.get("max_pain", 0)
        if max_pain and ltp > 0:
            if abs(ltp - max_pain) / ltp < 0.01:
                pts += 2
                details["max_pain"] = f"Near max pain ₹{max_pain}"

        score = min(100, int(pts / 20 * 100))
        return {"score": score, "raw_pts": pts, "details": details}
