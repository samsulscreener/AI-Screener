"""
volume_analyzer.py
------------------
Detects unusual volume activity, delivery % spikes,
and open interest buildup — key smart-money footprints.
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Optional
import requests

NSE_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com"}


class VolumeAnalyzer:
    def __init__(self, session: requests.Session, config: dict):
        self.session = session
        self.cfg = config["signals"]["volume"]

    # ------------------------------------------------------------------ #
    #  Volume Spike Detection
    # ------------------------------------------------------------------ #

    def analyze_volume(self, symbol: str, df: pd.DataFrame) -> dict:
        """
        Analyze OHLCV dataframe for volume anomalies.

        Signals:
          - Volume spike vs 20-day average
          - Price+Volume expansion (institutional accumulation)
          - Volume contraction on pullbacks (holding pattern)
          - Accumulation/Distribution
        """
        if df is None or len(df) < 22:
            return {"score": 0, "details": {}}

        vol = df["Volume"].values
        close = df["Close"].values

        vol_20avg = vol[-21:-1].mean()
        vol_today = vol[-1]
        spike_ratio = vol_today / vol_20avg if vol_20avg > 0 else 1.0
        spike_thresh = self.cfg["spike_multiplier"]

        # Price-Volume relationship
        price_up = close[-1] > close[-2]
        vol_expanding = vol_today > vol_20avg

        # Volume trend (5-day vs 20-day)
        vol_5avg = vol[-6:-1].mean()
        vol_trend_up = vol_5avg > vol_20avg

        # A/D (Accumulation / Distribution) indicator
        ad = self._calc_ad(df)
        ad_trend = (ad[-1] - ad[-6]) > 0 if len(ad) >= 6 else False

        # Volume on up-days vs down-days (last 10)
        up_days_vol = df[df["Close"] > df["Open"]]["Volume"].tail(10).mean()
        dn_days_vol = df[df["Close"] <= df["Open"]]["Volume"].tail(10).mean()
        vol_quality = (up_days_vol / dn_days_vol) if dn_days_vol > 0 else 1.0

        pts = 0
        details = {}

        # 1. Volume spike
        if spike_ratio >= spike_thresh:
            pts += 10
            details["spike"] = f"Volume {spike_ratio:.1f}x 20-day avg 🔴"
        elif spike_ratio >= 1.5:
            pts += 5
            details["spike"] = f"Volume {spike_ratio:.1f}x avg (moderate)"

        # 2. Price-volume confirmation
        if price_up and vol_expanding:
            pts += 5
            details["pv"] = "Price up + expanding volume ✅"
        elif not price_up and not vol_expanding:
            pts += 3
            details["pv"] = "Pullback on low volume (healthy)"

        # 3. Volume trend
        if vol_trend_up:
            pts += 3
            details["trend"] = "5d > 20d volume trend ✅"

        # 4. A/D indicator
        if ad_trend:
            pts += 4
            details["ad"] = "Accumulation/Distribution positive ✅"

        # 5. Vol quality (up vs down day volume)
        if vol_quality >= 1.5:
            pts += 3
            details["quality"] = f"Up-day vol/Down-day vol ratio: {vol_quality:.1f} ✅"

        score = min(100, int(pts / 25 * 100))
        return {
            "score": score,
            "raw_pts": pts,
            "spike_ratio": round(spike_ratio, 2),
            "vol_today": int(vol_today),
            "vol_20avg": int(vol_20avg),
            "details": details,
        }

    def _calc_ad(self, df: pd.DataFrame) -> np.ndarray:
        """Accumulation/Distribution Line."""
        high = df["High"].values
        low = df["Low"].values
        close = df["Close"].values
        volume = df["Volume"].values
        mfm = ((close - low) - (high - close)) / (high - low + 1e-9)
        mfv = mfm * volume
        return np.cumsum(mfv)

    # ------------------------------------------------------------------ #
    #  Delivery %
    # ------------------------------------------------------------------ #

    def get_delivery_pct(self, symbol: str, delivery_df: Optional[pd.DataFrame]) -> float:
        """
        Get delivery % for the symbol from bhavcopy/delivery data.
        Returns value between 0-100.
        """
        if delivery_df is None or delivery_df.empty:
            return self._fetch_delivery_from_nse(symbol)

        # Try to find column for symbol
        sym_col = None
        for col in ["SYMBOL", "Symbol", "symbol"]:
            if col in delivery_df.columns:
                sym_col = col
                break

        if sym_col:
            row = delivery_df[delivery_df[sym_col].str.upper() == symbol.upper()]
            if not row.empty:
                for col in ["DELIV_PER", "DelivPer", "delivery_pct"]:
                    if col in row.columns:
                        return float(row[col].values[0])
        return 0.0

    def _fetch_delivery_from_nse(self, symbol: str) -> float:
        """Fallback: fetch delivery % from NSE quote API."""
        try:
            url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
            resp = self.session.get(url, headers=NSE_HEADERS, timeout=10)
            data = resp.json()
            return float(data.get("securityInfo", {}).get("deliveryToTradedQty", 0))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------ #
    #  Options Open Interest
    # ------------------------------------------------------------------ #

    def get_oi_data(self, symbol: str) -> dict:
        """
        Fetch options chain to compute:
          - PCR (Put/Call Ratio)
          - Max Pain
          - OI buildup at key strikes
        """
        url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
        try:
            resp = self.session.get(url, headers=NSE_HEADERS, timeout=15)
            data = resp.json()
            records = data.get("records", {}).get("data", [])
            if not records:
                return {}

            total_call_oi = total_put_oi = 0
            strikes = {}
            for rec in records:
                strike = rec.get("strikePrice", 0)
                ce = rec.get("CE", {})
                pe = rec.get("PE", {})
                ce_oi = ce.get("openInterest", 0) if ce else 0
                pe_oi = pe.get("openInterest", 0) if pe else 0
                total_call_oi += ce_oi
                total_put_oi += pe_oi
                strikes[strike] = {"ce_oi": ce_oi, "pe_oi": pe_oi}

            pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 1.0

            # Max pain — strike with max total expired OTM OI loss
            max_pain = self._calc_max_pain(strikes)

            return {
                "pcr": pcr,
                "max_pain": max_pain,
                "total_call_oi": total_call_oi,
                "total_put_oi": total_put_oi,
                "pcr_bullish": 0.7 <= pcr <= 1.3,   # Balanced → potential upside
            }
        except Exception as e:
            logger.warning(f"OI fetch failed for {symbol}: {e}")
            return {}

    def _calc_max_pain(self, strikes: dict) -> float:
        """Calculate max pain strike price."""
        if not strikes:
            return 0.0
        min_loss = float("inf")
        max_pain_strike = 0
        for s in strikes:
            total_pain = sum(
                max(0, s2 - s) * v["ce_oi"] + max(0, s - s2) * v["pe_oi"]
                for s2, v in strikes.items()
            )
            if total_pain < min_loss:
                min_loss = total_pain
                max_pain_strike = s
        return max_pain_strike

    # ------------------------------------------------------------------ #
    #  Composite Score
    # ------------------------------------------------------------------ #

    def score(self, symbol: str, df: pd.DataFrame, delivery_df: Optional[pd.DataFrame] = None) -> dict:
        """Combined volume + delivery + OI score (0-100)."""
        vol_result = self.analyze_volume(symbol, df)
        score = vol_result["score"]
        details = vol_result["details"]

        # Delivery %
        del_pct = self.get_delivery_pct(symbol, delivery_df)
        del_thresh = self.cfg["delivery_pct_min"]
        if del_pct >= del_thresh:
            score = min(100, score + 15)
            details["delivery"] = f"Delivery {del_pct:.1f}% (>{del_thresh}%) ✅"

        # OI (optional for index stocks)
        oi = self.get_oi_data(symbol)
        if oi:
            if oi.get("pcr_bullish"):
                score = min(100, score + 5)
                details["pcr"] = f"PCR {oi['pcr']} (balanced) ✅"

        return {"score": score, "delivery_pct": del_pct, "oi": oi, "details": details}
