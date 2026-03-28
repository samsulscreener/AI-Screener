import numpy as np
import pandas as pd
from loguru import logger
import requests

NSE_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com"}


class VolumeAnalyzer:

    def __init__(self, session: requests.Session, config: dict):
        self.session = session
        self.cfg = config["signals"]["volume"]

    # ---------------- Volume ---------------- #

    def analyze_volume(self, symbol, df):

        if df is None or len(df) < 25:
            return {"score": 0, "details": {}}

        try:
            vol = df["Volume"].values
            close = df["Close"].values

            # SAFE slicing
            vol_20avg = np.nanmean(vol[-21:-1])
            vol_today = vol[-1]

            if vol_20avg <= 0 or np.isnan(vol_20avg):
                return {"score": 0, "details": {}}

            spike_ratio = vol_today / vol_20avg

            price_up = close[-1] > close[-2]

            vol_5avg = np.nanmean(vol[-6:-1])
            vol_trend_up = vol_5avg > vol_20avg

            # A/D
            ad = self._calc_ad(df)
            ad_trend = len(ad) > 5 and (ad[-1] > ad[-6])

            # Up vs down volume
            up_df = df[df["Close"] > df["Open"]]
            dn_df = df[df["Close"] <= df["Open"]]

            up_vol = up_df["Volume"].tail(10).mean() if not up_df.empty else 0
            dn_vol = dn_df["Volume"].tail(10).mean() if not dn_df.empty else 1

            vol_quality = up_vol / dn_vol if dn_vol > 0 else 1

            pts = 0
            details = {}

            # -------- scoring -------- #

            if spike_ratio >= self.cfg["spike_multiplier"]:
                pts += 10
                details["spike"] = f"{spike_ratio:.1f}x"

            elif spike_ratio >= 1.5:
                pts += 5

            if price_up and vol_today > vol_20avg:
                pts += 5

            if vol_trend_up:
                pts += 3

            if ad_trend:
                pts += 4

            if vol_quality >= 1.5:
                pts += 3

            score = int(min(100, pts / 25 * 100))

            return {
                "score": score,
                "spike_ratio": round(spike_ratio, 2),
                "details": details,
            }

        except Exception as e:
            logger.error(f"{symbol} volume error: {e}")
            return {"score": 0, "details": {}}

    # ---------------- AD ---------------- #

    def _calc_ad(self, df):

        high = df["High"].values
        low = df["Low"].values
        close = df["Close"].values
        vol = df["Volume"].values

        mfm = ((close - low) - (high - close)) / (high - low + 1e-9)
        return np.cumsum(mfm * vol)

    # ---------------- Delivery ---------------- #

    def get_delivery_pct(self, symbol, delivery_df):

        try:
            if delivery_df is None or delivery_df.empty:
                return self._fetch_delivery(symbol)

            sym_col = next((c for c in delivery_df.columns if c.lower() == "symbol"), None)

            if sym_col:
                row = delivery_df[delivery_df[sym_col].str.upper() == symbol.upper()]

                if not row.empty:
                    val = row.iloc[0]
                    for c in row.columns:
                        if "deliv" in c.lower():
                            return float(val[c])

            return 0.0

        except:
            return 0.0

    def _fetch_delivery(self, symbol):

        try:
            url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
            r = self.session.get(url, headers=NSE_HEADERS, timeout=10)

            if r.status_code != 200:
                return 0.0

            data = r.json()
            return float(data.get("securityInfo", {}).get("deliveryToTradedQty", 0))

        except:
            return 0.0

    # ---------------- OI ---------------- #

    def get_oi_data(self, symbol):

        try:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
            r = self.session.get(url, headers=NSE_HEADERS, timeout=10)

            if r.status_code != 200:
                return {}

            data = r.json()
            recs = data.get("records", {}).get("data", [])

            call_oi = sum((r.get("CE") or {}).get("openInterest", 0) for r in recs)
            put_oi = sum((r.get("PE") or {}).get("openInterest", 0) for r in recs)

            pcr = put_oi / call_oi if call_oi > 0 else 1

            return {
                "pcr": round(pcr, 2),
                "bullish": 0.7 <= pcr <= 1.3,
            }

        except:
            return {}

    # ---------------- FINAL ---------------- #

    def score(self, symbol, df, delivery_df=None):

        vol = self.analyze_volume(symbol, df)

        score = vol["score"]
        details = vol["details"]

        # delivery
        d = self.get_delivery_pct(symbol, delivery_df)

        if d >= self.cfg["delivery_pct_min"]:
            score = min(100, score + 10)
            details["delivery"] = d

        # OI
        oi = self.get_oi_data(symbol)

        if oi.get("bullish"):
            score = min(100, score + 5)

        return {
            "score": score,
            "delivery_pct": d,
            "oi": oi,
            "details": details,
        }
