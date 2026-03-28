import requests
import pandas as pd
from loguru import logger
from datetime import datetime, timedelta
import pytz

IST = pytz.timezone("Asia/Kolkata")

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nseindia.com",
}


class SmartMoneyAnalyzer:

    def __init__(self, session: requests.Session, config: dict):
        self.session = session
        self.cfg = config["signals"]["smart_money"]

        # 🔥 CACHE (critical)
        self._bulk_cache = None
        self._block_cache = None
        self._fii_cache = None

    # ---------------- SAFE REQUEST ---------------- #

    def _safe_json(self, url):
        try:
            r = self.session.get(url, headers=NSE_HEADERS, timeout=10)

            if r.status_code != 200:
                return None

            return r.json()

        except Exception:
            return None

    # ---------------- FII/DII ---------------- #

    def get_fii_dii_activity(self):

        if self._fii_cache:
            return self._fii_cache

        url = "https://www.nseindia.com/api/fiidiiTradeReact"
        data = self._safe_json(url)

        if not data:
            return {"fii_net_cr": 0, "dii_net_cr": 0, "fii_positive": False}

        rows = data if isinstance(data, list) else data.get("data", [])

        fii = dii = 0

        for r in rows[:2]:
            cat = str(r.get("category", "")).upper()
            val = float(str(r.get("netPurchasesSales", "0")).replace(",", "") or 0)

            if "FII" in cat or "FPI" in cat:
                fii += val
            elif "DII" in cat:
                dii += val

        res = {
            "fii_net_cr": fii,
            "dii_net_cr": dii,
            "fii_positive": fii > self.cfg["fii_net_buy_threshold_cr"],
        }

        self._fii_cache = res
        return res

    # ---------------- BULK ---------------- #

    def get_bulk_deals(self):

        if self._bulk_cache is not None:
            return self._bulk_cache

        url = "https://www.nseindia.com/api/bulk-deals"
        data = self._safe_json(url)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data.get("data", []))

        if df.empty:
            return df

        try:
            df["symbol"] = df.get("symbol", df.get("BD_SYMBOL", ""))
            df["qty"] = pd.to_numeric(df.get("qty", df.get("BD_QTY_TRD", 0)), errors="coerce")
            df["price"] = pd.to_numeric(df.get("price", df.get("BD_TP_WATP", 0)), errors="coerce")
            df["side"] = df.get("side", df.get("BD_BUYSELL", ""))

        except:
            pass

        self._bulk_cache = df
        return df

    # ---------------- DEALS ---------------- #

    def get_deals_for_symbol(self, symbol):

        df = self.get_bulk_deals()

        if df.empty or "symbol" not in df.columns:
            return {"net": 0}

        sym_df = df[df["symbol"].str.upper() == symbol.upper()]

        if sym_df.empty:
            return {"net": 0}

        buy = sell = 0

        for _, r in sym_df.iterrows():
            val = (r.get("qty", 0) * r.get("price", 0)) / 1e7

            if "B" in str(r.get("side", "")).upper():
                buy += val
            else:
                sell += val

        return {"net": buy - sell}

    # ---------------- INSIDER ---------------- #

    def get_insider_trades(self, symbol):

        url = f"https://www.nseindia.com/api/inside-trading?symbol={symbol}"
        data = self._safe_json(url)

        if not data:
            return False

        trades = data.get("data", [])

        buy = sell = 0

        for t in trades[:10]:  # limit for speed
            val = float(str(t.get("acquisitionDisposal", 0)).replace(",", "") or 0)

            if "Buy" in str(t.get("tdpTransactionType", "")):
                buy += val
            else:
                sell += val

        return buy > sell and buy > 0

    # ---------------- SCORE ---------------- #

    def score(self, symbol, fii_dii):

        pts = 0
        details = {}

        # FII
        if fii_dii.get("fii_positive"):
            pts += 8

        # DII
        if fii_dii.get("dii_net_cr", 0) > 0:
            pts += 4

        # Deals
        deals = self.get_deals_for_symbol(symbol)
        if deals["net"] > self.cfg["bulk_deal_min_value_cr"]:
            pts += 6
            details["bulk"] = deals["net"]

        # Insider
        if self.get_insider_trades(symbol):
            pts += 4
            details["insider"] = "buying"

        score = int(min(100, pts / 25 * 100))

        return {
            "score": score,
            "details": details,
        }
