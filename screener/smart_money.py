"""
smart_money.py
--------------
Tracks institutional & smart money signals:
  - FII / DII net activity (daily flows)
  - Bulk deals & Block deals from NSE
  - Insider trading (SAST filings via NSE)
  - Promoter pledging changes
  - Mutual Fund holdings changes (quarterly)
"""

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

    # ------------------------------------------------------------------ #
    #  FII / DII Daily Activity
    # ------------------------------------------------------------------ #

    def get_fii_dii_activity(self) -> dict:
        """
        Fetch today's FII/DII net buy/sell from NSE.
        Returns aggregated net values in crores.
        """
        url = "https://www.nseindia.com/api/fiidiiTradeReact"
        try:
            resp = self.session.get(url, headers=NSE_HEADERS, timeout=15)
            data = resp.json()
            rows = data if isinstance(data, list) else data.get("data", [])

            fii_net = dii_net = 0.0
            for row in rows[:2]:  # Latest 2 entries (Cash + Derivatives or same day)
                category = row.get("category", "").upper()
                net = float(str(row.get("netPurchasesSales", "0")).replace(",", ""))
                if "FII" in category or "FPI" in category:
                    fii_net += net
                elif "DII" in category:
                    dii_net += net

            result = {
                "fii_net_cr": round(fii_net, 2),
                "dii_net_cr": round(dii_net, 2),
                "fii_positive": fii_net > self.cfg["fii_net_buy_threshold_cr"],
                "combined_net_cr": round(fii_net + dii_net, 2),
            }
            logger.info(f"FII/DII: FII={fii_net}Cr, DII={dii_net}Cr")
            return result
        except Exception as e:
            logger.error(f"FII/DII fetch failed: {e}")
            return {"fii_net_cr": 0, "dii_net_cr": 0, "fii_positive": False}

    # ------------------------------------------------------------------ #
    #  Bulk & Block Deals
    # ------------------------------------------------------------------ #

    def get_bulk_deals(self, days_back: int = 3) -> pd.DataFrame:
        """Fetch recent bulk deals from NSE."""
        url = "https://www.nseindia.com/api/bulk-deals"
        try:
            resp = self.session.get(url, headers=NSE_HEADERS, timeout=15)
            data = resp.json()
            rows = data.get("data", [])
            df = pd.DataFrame(rows)
            if df.empty:
                return df

            df["date"] = pd.to_datetime(df.get("date", df.get("BD_DT_DATE", "")))
            cutoff = datetime.now(IST).date() - timedelta(days=days_back)
            df = df[df["date"].dt.date >= cutoff]

            # Normalize column names across API versions
            col_map = {
                "BD_SYMBOL": "symbol", "BD_SCRIP_CD": "symbol",
                "BD_CLIENT_NAME": "client", "BD_QTY_TRD": "qty",
                "BD_TP_WATP": "price", "BD_BUYSELL": "side",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            logger.info(f"Bulk deals fetched: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Bulk deals fetch failed: {e}")
            return pd.DataFrame()

    def get_block_deals(self) -> pd.DataFrame:
        """Fetch block deals from NSE."""
        url = "https://www.nseindia.com/api/block-deals"
        try:
            resp = self.session.get(url, headers=NSE_HEADERS, timeout=15)
            data = resp.json()
            df = pd.DataFrame(data.get("data", []))
            logger.info(f"Block deals fetched: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Block deals fetch failed: {e}")
            return pd.DataFrame()

    def get_deals_for_symbol(self, symbol: str) -> dict:
        """Aggregate bulk/block deal activity for a specific symbol."""
        bulk = self.get_bulk_deals()
        block = self.get_block_deals()
        results = {"bulk_buy": 0, "bulk_sell": 0, "block_value_cr": 0, "notable_buyers": []}

        if not bulk.empty and "symbol" in bulk.columns:
            sym_bulk = bulk[bulk["symbol"].str.upper() == symbol.upper()]
            for _, row in sym_bulk.iterrows():
                side = str(row.get("side", "")).upper()
                qty = float(str(row.get("qty", 0)).replace(",", "") or 0)
                price = float(str(row.get("price", 0)).replace(",", "") or 0)
                value_cr = (qty * price) / 1e7
                if "B" in side:
                    results["bulk_buy"] += value_cr
                    client = str(row.get("client", ""))
                    if client and value_cr >= self.cfg["bulk_deal_min_value_cr"]:
                        results["notable_buyers"].append(client)
                else:
                    results["bulk_sell"] += value_cr

        results["bulk_buy"] = round(results["bulk_buy"], 2)
        results["bulk_sell"] = round(results["bulk_sell"], 2)
        return results

    # ------------------------------------------------------------------ #
    #  Insider Trading (SAST / SEBI Filings)
    # ------------------------------------------------------------------ #

    def get_insider_trades(self, symbol: str) -> dict:
        """
        Fetch insider/promoter trading from NSE SAST filings.
        Returns net promoter activity in shares.
        """
        url = f"https://www.nseindia.com/api/inside-trading?symbol={symbol}"
        try:
            resp = self.session.get(url, headers=NSE_HEADERS, timeout=15)
            data = resp.json()
            trades = data.get("data", [])

            window = self.cfg.get("insider_window_days", 30)
            cutoff = datetime.now(IST).date() - timedelta(days=window)

            buy_qty = sell_qty = 0
            for t in trades:
                try:
                    t_date = pd.to_datetime(t.get("date", "")).date()
                    if t_date < cutoff:
                        continue
                    acq = float(str(t.get("acquisitionDisposal", "0")).replace(",", "") or 0)
                    trans_type = t.get("tdpTransactionType", "")
                    if "Acqu" in trans_type or "Buy" in trans_type:
                        buy_qty += acq
                    else:
                        sell_qty += acq
                except Exception:
                    pass

            return {
                "insider_buy_qty": buy_qty,
                "insider_sell_qty": sell_qty,
                "insider_net_positive": buy_qty > sell_qty and buy_qty > 0,
            }
        except Exception as e:
            logger.warning(f"Insider trades fetch failed for {symbol}: {e}")
            return {"insider_buy_qty": 0, "insider_sell_qty": 0, "insider_net_positive": False}

    # ------------------------------------------------------------------ #
    #  Composite Smart Money Score
    # ------------------------------------------------------------------ #

    def score(self, symbol: str, fii_dii: dict) -> dict:
        """
        Returns a smart money score (0-100) for a symbol.

        Scoring breakdown (each normalized to 0-25):
          - FII net positive day:        8 pts
          - DII net positive day:        4 pts
          - Bulk deal net buy:           8 pts
          - Insider net buy (30d):       5 pts
        """
        pts = 0
        details = {}

        # FII/DII global signal
        if fii_dii.get("fii_positive"):
            pts += 8
            details["fii"] = f"FII net buy ₹{fii_dii['fii_net_cr']}Cr ✅"
        if fii_dii.get("dii_net_cr", 0) > 0:
            pts += 4
            details["dii"] = f"DII net buy ₹{fii_dii['dii_net_cr']}Cr ✅"

        # Bulk/block deals for this symbol
        deals = self.get_deals_for_symbol(symbol)
        net_deal = deals["bulk_buy"] - deals["bulk_sell"]
        if net_deal > 0:
            pts += min(8, int(net_deal / self.cfg["bulk_deal_min_value_cr"]) * 2)
            details["bulk"] = f"Net bulk buy ₹{net_deal:.1f}Cr ✅"
            if deals["notable_buyers"]:
                details["buyers"] = ", ".join(deals["notable_buyers"][:3])

        # Insider trades
        insider = self.get_insider_trades(symbol)
        if insider["insider_net_positive"]:
            pts += 5
            details["insider"] = f"Insider buying {int(insider['insider_buy_qty']):,} shares ✅"

        score = min(100, int(pts / 25 * 100))
        return {"score": score, "raw_pts": pts, "details": details}
