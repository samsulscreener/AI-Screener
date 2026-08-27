"""
alerts.py  (v2)
---------------
FIXES vs v1:
  * v1 would have crashed on the first message: _format_telegram_message read
    result['emoji'] (never set by the scorer), ts.get('entry_low')/'entry_high'
    (the scorer emits 'entry'), plus supertrend_buy / ema_aligned / patterns /
    delivery_pct which no analyzer produced. It never actually ran because
    main.py never called send_telegram — so the breakage was invisible.
  * parse_mode switched from Markdown to HTML. Telegram's legacy Markdown
    breaks on any unescaped _ * [ ` in a company name or headline and the
    send fails with a 400. HTML only needs & < > escaped.
  * Messages are chunked under Telegram's 4096-char limit.
  * Sends synchronously via the HTTP API — no python-telegram-bot dependency,
    no asyncio.run() inside a possibly-already-running loop.
"""

import os
import html
import time
import requests
from loguru import logger
from typing import List

API = "https://api.telegram.org/bot{token}/sendMessage"
MAX_LEN = 3900


class AlertManager:

    def __init__(self, config: dict):
        self.cfg = config.get("alerts", {})
        self.tg = self.cfg.get("telegram", {})
        self.min_score = float(self.tg.get("min_score", 70))
        self.max_detailed = int(self.tg.get("max_detailed", 5))
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    # ------------------------------------------------------------------ #

    @staticmethod
    def _e(x):
        """Escape for Telegram HTML parse mode."""
        return html.escape(str(x if x is not None else "—"))

    @staticmethod
    def _fmt(v, suffix="", dash="—"):
        if v is None or v == "":
            return dash
        try:
            return f"{float(v):g}{suffix}"
        except (TypeError, ValueError):
            return str(v)

    def _signal_icon(self, result):
        return {"BUY": "🟢", "WATCH": "🟡", "WATCH_NO_ENTRY": "🟠",
                "WEAK": "⚪", "REJECTED": "🔴"}.get(result.get("signal"), "⚪")

    # ------------------------------------------------------------------ #

    def format_message(self, r: dict) -> str:
        """Keys here match exactly what Scorer.build_result emits."""
        s = r.get("scores", {}) or {}
        avail = r.get("available", {}) or {}
        t = r.get("technical", {}) or {}
        v = r.get("volume", {}) or {}
        sm = (r.get("smart_money", {}) or {}).get("details", {}) or {}
        fd = (r.get("fundamental", {}) or {}).get("details", {}) or {}
        ts = r.get("trade_setup", {}) or {}
        news = r.get("news", {}) or {}

        def line(label, key, icon):
            if not avail.get(key):
                return f"{icon} {label}: <i>no data</i>"
            return f"{icon} {label}: <b>{self._fmt(s.get(key))}</b>/100"

        pct = r.get("percentile")
        pct_txt = f" · pctile {pct:.0f}" if pct is not None else ""

        parts = [
            f"{self._signal_icon(r)} <b>{self._e(r.get('symbol'))}</b> — "
            f"{self._e(r.get('signal'))} <b>{self._fmt(r.get('composite_score'))}</b>/100{pct_txt}",
            f"₹{self._fmt(r.get('ltp'))} · {self._e(r.get('setup_type'))} · "
            f"regime: {self._e(r.get('regime'))}",
            "",
            "<b>Signals</b>",
            line("Smart money", "smart_money", "🏦"),
            line("Volume", "volume", "📊"),
            line("Technical", "technical", "📈"),
            line("News", "news", "📰"),
            line("Fundamental", "fundamental", "💡"),
            "",
            "<b>Technical</b>",
            f"RSI {self._fmt(t.get('rsi'))} · ATR {self._fmt(t.get('atr_pct'), '%')} · "
            f"Supertrend {'BUY' if t.get('supertrend_buy') else 'SELL'}",
            f"EMA stack {'aligned' if t.get('ema_aligned') else 'mixed'} · "
            f"{self._fmt(t.get('pct_from_20d_high'), '%')} from 20d high",
            "",
            "<b>Volume</b>",
            f"Rel vol {self._fmt(v.get('spike_ratio'), 'x')} · "
            f"Delivery {self._fmt(v.get('delivery_pct'), '%')} · "
            f"Accum {self._fmt(v.get('accumulation_ratio'), 'x')}",
        ]

        if sm:
            parts += ["", "<b>Smart money</b>",
                      " · ".join(f"{self._e(k)}: {self._e(val)}" for k, val in sm.items())]
        if fd:
            parts += ["", "<b>Fundamentals</b>",
                      " · ".join(f"{self._e(k)}: {self._e(val)}" for k, val in fd.items())]

        heads = news.get("headlines") or []
        if heads:
            parts += ["", "<b>News</b>"]
            parts += [f"• {self._e(h.get('headline', ''))[:110]}" for h in heads[:2]]

        if ts.get("valid"):
            parts += [
                "", "<b>Trade setup</b> <i>(ATR-scaled)</i>",
                f"Entry ₹{self._fmt(ts.get('entry'))} · "
                f"T1 ₹{self._fmt(ts.get('target_1'))} · T2 ₹{self._fmt(ts.get('target_2'))}",
                f"SL ₹{self._fmt(ts.get('stop_loss'))} "
                f"(risk {self._fmt(ts.get('risk_pct'), '%')}) · "
                f"R/R {self._fmt(ts.get('rr_ratio'))}x",
            ]
        else:
            parts += ["", f"<i>No valid entry: {self._e(ts.get('reason', 'n/a'))}</i>"]

        return "\n".join(parts)

    # ------------------------------------------------------------------ #

    def _post(self, text):
        try:
            resp = requests.post(
                API.format(token=self.token),
                json={"chat_id": self.chat_id, "text": text[:MAX_LEN],
                      "parse_mode": "HTML", "disable_web_page_preview": True},
                timeout=15,
            )
            if resp.status_code != 200:
                logger.error(f"Telegram {resp.status_code}: {resp.text[:300]}")
                return False
            return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def send_telegram(self, results: List[dict], regime: dict = None) -> int:
        if not self.tg.get("enabled"):
            logger.info("Telegram alerts disabled in config")
            return 0
        if not self.token or not self.chat_id:
            logger.warning("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — no alerts sent")
            return 0

        qualifying = [
            r for r in results
            if r.get("composite_score", 0) >= self.min_score
            and r.get("signal") in ("BUY", "WATCH")
        ]
        if not qualifying:
            logger.info(f"No results at or above score {self.min_score}")
            return 0

        regime = regime or {}
        header = [
            f"🚨 <b>Screener — {len(qualifying)} setup(s)</b>",
            f"Regime: <b>{self._e(regime.get('regime', 'unknown'))}</b>"
            + (f" ({self._e('; '.join(regime.get('reasons', [])))})" if regime.get("reasons") else ""),
            "",
        ]
        for r in qualifying[:15]:
            header.append(
                f"{self._signal_icon(r)} <b>{self._e(r['symbol'])}</b> "
                f"{self._fmt(r.get('composite_score'))} · {self._e(r.get('setup_type'))} · "
                f"₹{self._fmt(r.get('ltp'))}")
        self._post("\n".join(header))

        sent = 0
        for r in qualifying[:self.max_detailed]:
            if self._post(self.format_message(r)):
                sent += 1
            time.sleep(0.4)          # Telegram rate limit

        logger.info(f"Telegram: {sent} detailed alert(s) sent")
        return sent

    def send_email(self, results: List[dict]) -> None:
        if not self.cfg.get("email", {}).get("enabled"):
            return
        logger.info("Email alerts not implemented — configure SMTP first")
