"""
alerts.py
---------
Send high-conviction alerts via Telegram and/or Email.
"""

import os
import asyncio
from loguru import logger
from typing import List


class AlertManager:
    def __init__(self, config: dict):
        self.cfg = config.get("alerts", {})
        self.tg_cfg = self.cfg.get("telegram", {})
        self.min_score = self.tg_cfg.get("min_score", 70)
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    def _format_telegram_message(self, result: dict) -> str:
        """Format a result dict into a Telegram-ready message."""
        s = result["scores"]
        ts = result.get("trade_setup", {})
        news_headlines = result.get("news", {}).get("headlines", [])
        top_headline = news_headlines[0]["headline"] if news_headlines else "—"

        sm_details = result.get("smart_money", {})
        fii_line = sm_details.get("fii_positive") or "No notable FII activity"
        bulk_line = sm_details.get("bulk_deals") or "—"
        insider_line = sm_details.get("insider") or "—"
        patterns = result.get("technical", {}).get("patterns", [])

        msg = f"""
{result['emoji']} *{result['signal']} — Score: {result['composite_score']}/100*

📌 *{result['symbol']}* | NSE | ₹{result['ltp']}
📊 *Setup:* {result['setup_type']}

*Signal Scores:*
├ 🏦 Smart Money:  {s['smart_money']}/100
├ 📊 Volume:       {s['volume']}/100
├ 📈 Technical:    {s['technical']}/100
├ 📰 News:         {s['news']}/100
└ 💡 Fundamental:  {s['fundamental']}/100

*Smart Money:*
• {fii_line}
• Bulk/Block: {bulk_line}
• Insider: {insider_line}

*Technical:*
• RSI: {result['technical'].get('rsi', '—')}
• Supertrend: {'BUY ✅' if result['technical'].get('supertrend_buy') else 'SELL ❌'}
• EMA Stack: {'Aligned ✅' if result['technical'].get('ema_aligned') else 'Mixed'}
• Pattern: {', '.join(patterns) if patterns else '—'}

*Volume:*
• Spike: {result['volume'].get('spike_ratio', '—')}x avg
• Delivery %: {result['volume'].get('delivery_pct', '—')}%

*News:* {top_headline[:100]}

*Trade Setup:*
⚡ Entry: ₹{ts.get('entry_low', '—')} – ₹{ts.get('entry_high', '—')}
🎯 Target: ₹{ts.get('target', '—')}
🛡️ SL: ₹{ts.get('stop_loss', '—')} | R/R: {ts.get('rr_ratio', '—')}x
""".strip()
        return msg

    def send_telegram(self, results: List[dict]) -> None:
        """Send Telegram alerts for all high-score results."""
        if not self.tg_cfg.get("enabled") or not self.token or not self.chat_id:
            logger.info("Telegram alerts disabled or not configured.")
            return

        qualifying = [r for r in results if r["composite_score"] >= self.min_score]
        if not qualifying:
            logger.info("No qualifying results for Telegram alert.")
            return

        asyncio.run(self._send_all(qualifying))

    async def _send_all(self, results: List[dict]):
        try:
            from telegram import Bot
            bot = Bot(token=self.token)

            # Summary message first
            summary = f"🚨 *India Stock Screener — {len(results)} Strong Setup(s)*\n\n"
            for r in results[:10]:
                summary += f"• {r['symbol']} | Score: {r['composite_score']} | {r['setup_type']} | ₹{r['ltp']}\n"
            await bot.send_message(chat_id=self.chat_id, text=summary, parse_mode="Markdown")

            # Individual detailed messages
            for r in results[:5]:  # Cap detailed alerts at 5
                msg = self._format_telegram_message(r)
                await bot.send_message(chat_id=self.chat_id, text=msg, parse_mode="Markdown")

            logger.info(f"Telegram: {len(results)} alerts sent")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    def send_email(self, results: List[dict]) -> None:
        """Send email summary (optional)."""
        email_cfg = self.cfg.get("email", {})
        if not email_cfg.get("enabled"):
            return
        # Email implementation using smtplib
        logger.info("Email alerts: implementation pending SMTP configuration")
