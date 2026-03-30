from loguru import logger
import os
import asyncio
from typing import List


class AIAlertFormatter:

    def __init__(self, config: dict = None):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.config = config or {}

    # ---------------- SAFE TEXT ---------------- #

    def _safe(self, x):
        if x is None:
            return "N/A"
        return str(x)

    # ---------------- MARKET BRIEF ---------------- #

    def format_market_briefing(self, briefing: dict) -> str:

        return f"""
📊 Daily Market Briefing

Mood: {self._safe(briefing.get('market_mood'))}
Bias: {self._safe(briefing.get('nifty_bias'))}

Guidance:
{self._safe(briefing.get('trader_guidance'))}
""".strip()

    # ---------------- REPORT ---------------- #

    def format_research_report(self, report: dict) -> str:

        ts = report.get("trade_setup", {})

        return f"""
📌 {self._safe(report.get('symbol'))} | ₹{self._safe(report.get('_ltp'))}

Recommendation: {self._safe(report.get('final_recommendation'))}
Conviction: {self._safe(report.get('conviction_score'))}/10
Setup: {self._safe(report.get('setup_type'))}

Entry: {self._safe(ts.get('entry') or ts.get('entry_zone_low'))}
Target: {self._safe(ts.get('target') or ts.get('target_1'))}
Stop Loss: {self._safe(ts.get('stop_loss'))}
RR: {self._safe(ts.get('rr_ratio') or ts.get('risk_reward'))}

Summary:
{self._safe(report.get('summary') or report.get('executive_summary'))}
""".strip()

    # ---------------- SUMMARY ---------------- #

    def format_summary_table(self, reports: List[dict]):

        if not reports:
            return "No setups found."

        lines = [f"AI Research: {len(reports)} setups\n"]

        for r in reports:
            lines.append(
                f"{self._safe(r.get('symbol'))} | ₹{self._safe(r.get('_ltp'))} | "
                f"{self._safe(r.get('final_recommendation'))} "
                f"({self._safe(r.get('conviction_score'))}/10)"
            )

        return "\n".join(lines)

    # ---------------- TELEGRAM ---------------- #

    def send_telegram_reports(self, reports: List[dict], briefing: dict = None):

        if not self.token or not self.chat_id:
            logger.info("Telegram not configured")
            return

        asyncio.run(self._send(reports, briefing))

    async def _send(self, reports, briefing):

        try:
            from telegram import Bot

            bot = Bot(token=self.token)

            # 🔥 NO Markdown → NO errors
            parse_mode = None

            if briefing:
                await bot.send_message(
                    chat_id=self.chat_id,
                    text=self.format_market_briefing(briefing),
                    parse_mode=parse_mode
                )

            if reports:
                await bot.send_message(
                    chat_id=self.chat_id,
                    text=self.format_summary_table(reports),
                    parse_mode=parse_mode
                )

            for r in reports[:5]:
                await bot.send_message(
                    chat_id=self.chat_id,
                    text=self.format_research_report(r),
                    parse_mode=parse_mode
                )
                await asyncio.sleep(1)

            logger.info(f"Telegram sent: {len(reports)} reports")

        except Exception as e:
            logger.error(f"Telegram failed: {e}")
