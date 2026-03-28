"""
alert_formatter.py
------------------
Formats AI research reports into rich Telegram and email alerts.
"""

import os
import asyncio
from loguru import logger
from typing import List


class AIAlertFormatter:

    def __init__(self, config: dict = None):
        self.token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.config  = config or {}
        self.min_score = self.config.get("alerts", {}).get("telegram", {}).get("min_score", 70)

    def format_market_briefing(self, briefing: dict) -> str:
        mood_emoji = {"RISK_ON": "🟢", "RISK_OFF": "🔴", "NEUTRAL": "🟡"}.get(briefing.get("market_mood", ""), "⚪")
        bias_emoji = {"BULLISH": "↗️", "BEARISH": "↘️", "SIDEWAYS": "→"}.get(briefing.get("nifty_bias", ""), "→")
        sectors    = "\n".join(f"  ✅ {s}" for s in briefing.get("sectors_to_watch", []))
        avoid      = "\n".join(f"  ⚠️ {s}" for s in briefing.get("sectors_to_avoid", []))
        themes     = "\n".join(f"  💡 {t}" for t in briefing.get("top_3_themes", []))
        levels     = briefing.get("key_levels", {})

        return f"""
{mood_emoji} *Daily Market Briefing*
Mood: *{briefing.get('market_mood')}* | Nifty bias: {bias_emoji} *{briefing.get('nifty_bias')}*

*Key Levels:*
Nifty support: {levels.get('nifty_support', ['N/A', 'N/A'])[0]} / {levels.get('nifty_support', ['N/A', 'N/A'])[-1]}
Nifty resistance: {levels.get('nifty_resistance', ['N/A', 'N/A'])[0]} / {levels.get('nifty_resistance', ['N/A', 'N/A'])[-1]}

*FII/DII:* {briefing.get('fii_dii_interpretation', 'N/A')}

*Global impact:* {briefing.get('global_impact', 'N/A')}

*Themes today:*
{themes}

*Sectors to watch:*
{sectors}

{f"*Avoid today:*{chr(10)}{avoid}" if avoid else ""}

*Trader guidance:*
{briefing.get('trader_guidance', 'N/A')}

{f"⚠️ *Risk:* {briefing.get('risk_warning', '')}" if briefing.get('risk_warning') else ""}
""".strip()

    def format_research_report(self, report: dict) -> str:
        ts       = report.get("trade_setup", {})
        bull     = report.get("bull_case", {})
        bear     = report.get("bear_case", {})
        news_cat = report.get("news_catalyst_analysis", {})
        sm       = report.get("smart_money_analysis", {})
        groq     = report.get("_groq_layer", {})
        news_d   = report.get("_news_data", {})
        tech     = report.get("technical_analysis", {})

        reco_emoji = {
            "STRONG BUY": "🔴🔴", "BUY": "🟢",
            "HOLD": "🟡", "AVOID": "⚫",
        }.get(report.get("final_recommendation", ""), "⚪")

        headlines = "\n".join(
            f"  • {h}" for h in news_d.get("top_headlines", [])[:3]
        ) or "  No recent headlines"

        catalysts = "\n".join(
            f"  ✅ {c}" for c in bull.get("key_drivers", [])
        ) or "  N/A"

        risks = "\n".join(
            f"  ⚠️ {r}" for r in bear.get("key_risks", [])
        ) or "  N/A"

        watch_levels = "\n".join(
            f"  📍 {w}" for w in report.get("risk_management", {}).get("key_watch_levels", [])
        ) or "  N/A"

        return f"""
{reco_emoji} *{report.get('final_recommendation')}* — Conviction: {report.get('conviction_score')}/10

📌 *{report.get('symbol')}* | ₹{report.get('_ltp')} | {report.get('setup_type')} | {report.get('time_horizon')}

_{report.get('executive_summary', '')}_

*🏦 Smart Money:*
{sm.get('summary', 'N/A')}

*📈 Trade Setup:*
Entry: ₹{ts.get('entry_zone_low')} – ₹{ts.get('entry_zone_high')}
Target 1: ₹{ts.get('target_1')} | Target 2: ₹{ts.get('target_2')}
Stop Loss: ₹{ts.get('stop_loss')} | R/R: {ts.get('risk_reward')}x
_{ts.get('position_sizing_note', '')}_

*📰 News ({news_d.get('article_count', 0)} articles | sentiment: {news_d.get('avg_sentiment', 0):+.2f}):*
{headlines}
Top catalyst: {news_cat.get('top_catalyst', 'N/A')}

*✅ Bull drivers:*
{catalysts}

*⚠️ Key risks:*
{risks}

*📍 Watch levels:*
{watch_levels}

*⚡ Quick take (Groq L1):* {groq.get('groq_reasoning', 'N/A')}
*🔬 Data quality:* {report.get('gemini_confidence_note', 'N/A')}
""".strip()

    def format_summary_table(self, reports: List[dict]) -> str:
        if not reports:
            return "No research reports available."
        lines = [f"🧠 *AI Research — {len(reports)} Setups*\n"]
        for r in reports:
            reco  = r.get("final_recommendation", "")
            score = r.get("conviction_score", 0)
            setup = r.get("setup_type", "")
            sym   = r.get("symbol", "")
            ltp   = r.get("_ltp", 0)
            ts    = r.get("trade_setup", {})
            lines.append(
                f"• *{sym}* ₹{ltp} | {reco} ({score}/10) | {setup} "
                f"| T1: ₹{ts.get('target_1','?')} | SL: ₹{ts.get('stop_loss','?')}"
            )
        return "\n".join(lines)

    def send_telegram_reports(self, reports: List[dict], briefing: dict = None):
        if not self.token or not self.chat_id:
            logger.info("Telegram not configured — skipping alerts")
            return
        asyncio.run(self._send_all_telegram(reports, briefing))

    async def _send_all_telegram(self, reports: List[dict], briefing: dict = None):
        try:
            from telegram import Bot
            from telegram.constants import ParseMode
            bot = Bot(token=self.token)

            # 1. Market briefing
            if briefing:
                msg = self.format_market_briefing(briefing)
                await bot.send_message(chat_id=self.chat_id, text=msg, parse_mode=ParseMode.MARKDOWN)

            # 2. Summary table
            if reports:
                summary = self.format_summary_table(reports)
                await bot.send_message(chat_id=self.chat_id, text=summary, parse_mode=ParseMode.MARKDOWN)

            # 3. Individual detailed reports (top 5 only)
            for r in reports[:5]:
                msg = self.format_research_report(r)
                await bot.send_message(chat_id=self.chat_id, text=msg, parse_mode=ParseMode.MARKDOWN)
                import asyncio as _a; await _a.sleep(1)

            logger.info(f"Telegram: sent {len(reports)} AI research alerts")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
