"""
scripts/run_scheduler.py
------------------------
In-process scheduler using APScheduler.
Runs the screener on Indian market hours automatically.
Use this inside Docker or on a VPS instead of GitHub Actions cron.

Run: python scripts/run_scheduler.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from loguru import logger
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")


def load_config(path="config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_screen(mode: str):
    """Execute one screener run."""
    from screener.screener import IndiaStockScreener
    from screener.alerts import AlertManager

    logger.info(f"⏰ Scheduled run triggered | Mode: {mode.upper()} | {datetime.now(IST).strftime('%H:%M IST')}")
    try:
        cfg = load_config()
        screener = IndiaStockScreener()
        df = screener.run(mode=mode, max_workers=6)

        if not df.empty:
            logger.info(f"✅ {len(df)} setups found. Sending alerts...")
            # alerts handled inside screener / AlertManager
        else:
            logger.info("No qualifying setups this run.")
    except Exception as e:
        logger.error(f"Scheduled run failed: {e}")


def main():
    cfg = load_config()
    sched_cfg = cfg.get("schedule", {})

    scheduler = BlockingScheduler(timezone=IST)

    # ── Intraday ──────────────────────────────────────────────
    if sched_cfg.get("intraday", {}).get("enabled", True):
        times = sched_cfg["intraday"].get("times", ["09:20", "10:00", "11:00", "13:00", "14:30"])
        for t in times:
            h, m = t.split(":")
            scheduler.add_job(
                run_screen,
                CronTrigger(day_of_week="mon-fri", hour=int(h), minute=int(m), timezone=IST),
                args=["intraday"],
                name=f"Intraday-{t}",
                misfire_grace_time=300,
            )
            logger.info(f"Scheduled INTRADAY run at {t} IST (Mon–Fri)")

    # ── BTST ──────────────────────────────────────────────────
    if sched_cfg.get("btst", {}).get("enabled", True):
        t = sched_cfg["btst"].get("time", "15:10")
        h, m = t.split(":")
        scheduler.add_job(
            run_screen,
            CronTrigger(day_of_week="mon-fri", hour=int(h), minute=int(m), timezone=IST),
            args=["btst"],
            name="BTST",
            misfire_grace_time=300,
        )
        logger.info(f"Scheduled BTST run at {t} IST (Mon–Fri)")

    # ── Swing ─────────────────────────────────────────────────
    if sched_cfg.get("swing", {}).get("enabled", True):
        t = sched_cfg["swing"].get("time", "16:00")
        h, m = t.split(":")
        scheduler.add_job(
            run_screen,
            CronTrigger(day_of_week="fri", hour=int(h), minute=int(m), timezone=IST),
            args=["swing"],
            name="Swing-Friday",
            misfire_grace_time=600,
        )
        logger.info(f"Scheduled SWING run at {t} IST (Friday)")

    logger.info("🚀 Scheduler started. Waiting for next market trigger...")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
