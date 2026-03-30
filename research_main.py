#!/usr/bin/env python3
import os
import sys
import json
import argparse
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

console = Console()


def print_banner():
    console.print("""
[bold cyan]
╔══════════════════════════════════════════════════════════════╗
║   🧠  India Stock AI Research Engine  v1.0                  ║
╚══════════════════════════════════════════════════════════════╝
[/bold cyan]""")


def print_report(report: dict):
    sym  = report.get("symbol", "?")
    reco = report.get("final_recommendation", "?")
    conv = report.get("conviction_score", 0)
    ts   = report.get("trade_setup", {})

    console.print(Panel(
        f"{reco} | Conviction: {conv}/10 | Setup: {report.get('setup_type')}",
        title=f"{sym} ₹{report.get('_')}",
        border_style="cyan",
    ))

    t = Table(box=box.SIMPLE, show_header=False)
    t.add_column("")
    t.add_column("")

    t.add_row("Entry", str(ts.get("entry")))
    t.add_row("Target", str(ts.get("target")))
    t.add_row("SL", str(ts.get("stop_loss")))
    t.add_row("RR", str(ts.get("rr_ratio")))

    console.print(t)
    console.print()


def main():
    print_banner()

    parser = argparse.ArgumentParser()

    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--from-screener", action="store_true")
    parser.add_argument("--mode", default="btst")
    parser.add_argument("--briefing-only", action="store_true")
    parser.add_argument("--force-deep", action="store_true")

    # 🔥 FIXED: dynamic + correct threshold
    parser.add_argument(
        "--min-score",
        default=int(os.getenv("MIN_SCORE", 35)),
        type=int
    )

    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-alert", action="store_true")
    parser.add_argument("--save-json", action="store_true")

    args = parser.parse_args()

    try:
        import yaml

        try:
            with open(args.config) as f:
                config = yaml.safe_load(f)
        except:
            config = {}

        from ai_research.research_engine import ResearchEngine
        from ai_research.alert_formatter import AIAlertFormatter

        engine = ResearchEngine(config=config)
        alerter = AIAlertFormatter(config=config)

        reports = []

        # 🚀 RUN SCREENER
        screener_results_raw = []
        screener_df = None

        if args.from_screener or not args.symbols:

            console.print(f"[cyan]Running screener...[/cyan]")

            from screener.screener import IndiaStockScreener

            screener = IndiaStockScreener(config_path=args.config)
            screener_df = screener.run(mode=args.mode)

            if not screener_df.empty:

                for _, row in screener_df.iterrows():
                    screener_results_raw.append({
                        "symbol": row.get("Symbol"),
                        "ltp": (
                            row.get("LTP"),
                            or row.get("Close")
                            or row.get("close")
                            or 0
                        ),
                        "composite_score": row.get("Score", 0),
                        "setup_type": row.get("Setup", ""),
                        "technical": {"rsi": row.get("RSI")},
                        "volume": {"spike_ratio": row.get("Vol_Spike")},
                        "trade_setup": {
                            "entry": row.get("Entry"),
                            "target": row.get("Target"),
                            "stop_loss": row.get("SL"),
                            "rr_ratio": row.get("RR"),
                        }
                    })

        # 🚀 FILTER + RESEARCH
        if screener_results_raw:

            console.print(f"[yellow]Filtering min score: {args.min_score}[/yellow]")

            filtered = [
                r for r in screener_results_raw
                if (r.get("composite_score") or 0) >= args.min_score
            ]

            console.print(f"[green]Researching {len(filtered)} symbols[/green]")

            for r in filtered[:10]:

                report = engine.research(
                    symbol=r["symbol"],
                    screener_result=r,
                    force_deep=args.force_deep
                )

                reports.append(report)
                print_report(report)

        # SAVE
        if args.save_json and reports:
            os.makedirs("data", exist_ok=True)
            with open("data/reports.json", "w") as f:
                json.dump(reports, f, indent=2)

        # ALERT
        if not args.no_alert and reports:
            alerter.send_telegram_reports(reports)

        console.print(f"\nTotal reports: {len(reports)}")

    except Exception as e:
        logger.exception(e)
        console.print(f"[red]{e}[/red]")


if __name__ == "__main__":
    main()
