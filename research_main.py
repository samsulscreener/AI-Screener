#!/usr/bin/env python3
"""
research_main.py
----------------
CLI entrypoint for the AI Research layer.

Usage:
  # Research specific symbols
  python research_main.py --symbols TATAPOWER RELIANCE INFY

  # Research all results from latest screener run
  python research_main.py --from-screener --mode btst

  # Generate daily market briefing only
  python research_main.py --briefing-only

  # Force all symbols through Gemini (skip Groq gating)
  python research_main.py --symbols BAJFINANCE --force-deep
"""
import os
from dotenv import load_dotenv

load_dotenv()
import sys
import os
import json
import argparse
from loguru import logger
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.markdown import Markdown

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

console = Console()


def print_banner():
    console.print("""
[bold cyan]
╔══════════════════════════════════════════════════════════════╗
║   🧠  India Stock AI Research Engine  v1.0                  ║
║   Groq (L1 Fast Triage) + Gemini (L2 Deep Research)        ║
║   + MarketAux News · NSE Smart Money · Options Flow         ║
╚══════════════════════════════════════════════════════════════╝
[/bold cyan]""")


def print_report(report: dict):
    """Pretty-print a single research report to terminal."""
    sym  = report.get("symbol", "?")
    reco = report.get("final_recommendation", "?")
    conv = report.get("conviction_score", 0)
    ts   = report.get("trade_setup", {})
    groq = report.get("_groq_layer", {})
    news = report.get("_news_data", {})

    reco_color = {
        "STRONG BUY": "bold green", "BUY": "green",
        "HOLD": "yellow", "AVOID": "red",
    }.get(reco, "white")

    console.print(Panel(
        f"[{reco_color}]{reco}[/{reco_color}]  Conviction: [bold]{conv}/10[/bold]  "
        f"Setup: [cyan]{report.get('setup_type')}[/cyan]  "
        f"Horizon: {report.get('time_horizon')}",
        title=f"[bold]{sym}[/bold] ₹{report.get('_ltp')}",
        border_style="cyan",
    ))

    # Executive summary
    summary = report.get("executive_summary", "")
    if summary:
        console.print(f"\n[italic]{summary}[/italic]\n")

    # Trade setup table
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column("", style="dim", width=14)
    t.add_column("", style="bold")
    t.add_row("Entry zone",   f"₹{ts.get('entry_zone_low')} – ₹{ts.get('entry_zone_high')}")
    t.add_row("Target 1",     f"₹{ts.get('target_1')}")
    t.add_row("Target 2",     f"₹{ts.get('target_2')}")
    t.add_row("Stop Loss",    f"₹{ts.get('stop_loss')}")
    t.add_row("Risk/Reward",  f"{ts.get('risk_reward')}x")
    t.add_row("Position",     str(ts.get("position_sizing_note", "")))
    console.print(t)

    # Bull/Bear
    bull = report.get("bull_case", {})
    bear = report.get("bear_case", {})
    if bull.get("thesis"):
        console.print(f"[green]✅ Bull:[/green] {bull['thesis']}")
    if bear.get("thesis"):
        console.print(f"[red]⚠️  Bear:[/red] {bear['thesis']}")

    # News
    if news.get("top_headlines"):
        console.print(f"\n[cyan]📰 News ({news.get('article_count', 0)} articles | "
                      f"sentiment: {news.get('avg_sentiment', 0):+.2f}):[/cyan]")
        for h in news["top_headlines"][:3]:
            console.print(f"  • {h[:100]}")

    # Groq quick take
    if groq.get("groq_reasoning"):
        console.print(f"\n[dim]⚡ Groq: {groq['groq_reasoning']}[/dim]")

    console.print()


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="India Stock AI Research Engine")
    parser.add_argument("--symbols",      nargs="+", help="NSE symbols to research")
    parser.add_argument("--from-screener", action="store_true", help="Run screener first, then research results")
    parser.add_argument("--mode",         default="btst", choices=["intraday", "btst", "swing", "all"])
    parser.add_argument("--briefing-only", action="store_true", help="Generate daily market briefing only")
    parser.add_argument("--force-deep",   action="store_true", help="Skip Groq gating, all go to Gemini")
    parser.add_argument("--min-score",    default=60, type=int)
    parser.add_argument("--config",       default="config.yaml")
    parser.add_argument("--no-alert",     action="store_true")
    parser.add_argument("--save-json",    action="store_true", help="Save reports as JSON files")
    args = parser.parse_args()

    try:
        import yaml
        try:
            with open(args.config) as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            config = {}

        from ai_research.research_engine import ResearchEngine
        from ai_research.alert_formatter import AIAlertFormatter

        engine    = ResearchEngine(config=config)
        alerter   = AIAlertFormatter(config=config)
        reports   = []
        briefing  = None

        # ── Market briefing ───────────────────────────────────────────────
        if not args.symbols or args.briefing_only:
            console.print("[cyan]📊 Generating daily market briefing...[/cyan]")
            briefing = engine.get_market_briefing()
            if briefing:
                console.print(Panel(
                    f"Mood: [bold]{briefing.get('market_mood')}[/bold]  "
                    f"Nifty bias: [bold]{briefing.get('nifty_bias')}[/bold]\n\n"
                    f"{briefing.get('trader_guidance', '')}",
                    title="📈 Daily Market Briefing",
                    border_style="blue",
                ))
                if briefing.get("sectors_to_watch"):
                    console.print(f"[green]Sectors to watch:[/green] {', '.join(briefing['sectors_to_watch'])}")
                if briefing.get("risk_warning"):
                    console.print(f"[yellow]⚠️  Risk:[/yellow] {briefing['risk_warning']}")
                console.print()

        if args.briefing_only:
            return

        # ── Screener run ──────────────────────────────────────────────────
        screener_results_raw = []
        screener_df          = None

        if args.from_screener or not args.symbols:
            console.print(f"[cyan]🚀 Running screener (mode: {args.mode})...[/cyan]")
            try:
                from screener.screener import IndiaStockScreener
                screener  = IndiaStockScreener(config_path=args.config)
                engine.screener = screener
                screener_df = screener.run(mode=args.mode)
                # Need raw results — re-expose via the screener or reconstruct
                # For now, re-build minimal result dicts from DF
                if not screener_df.empty:
                    for _, row in screener_df.iterrows():
                        screener_results_raw.append({
                            "symbol":          row.get("Symbol", ""),
                            "ltp":             row.get("LTP", 0),
                            "composite_score": row.get("Score", 0),
                            "setup_type":      row.get("Setup", "BTST"),
                            "sector":          row.get("Sector", ""),
                            "scores": {
                                "smart_money": row.get("SM_Score", 0),
                                "volume":      row.get("Vol_Score", 0),
                                "technical":   row.get("Tech_Score", 0),
                                "news":        row.get("News_Score", 0),
                                "fundamental": row.get("Fund_Score", 0),
                            },
                            "technical": {
                                "rsi":            row.get("RSI"),
                                "supertrend_buy": row.get("ST_BUY"),
                                "ema_aligned":    False,
                                "patterns":       [],
                            },
                            "volume": {
                                "spike_ratio":  row.get("Vol_Spike"),
                                "delivery_pct": row.get("Delivery%"),
                                "oi":           {},
                            },
                            "smart_money": {},
                            "trade_setup": {
                                "entry_low":  row.get("Entry_Low"),
                                "entry_high": row.get("Entry_High"),
                                "target":     row.get("Target"),
                                "stop_loss":  row.get("SL"),
                                "rr_ratio":   row.get("RR"),
                            },
                            "all_details":  {},
                            "news": {},
                        })
            except ImportError:
                console.print("[yellow]⚠️  Screener module not found. Use --symbols instead.[/yellow]")

        # ── Direct symbol research ────────────────────────────────────────
        if args.symbols:
            for sym in args.symbols:
                dummy_result = {
                    "symbol": sym.upper(), "ltp": 0, "composite_score": 70,
                    "setup_type": "BTST", "sector": "", "scores": {},
                    "technical": {}, "volume": {}, "smart_money": {},
                    "trade_setup": {}, "all_details": {}, "news": {},
                }
                report = engine.research(
                    symbol=sym.upper(),
                    screener_result=dummy_result,
                    force_deep=args.force_deep,
                )
                reports.append(report)
                print_report(report)

        elif screener_results_raw:
            import pandas as pd
            reports = engine.research_all(
                screener_df          = screener_df,
                screener_results_raw = screener_results_raw,
                min_score            = args.min_score,
                max_symbols          = 10,
            )
            for r in reports:
                print_report(r)

        # ── Save JSON ─────────────────────────────────────────────────────
        if args.save_json and reports:
            os.makedirs("data/research", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            path = f"data/research/research_{ts}.json"
            with open(path, "w") as f:
                json.dump(reports, f, indent=2, default=str)
            console.print(f"[dim]Reports saved: {path}[/dim]")

        # ── Telegram alerts ───────────────────────────────────────────────
        if not args.no_alert and reports:
            console.print("[cyan]📬 Sending Telegram alerts...[/cyan]")
            alerter.send_telegram_reports(reports, briefing)

        console.print(f"\n[bold]Total AI research reports: {len(reports)}[/bold]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    except Exception as e:
        logger.exception(f"Research engine error: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
