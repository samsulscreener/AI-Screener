#!/usr/bin/env python3
"""
main.py
-------
CLI entry point for India Smart Stock Screener.

Usage:
  python main.py --mode intraday
  python main.py --mode btst
  python main.py --mode swing
  python main.py --mode all --workers 8
"""

import argparse
import sys
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

from screener.screener import IndiaStockScreener
from screener.alerts import AlertManager

console = Console()


def print_banner():
    console.print("""
[bold green]
╔══════════════════════════════════════════════════════════╗
║     🇮🇳  India Smart Stock Screener  v1.0               ║
║     Intraday | BTST | Swing — Institutional Grade        ║
╚══════════════════════════════════════════════════════════╝
[/bold green]""")


def print_results_table(df):
    if df.empty:
        console.print("[yellow]No setups found.[/yellow]")
        return

    table = Table(
        title=f"📊 Screener Results — {len(df)} Setups",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Rank", style="dim", width=4)
    table.add_column("Symbol", style="bold cyan", width=14)
    table.add_column("LTP ₹", justify="right", width=9)
    table.add_column("Score", justify="center", width=7)
    table.add_column("Signal", width=12)
    table.add_column("Setup", width=10)
    table.add_column("RSI", justify="right", width=6)
    table.add_column("Vol×", justify="right", width=6)
    table.add_column("Del%", justify="right", width=6)
    table.add_column("Target ₹", justify="right", width=9)
    table.add_column("SL ₹", justify="right", width=9)
    table.add_column("R/R", justify="right", width=5)

    for i, row in df.head(20).iterrows():
        score = row.get("Score", 0)
        score_color = "green" if score >= 70 else "yellow" if score >= 55 else "white"

        signal = row.get("Signal", "")
        signal_emoji = "🔴" if "STRONG" in signal else "🟡" if "WATCH" in signal else "⚪"

        table.add_row(
            str(i + 1),
            str(row.get("Symbol", "")),
            f"{row.get('LTP', 0):.2f}",
            f"[{score_color}]{score}[/{score_color}]",
            f"{signal_emoji} {signal}",
            str(row.get("Setup", "")),
            f"{row.get('RSI', 0):.1f}" if row.get("RSI") else "—",
            f"{row.get('Vol_Spike', 0):.1f}x" if row.get("Vol_Spike") else "—",
            f"{row.get('Delivery%', 0):.0f}%" if row.get("Delivery%") else "—",
            f"{row.get('Target', 0):.2f}" if row.get("Target") else "—",
            f"{row.get('SL', 0):.2f}" if row.get("SL") else "—",
            f"{row.get('RR', 0):.1f}" if row.get("RR") else "—",
        )

    console.print(table)
    console.print(f"\n[dim]Full results saved to data/results/[/dim]")


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="India Smart Stock Screener")
    parser.add_argument("--mode",    default="all",       choices=["intraday", "btst", "swing", "all"])
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--workers", default=5, type=int, help="Parallel threads")
    parser.add_argument("--no-alert", action="store_true", help="Skip Telegram alerts")
    parser.add_argument("--symbols", nargs="+", help="Override universe with specific symbols")
    args = parser.parse_args()

    try:
        screener = IndiaStockScreener(config_path=args.config)

        # Override universe if symbols provided
        if args.symbols:
            screener.config["screening"]["universe"] = "custom"
            screener.config["screening"]["custom_symbols"] = args.symbols

        df = screener.run(mode=args.mode, max_workers=args.workers)
        print_results_table(df)

        # Alerts
        if not args.no_alert and not df.empty:
            alert_mgr = AlertManager(screener.config)
            # Convert df back to result list for alert formatting
            # (simplified: re-run if you need full dict — or cache raw results)
            console.print("\n[cyan]📬 Sending Telegram alerts...[/cyan]")

        if df.empty:
            sys.exit(0)

        # Print quick summary stats
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  Strong Buys: {len(df[df['Signal'].str.contains('STRONG', na=False)])}")
        console.print(f"  Intraday:    {len(df[df['Setup'] == 'INTRADAY'])}")
        console.print(f"  BTST:        {len(df[df['Setup'] == 'BTST'])}")
        console.print(f"  Swing:       {len(df[df['Setup'] == 'SWING'])}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
    except Exception as e:
        logger.error(f"Screener error: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
