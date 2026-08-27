#!/usr/bin/env python3
"""
main.py  (v2)
-------------
FIXES vs v1:
  * Telegram alerts are ACTUALLY SENT. v1 constructed an AlertManager, printed
    "Sending Telegram alerts...", and never called send_telegram — and if it
    had, it would have crashed on missing keys.
  * The results table prints columns that Scorer.to_dataframe actually emits.
    v1 asked for LTP / RSI / Vol_Spike / Delivery% which were never produced,
    so every one of those columns rendered as an em-dash.
  * Summary counts match the real signal vocabulary (v1 looked for the string
    "STRONG", which the scorer never emits).
  * Non-zero exit only on real failure, so cron/GitHub Actions can distinguish
    "no setups today" from "the screener broke".

Usage:
  python main.py --mode swing
  python main.py --mode intraday --workers 8
  python main.py --symbols TATAPOWER IRFC --no-alert
  python main.py --min-score 60 --top 30
"""

import argparse
import sys
import traceback

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

from screener.screener import IndiaStockScreener
from screener.alerts import AlertManager

console = Console()

SIGNAL_STYLE = {
    "BUY": ("green", "🟢"),
    "WATCH": ("yellow", "🟡"),
    "WATCH_NO_ENTRY": ("yellow", "🟠"),
    "WEAK": ("white", "⚪"),
    "IGNORE": ("dim", "·"),
    "REJECTED": ("red", "🔴"),
    "INSUFFICIENT_DATA": ("dim", "?"),
}


def print_banner():
    console.print("[bold green]\n  India Smart Stock Screener  v2  "
                  "[dim]— calibrated 0-100 scoring[/dim]\n[/bold green]")


def fmt(v, suffix="", dash="—", nd=2):
    if v is None or v == "":
        return dash
    try:
        return f"{float(v):.{nd}f}{suffix}"
    except (TypeError, ValueError):
        return str(v)


def print_results_table(df, regime, top=20):
    if df is None or df.empty:
        console.print("[yellow]No symbols scored.[/yellow]")
        return

    console.print(
        f"[dim]Market regime:[/dim] [bold]{regime.get('regime', 'unknown').upper()}[/bold]"
        f"  [dim]threshold {regime.get('threshold_delta', 0):+d}[/dim]"
        + (f"  [dim]({'; '.join(regime.get('reasons', []))})[/dim]"
           if regime.get("reasons") else "")
    )

    table = Table(title=f"Top {min(top, len(df))} of {len(df)} scored",
                  box=box.SIMPLE_HEAVY, header_style="bold magenta")

    for col, w, just in [
        ("#", 3, "right"), ("Symbol", 13, "left"), ("LTP", 9, "right"),
        ("Score", 6, "right"), ("Pct", 5, "right"), ("Signal", 15, "left"),
        ("Setup", 9, "left"), ("RSI", 5, "right"), ("ATR%", 5, "right"),
        ("Vol×", 5, "right"), ("Del%", 5, "right"), ("Entry", 9, "right"),
        ("Target", 9, "right"), ("SL", 9, "right"), ("R/R", 5, "right"),
        ("Live", 4, "right"),
    ]:
        table.add_column(col, width=w, justify=just)

    for i, row in df.head(top).iterrows():
        sig = row.get("Signal", "")
        style, icon = SIGNAL_STYLE.get(sig, ("white", " "))
        score = row.get("Score") or 0
        sc_col = "bold green" if score >= 70 else "yellow" if score >= 55 else "white"

        table.add_row(
            str(i + 1),
            str(row.get("Symbol", "")),
            fmt(row.get("LTP")),
            f"[{sc_col}]{fmt(score, nd=1)}[/{sc_col}]",
            fmt(row.get("Pctile"), nd=0),
            f"[{style}]{icon} {sig}[/{style}]",
            str(row.get("Setup", "")),
            fmt(row.get("RSI"), nd=0),
            fmt(row.get("ATR%"), nd=1),
            fmt(row.get("Vol_Spike"), nd=1),
            fmt(row.get("Delivery%"), nd=0),
            fmt(row.get("Entry")),
            fmt(row.get("Target")),
            fmt(row.get("SL")),
            fmt(row.get("RR"), nd=1),
            str(row.get("Live", "")),
        )

    console.print(table)


def print_summary(df):
    if df is None or df.empty:
        return
    console.print("\n[bold]Summary[/bold]")
    counts = df["Signal"].value_counts().to_dict()
    for sig in ("BUY", "WATCH", "WATCH_NO_ENTRY", "WEAK", "IGNORE",
                "REJECTED", "INSUFFICIENT_DATA"):
        if counts.get(sig):
            style, icon = SIGNAL_STYLE.get(sig, ("white", " "))
            console.print(f"  [{style}]{icon} {sig:<20}[/{style}] {counts[sig]}")

    setups = df[df["Signal"].isin(["BUY", "WATCH"])]["Setup"].value_counts().to_dict()
    if setups:
        console.print("  [dim]actionable by setup:[/dim] "
                      + ", ".join(f"{k} {v}" for k, v in setups.items()))

    low = int((df["Live"] < 3).sum()) if "Live" in df.columns else 0
    if low:
        console.print(f"  [dim]{low} symbol(s) scored on fewer than 3 live components — "
                      f"check your NSE connectivity before trusting those.[/dim]")


def main():
    print_banner()

    p = argparse.ArgumentParser(description="India Smart Stock Screener")
    p.add_argument("--mode", default="all", choices=["intraday", "btst", "swing", "all"])
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--workers", default=6, type=int)
    p.add_argument("--no-alert", action="store_true", help="Skip Telegram alerts")
    p.add_argument("--symbols", nargs="+", help="Override universe")
    p.add_argument("--universe", help="Override configured universe")
    p.add_argument("--min-score", type=float, help="Display filter only (nothing is dropped from the DB)")
    p.add_argument("--top", type=int, default=20, help="Rows to print")
    args = p.parse_args()

    try:
        s = IndiaStockScreener(config_path=args.config)

        if args.symbols:
            s.config["screening"]["universe"] = "custom"
            s.config["screening"]["custom_symbols"] = args.symbols
        elif args.universe:
            s.config["screening"]["universe"] = args.universe

        df = s.run(mode=args.mode, max_workers=args.workers)

        if df.empty:
            console.print("[yellow]No symbols scored — check the log for data-fetch "
                          "failures before assuming the market is quiet.[/yellow]")
            sys.exit(0)

        display = df
        if args.min_score is not None:
            display = df[df["Score"] >= args.min_score]

        print_results_table(display, s.regime, top=args.top)
        print_summary(df)

        console.print(f"\n[dim]Run {s.run_id} — "
                      f"{len(s.last_results)} rows written to the signals table.[/dim]")

        # ---- alerts: actually sent now ----
        if not args.no_alert:
            sent = AlertManager(s.config).send_telegram(s.last_results, regime=s.regime)
            if sent:
                console.print(f"[cyan]{sent} Telegram alert(s) sent.[/cyan]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Screener error: {e}\n{traceback.format_exc()}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
