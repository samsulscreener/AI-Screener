#!/bin/bash
# =============================================================
#  scripts/run_screener.sh
#  Cron-ready wrapper. Add to crontab:
#
#  # Intraday every 30min during market hours (IST = UTC+5:30)
#  20,50 3-9 * * 1-5  /path/to/run_screener.sh intraday >> /var/log/screener.log 2>&1
#
#  # BTST at 3:10 PM IST = 9:40 AM UTC
#  40 9 * * 1-5        /path/to/run_screener.sh btst >> /var/log/screener.log 2>&1
#
#  # Swing every Friday 4:00 PM IST = 10:30 AM UTC
#  30 10 * * 5         /path/to/run_screener.sh swing >> /var/log/screener.log 2>&1
# =============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODE="${1:-all}"

echo "============================================"
echo "  India Stock Screener — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Mode: $MODE"
echo "============================================"

cd "$PROJECT_DIR"

# Activate virtual environment if present
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Load .env
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

python main.py --mode "$MODE" --workers 6
echo "Done."
