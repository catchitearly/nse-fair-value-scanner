"""
scanner.py
-----------
NSE Fair Value Scanner — main entry point.

LIVE SCAN:
    python scanner.py

BACKTEST with monthly returns:
    python scanner.py --scan-date 2023-01-01
    python scanner.py --scan-date 2022-06-15

PARALLEL (faster):
    python scanner.py --workers 4
    python scanner.py --scan-date 2023-01-01 --workers 4

ALL OPTIONS:
    --scan-date   YYYY-MM-DD   Backtest date (blank = live)
    --workers     N            Parallel workers (default 1, max recommended 4)
    --mcap-min    N            Min market cap in Rs Crore (default 500)
    --mcap-max    N            Max market cap in Rs Crore (default 10000)
    --tickers     A B C        Specific NSE tickers (no .NS suffix)
    --limit       N            First N tickers only (for quick tests)
    --delay       N            Seconds between API calls (default 0.4)
    --output      DIR          Output folder (default output/)
"""

import argparse
import logging
import os
import sys
from datetime import datetime, date

from data_fetcher import (
    get_nse_stock_list, fetch_all_stocks,
    parse_scan_date, enrich_with_monthly_returns,
    check_results_season, RETURN_INTERVALS
)
from valuation_engine import run_valuation
from report_generator import generate_report

os.makedirs("logs",   exist_ok=True)
os.makedirs("output", exist_ok=True)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"logs/scan_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
        ),
    ]
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="NSE Fair Value Scanner",
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--scan-date", dest="scan_date", type=str, default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Backtest date — scan AS OF this date.\n"
            "  Leave blank for TODAY's live data.\n"
            "  Example: --scan-date 2023-01-01\n"
            "  Tip: Avoid results-season grey zones (Oct 1-14, Jan 1-14,\n"
            "       Apr 1-14, Jul 1-14) for cleaner backtest data."
        )
    )
    p.add_argument(
        "--workers", type=int, default=1, metavar="N",
        help=(
            "Parallel workers for faster fetching.\n"
            "  1  = sequential, safest (default)\n"
            "  2  = 2x faster\n"
            "  4  = 4x faster (recommended max)\n"
            "  8  = fastest but risk of yfinance rate-limit\n"
            "  Example: --workers 4"
        )
    )
    p.add_argument(
        "--mcap-min", dest="mcap_min", type=float, default=500,
        metavar="CRORE",
        help="Min market cap in Rs Crore (default: 500)"
    )
    p.add_argument(
        "--mcap-max", dest="mcap_max", type=float, default=10000,
        metavar="CRORE",
        help="Max market cap in Rs Crore (default: 10000)"
    )
    p.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Specific NSE tickers (no .NS). E.g.: --tickers INFY TCS WIPRO"
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="First N tickers only. E.g.: --limit 50  (good for quick tests)"
    )
    p.add_argument(
        "--delay", type=float, default=0.4,
        help="Seconds between API calls (default 0.4). Reduce with more workers."
    )
    p.add_argument(
        "--output", type=str, default="output",
        help="Output directory (default: output/)"
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args        = parse_args()
    scan_date   = parse_scan_date(args.scan_date)
    is_backtest = scan_date is not None
    workers     = max(1, args.workers)

    # ── Header ────────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("  NSE FAIR VALUE SCANNER")
    logger.info(f"  Run time  : {datetime.now().strftime('%d %b %Y  %H:%M:%S')}")
    logger.info(f"  Mode      : {'BACKTEST  (scan date: ' + str(scan_date) + ')' if is_backtest else 'LIVE (today)'}")
    logger.info(f"  MCap range: Rs{args.mcap_min:,.0f} cr  to  Rs{args.mcap_max:,.0f} cr")
    logger.info(f"  Workers   : {workers} ({'parallel' if workers > 1 else 'sequential'})")
    logger.info("=" * 65)

    # ── Results season warning ────────────────────────────────────────────────
    if is_backtest:
        is_grey, warning_msg = check_results_season(scan_date)
        if is_grey:
            logger.warning(warning_msg)
            # Don't exit — user may want to proceed anyway

    # ── 1. Ticker list ────────────────────────────────────────────────────────
    if args.tickers:
        tickers = [t.upper() + ".NS" for t in args.tickers]
        logger.info(f"Custom tickers: {tickers}")
    else:
        tickers = get_nse_stock_list()

    if args.limit:
        tickers = tickers[:args.limit]
        logger.info(f"Limiting to first {args.limit} tickers")

    # ── 2. Fetch financial data ───────────────────────────────────────────────
    raw_df = fetch_all_stocks(
        tickers,
        scan_date = scan_date,
        delay     = args.delay,
        mcap_min  = args.mcap_min,
        mcap_max  = args.mcap_max,
        workers   = workers,
    )

    if raw_df.empty:
        logger.error(
            "No stocks matched the market-cap filter or had sufficient yfinance data.\n"
            "  Possible reasons:\n"
            "    1. --limit is too small and all first-N tickers are SME/obscure stocks\n"
            "       with no yfinance data. Try --limit 100 or remove --limit entirely.\n"
            "    2. Market cap range is too narrow. Check --mcap-min / --mcap-max.\n"
            "    3. yfinance is rate-limiting. Wait a few minutes and retry.\n"
            "  Tip: Run with --tickers INFY TCS HCLTECH first to verify connectivity."
        )
        sys.exit(1)

    logger.info(f"Stocks passing market-cap filter: {len(raw_df)}")

    # ── 3. Valuation models ───────────────────────────────────────────────────
    result_df = run_valuation(raw_df)

    # ── 4. Monthly return enrichment (backtest only) ──────────────────────────
    if is_backtest:
        logger.info("Fetching monthly price snapshots for return calculation ...")
        result_df = enrich_with_monthly_returns(
            result_df, scan_date, workers=workers
        )
        _print_backtest_accuracy(result_df, scan_date)

    # ── 5. Generate report ────────────────────────────────────────────────────
    excel_path = generate_report(
        result_df,
        output_dir  = args.output,
        scan_date   = scan_date,
        is_backtest = is_backtest,
    )

    # ── 6. Final summary ──────────────────────────────────────────────────────
    logger.info("\n" + "─" * 65)
    logger.info("  SCAN COMPLETE")
    for v in ["Strong Buy", "Buy", "Hold", "Avoid", "Strong Avoid", "Insufficient Data"]:
        matches = result_df[result_df["verdict"].str.contains(v, na=False)]
        if len(matches):
            logger.info(f"  {v:22s}: {len(matches)} stocks")
    logger.info(f"\n  Excel report -> {excel_path}")
    logger.info("─" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ACCURACY CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def _print_backtest_accuracy(df, scan_date):
    """
    Prints a monthly return table for each verdict category.
    Shows return at 30,60,90,...,365 days for Strong Buy vs Buy vs Hold etc.
    """
    # Only show intervals that have data
    available = [d for d in RETURN_INTERVALS if f"return_{d}d" in df.columns
                 and df[f"return_{d}d"].notna().any()]

    if not available:
        logger.info("No forward return data available yet (future dates).")
        return

    verdicts = ["Strong Buy", "Buy", "Hold", "Avoid", "Strong Avoid"]

    logger.info("\n" + "═" * 90)
    logger.info(f"  BACKTEST ACCURACY  |  Scan date: {scan_date}")
    logger.info(f"  Average return % by verdict at each time interval")
    logger.info("═" * 90)

    # Header row
    header = f"  {'Verdict':<20} {'N':>4}"
    for d in available:
        header += f"  {'+'+str(d)+'d':>7}"
    logger.info(header)
    logger.info("  " + "─" * (24 + len(available) * 9))

    for v in verdicts:
        subset = df[df["verdict"].str.contains(v, na=False)]
        if subset.empty:
            continue
        row = f"  {v:<20} {len(subset):>4}"
        for d in available:
            col  = f"return_{d}d"
            vals = subset[col].dropna()
            if vals.empty:
                row += f"  {'N/A':>7}"
            else:
                avg = vals.mean()
                row += f"  {avg:>+6.1f}%"
        logger.info(row)

    logger.info("  " + "─" * (24 + len(available) * 9))

    # % positive row
    logger.info(f"\n  % of stocks with POSITIVE return:")
    logger.info(f"  {'Verdict':<20} {'N':>4}" + "".join(
        [f"  {'+'+str(d)+'d':>7}" for d in available]
    ))
    logger.info("  " + "─" * (24 + len(available) * 9))

    for v in verdicts:
        subset = df[df["verdict"].str.contains(v, na=False)]
        if subset.empty:
            continue
        row = f"  {v:<20} {len(subset):>4}"
        for d in available:
            col  = f"return_{d}d"
            vals = subset[col].dropna()
            if vals.empty:
                row += f"  {'N/A':>7}"
            else:
                pct_pos = (vals > 0).mean() * 100
                row += f"  {pct_pos:>6.1f}%"
        logger.info(row)

    logger.info("═" * 90)


if __name__ == "__main__":
    main()
