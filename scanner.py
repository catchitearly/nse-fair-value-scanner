"""
scanner.py
-----------
NSE Fair Value Scanner — reads stock universe from stocks.csv.

USAGE:
    python scanner.py                            # live scan, all stocks in stocks.csv
    python scanner.py --scan-date 2023-06-01     # backtest as of that date
    python scanner.py --workers 4                # 4x faster parallel fetch
    python scanner.py --limit 20                 # first 20 rows of stocks.csv (quick test)
    python scanner.py --csv my_stocks.csv        # use a different CSV file

stocks.csv format:
    ticker,name,notes
    INFY,Infosys,IT services
    TCS,Tata Consultancy Services,IT services
    (place this file in the same folder as scanner.py)
"""

import argparse
import logging
import os
import sys
from datetime import datetime

from data_fetcher import (
    load_tickers_from_csv, fetch_all_stocks,
    parse_scan_date, enrich_with_monthly_returns,
    check_results_season, RETURN_INTERVALS,
    setup_yfinance_session,
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
# ARGUMENTS
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="NSE Fair Value Scanner — stock universe from stocks.csv",
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--csv", type=str, default="stocks.csv", metavar="FILE",
        help=(
            "Path to your stock universe CSV file.\n"
            "  Default: stocks.csv (in the same folder as scanner.py)\n"
            "  Must have a 'ticker' column. Optional: name, notes, sector.\n"
            "  Example: --csv my_watchlist.csv"
        )
    )
    p.add_argument(
        "--scan-date", dest="scan_date", type=str, default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Backtest date — scan AS OF this date.\n"
            "  Leave blank for TODAY's live data.\n"
            "  Example: --scan-date 2023-06-01\n"
            "  Safe zones: Nov15-Dec31 | Feb15-Mar31 | May15-Jun30 | Aug15-Sep30"
        )
    )
    p.add_argument(
        "--workers", type=int, default=1, metavar="N",
        help=(
            "Parallel fetch workers (faster).\n"
            "  1 = sequential, safest (default)\n"
            "  2 = ~2x faster\n"
            "  4 = ~4x faster (recommended max)\n"
            "  8 = fastest, risk of yfinance rate-limit"
        )
    )
    p.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help=(
            "Scan only first N rows of the CSV.\n"
            "  Useful for quick tests. E.g.: --limit 10"
        )
    )
    p.add_argument(
        "--delay", type=float, default=0.4, metavar="SEC",
        help="Seconds between API calls (default: 0.4)"
    )
    p.add_argument(
        "--output", type=str, default="output", metavar="DIR",
        help="Output folder for Excel report (default: output/)"
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

    # ── Setup yfinance session (curl_cffi Chrome impersonation) ─────────────
    setup_yfinance_session()

    # ── Header ────────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("  NSE FAIR VALUE SCANNER")
    logger.info(f"  Run time  : {datetime.now().strftime('%d %b %Y  %H:%M:%S')}")
    logger.info(f"  Mode      : {'BACKTEST  (scan date: ' + str(scan_date) + ')' if is_backtest else 'LIVE (today)'}")
    logger.info(f"  Stock CSV : {args.csv}")
    logger.info(f"  Workers   : {workers} ({'parallel' if workers > 1 else 'sequential'})")
    logger.info("=" * 65)

    # ── Results season warning ────────────────────────────────────────────────
    if is_backtest:
        is_grey, warning_msg = check_results_season(scan_date)
        if is_grey:
            logger.warning(warning_msg)

    # ── 1. Load stock universe from CSV ───────────────────────────────────────
    tickers, meta_df = load_tickers_from_csv(args.csv)

    if not tickers:
        logger.error(
            f"No tickers loaded from '{args.csv}'. Cannot proceed.\n"
            f"  Make sure the file exists and has a 'ticker' column."
        )
        sys.exit(1)

    if args.limit:
        tickers  = tickers[:args.limit]
        meta_df  = meta_df.head(args.limit)
        logger.info(f"Limiting to first {args.limit} stocks from CSV")

    logger.info(f"Stock universe: {len(tickers)} stocks")

    # ── 2. Fetch data ─────────────────────────────────────────────────────────
    raw_df = fetch_all_stocks(
        tickers,
        scan_date   = scan_date,
        delay       = args.delay,
        workers     = workers,
        csv_meta_df = meta_df,
    )

    if raw_df.empty:
        logger.error(
            "No data fetched for any stock.\n"
            "  Possible reasons:\n"
            "    1. yfinance is rate-limiting — wait a few minutes and retry.\n"
            "    2. Tickers in your CSV are invalid or not on NSE.\n"
            "    3. No internet connectivity.\n"
            "  Tip: Test with --limit 5 --workers 1 first."
        )
        sys.exit(1)

    logger.info(f"Data fetched for {len(raw_df)} / {len(tickers)} stocks")

    # ── 3. Valuation ──────────────────────────────────────────────────────────
    result_df = run_valuation(raw_df)

    # ── 4. Monthly returns (backtest only) ────────────────────────────────────
    if is_backtest:
        logger.info("Fetching monthly price snapshots for return analysis ...")
        result_df = enrich_with_monthly_returns(
            result_df, scan_date, workers=workers
        )
        _print_backtest_accuracy(result_df, scan_date)

    # ── 5. Report ─────────────────────────────────────────────────────────────
    excel_path = generate_report(
        result_df,
        output_dir  = args.output,
        scan_date   = scan_date,
        is_backtest = is_backtest,
    )

    # ── 6. Summary ────────────────────────────────────────────────────────────
    logger.info("\n" + "─" * 65)
    logger.info("  SCAN COMPLETE")
    for v in ["Strong Buy", "Buy", "Hold", "Avoid", "Strong Avoid", "Insufficient Data"]:
        matches = result_df[result_df["verdict"].str.contains(v, na=False)]
        if len(matches):
            logger.info(f"  {v:22s}: {len(matches)} stocks")
    logger.info(f"\n  Excel report → {excel_path}")
    logger.info("─" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ACCURACY CONSOLE PRINT
# ══════════════════════════════════════════════════════════════════════════════

def _print_backtest_accuracy(df, scan_date):
    available = [
        d for d in RETURN_INTERVALS
        if f"return_{d}d" in df.columns and df[f"return_{d}d"].notna().any()
    ]
    if not available:
        logger.info("No forward return data available (all intervals in future).")
        return

    verdicts = ["Strong Buy", "Buy", "Hold", "Avoid", "Strong Avoid"]

    logger.info("\n" + "═" * 90)
    logger.info(f"  BACKTEST ACCURACY  |  Scan date: {scan_date}")
    logger.info("  Average return % by verdict at each time interval")
    logger.info("═" * 90)

    hdr = f"  {'Verdict':<20} {'N':>4}" + "".join(f"  {'+'+str(d)+'d':>7}" for d in available)
    logger.info(hdr)
    logger.info("  " + "─" * (26 + len(available) * 9))

    for v in verdicts:
        subset = df[df["verdict"].str.contains(v, na=False)]
        if subset.empty:
            continue
        row = f"  {v:<20} {len(subset):>4}"
        for d in available:
            vals = subset[f"return_{d}d"].dropna()
            row += f"  {vals.mean():>+6.1f}%" if len(vals) else f"  {'N/A':>7}"
        logger.info(row)

    logger.info("\n  % of stocks with POSITIVE return:")
    logger.info(hdr)
    logger.info("  " + "─" * (26 + len(available) * 9))

    for v in verdicts:
        subset = df[df["verdict"].str.contains(v, na=False)]
        if subset.empty:
            continue
        row = f"  {v:<20} {len(subset):>4}"
        for d in available:
            vals = subset[f"return_{d}d"].dropna()
            row += f"  {(vals > 0).mean()*100:>6.1f}%" if len(vals) else f"  {'N/A':>7}"
        logger.info(row)

    logger.info("═" * 90)


if __name__ == "__main__":
    main()
