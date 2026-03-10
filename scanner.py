"""
scanner.py
-----------
Main entry point for the NSE Fair Value Scanner.

LIVE SCAN (today's data):
    python scanner.py

BACKTEST (historical date):
    python scanner.py --scan-date 2023-01-01
    python scanner.py --scan-date 2022-06-15 --forward-days 365

OTHER OPTIONS:
    python scanner.py --tickers INFY TCS HCLTECH   # specific stocks only
    python scanner.py --limit 50                    # first N tickers (test)
    python scanner.py --delay 0.5                   # API call delay (sec)
    python scanner.py --output results/             # custom output folder
"""

import argparse
import logging
import os
import sys
from datetime import datetime, date

from data_fetcher     import get_nse_stock_list, fetch_all_stocks, parse_scan_date, enrich_with_forward_price
from valuation_engine import run_valuation
from report_generator import generate_report

os.makedirs("logs",   exist_ok=True)
os.makedirs("output", exist_ok=True)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/scan_{datetime.now().strftime('%Y%m%d_%H%M')}.log"),
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="NSE Mid-Cap Fair Value Scanner (Rs500 cr - Rs10,000 cr)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--scan-date", dest="scan_date", type=str, default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Date to run the scan AS OF (for backtesting).\n"
            "  Example: --scan-date 2023-01-01\n"
            "  Leave blank to use TODAY'S live data.\n"
            "  Note: yfinance provides ~5 years of historical financials."
        )
    )
    p.add_argument(
        "--forward-days", dest="forward_days", type=int, default=365,
        metavar="N",
        help=(
            "Days after scan_date to fetch actual price for return calculation.\n"
            "  Default: 365 (1 year forward return).\n"
            "  Only applies in backtest mode AND if that date is already past.\n"
            "  Example: --forward-days 180"
        )
    )
    p.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Specific NSE tickers (no .NS suffix). E.g.: INFY TCS WIPRO"
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of tickers (useful for quick tests). E.g.: --limit 50"
    )
    p.add_argument(
        "--mcap-min", dest="mcap_min", type=float, default=500,
        metavar="CRORE",
        help="Minimum market cap in Rs Crore (default: 500)"
    )
    p.add_argument(
        "--mcap-max", dest="mcap_max", type=float, default=10000,
        metavar="CRORE",
        help="Maximum market cap in Rs Crore (default: 10000)"
    )
    p.add_argument(
        "--delay", type=float, default=0.4,
        help="Delay in seconds between API calls (default: 0.4)"
    )
    p.add_argument(
        "--output", type=str, default="output",
        help="Output directory for Excel report (default: output/)"
    )
    return p.parse_args()


def main():
    args      = parse_args()
    scan_date = parse_scan_date(args.scan_date)   # None = live mode
    is_backtest = scan_date is not None

    # ── Header ────────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("  NSE FAIR VALUE SCANNER")
    logger.info(f"  Run time  : {datetime.now().strftime('%d %b %Y  %H:%M:%S')}")
    if is_backtest:
        logger.info(f"  Mode      : BACKTEST  (scan date: {scan_date})")
        logger.info(f"  Fwd window: {args.forward_days} days after scan date")
    else:
        logger.info("  Mode      : LIVE  (using today's data)")
    logger.info(f"  MCap range: Rs{args.mcap_min} cr  —  Rs{args.mcap_max} cr")
    logger.info("=" * 65)

    # ── 1. Build ticker list ──────────────────────────────────────────────────
    if args.tickers:
        tickers = [t.upper() + ".NS" for t in args.tickers]
        logger.info(f"Custom tickers: {tickers}")
    else:
        tickers = get_nse_stock_list()

    if args.limit:
        tickers = tickers[:args.limit]
        logger.info(f"Limiting to first {args.limit} tickers")

    # ── 2. Fetch data (live or historical) ───────────────────────────────────
    raw_df = fetch_all_stocks(tickers, scan_date=scan_date, delay=args.delay,
                               mcap_min=args.mcap_min, mcap_max=args.mcap_max)

    if raw_df.empty:
        logger.error("No stocks matched the market-cap filter. Exiting.")
        sys.exit(1)

    logger.info(f"Stocks passing market-cap filter: {len(raw_df)}")

    # ── 3. Run valuation models ───────────────────────────────────────────────
    result_df = run_valuation(raw_df)

    # ── 4. Backtest: fetch forward prices & compute actual returns ────────────
    if is_backtest:
        logger.info(f"Enriching with forward prices ({args.forward_days} days out) ...")
        result_df = enrich_with_forward_price(
            result_df, scan_date, forward_days=args.forward_days
        )
        # Log backtest accuracy summary if returns are available
        if result_df["actual_return_pct"].notna().any():
            _print_backtest_accuracy(result_df)

    # ── 5. Generate report ────────────────────────────────────────────────────
    excel_path = generate_report(
        result_df,
        output_dir  = args.output,
        scan_date   = scan_date,
        is_backtest = is_backtest,
        forward_days= args.forward_days,
    )

    # ── 6. Summary ────────────────────────────────────────────────────────────
    logger.info("\n-- SCAN COMPLETE " + "-" * 48)
    for v in ["Strong Buy", "Buy", "Hold", "Avoid", "Strong Avoid", "Insufficient Data"]:
        # match emoji prefix too
        matches = result_df[result_df["verdict"].str.contains(v, na=False)]
        if len(matches):
            logger.info(f"  {v}: {len(matches)} stocks")
    logger.info(f"\n  Excel report -> {excel_path}")
    logger.info("-" * 65)


def _print_backtest_accuracy(df):
    """Prints a quick accuracy table: how did each verdict actually perform?"""
    logger.info("\n-- BACKTEST ACCURACY " + "-" * 44)
    logger.info("  How each verdict category actually performed:")
    logger.info(f"  {'Verdict':<22} {'Count':>5}  {'Avg Return':>12}  {'% Positive':>12}")
    logger.info(f"  {'-'*22}  {'-'*5}  {'-'*12}  {'-'*12}")

    for v in ["Strong Buy", "Buy", "Hold", "Avoid", "Strong Avoid"]:
        subset = df[df["verdict"].str.contains(v, na=False) & df["actual_return_pct"].notna()]
        if subset.empty:
            continue
        avg_ret  = subset["actual_return_pct"].mean()
        pct_pos  = (subset["actual_return_pct"] > 0).mean() * 100
        logger.info(f"  {v:<22} {len(subset):>5}  {avg_ret:>+11.1f}%  {pct_pos:>11.1f}%")

    logger.info("-" * 65)


if __name__ == "__main__":
    main()
