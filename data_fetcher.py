"""
data_fetcher.py
---------------
Fetches NSE stock data from yfinance.

Stock universe: reads from stocks.csv in the repo root.
  CSV must have a 'ticker' column (NSE symbol without .NS suffix).
  Optional columns: name, notes, sector — used as metadata fallback.

No market-cap filtering — you control the universe via stocks.csv.

Features:
  - Live mode    : today's price + latest financials
  - Backtest mode: historical price on scan_date + filings before scan_date
  - Parallel fetch via ThreadPoolExecutor (--workers)
  - Monthly return intervals: 30,60,...,365 days
  - Results season warning for grey-zone scan dates
"""

import time
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)

CR_TO_INR = 1e7
TODAY     = date.today()

# Monthly return checkpoints (days from scan_date)
RETURN_INTERVALS = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365]

# Results season grey zones: (month, day_start, day_end, description)
GREY_ZONES = [
    (10,  1, 14, "Q1 FY results (Apr-Jun) — some companies not yet filed"),
    (11,  1, 14, "Q1 FY results (Apr-Jun) — stragglers still filing"),
    ( 1,  1, 14, "Q2 FY results (Jul-Sep) — some companies not yet filed"),
    ( 2,  1, 14, "Q2 FY results (Jul-Sep) — stragglers still filing"),
    ( 4,  1, 14, "Q3 FY results (Oct-Dec) — some companies not yet filed"),
    ( 5,  1, 14, "Q3 FY results (Oct-Dec) — stragglers still filing"),
    ( 7,  1, 14, "Q4/Full-Year results (Jan-Mar) — some companies not yet filed"),
    ( 8,  1, 14, "Q4/Full-Year results (Jan-Mar) — stragglers still filing"),
]


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS SEASON WARNING
# ══════════════════════════════════════════════════════════════════════════════

def check_results_season(scan_date):
    """
    Returns (is_grey_zone: bool, warning_message: str).
    Call before fetching to warn user about mixed financial data risk.
    """
    if scan_date is None:
        return False, ""
    m, d = scan_date.month, scan_date.day
    for gz_month, gz_start, gz_end, gz_reason in GREY_ZONES:
        if m == gz_month and gz_start <= d <= gz_end:
            msg = (
                f"\n{'=' * 65}\n"
                f"  ⚠️  RESULTS SEASON WARNING\n"
                f"{'=' * 65}\n"
                f"  scan_date {scan_date} is in a quarterly results grey zone.\n"
                f"  Reason : {gz_reason}\n"
                f"\n"
                f"  Risk   : Some companies have filed new quarterly results\n"
                f"           while others have not. Your scanner will MIX old\n"
                f"           and new financial data — comparisons unreliable.\n"
                f"\n"
                f"  RECOMMENDED SAFE SCAN DATES:\n"
                f"    Nov 15 – Dec 31  → Q1 (Apr-Jun) results fully settled\n"
                f"    Feb 15 – Mar 31  → Q2 (Jul-Sep) results fully settled\n"
                f"    May 15 – Jun 30  → Q3 (Oct-Dec) results fully settled\n"
                f"    Aug 15 – Sep 30  → Q4 (Jan-Mar) results fully settled\n"
                f"{'=' * 65}\n"
            )
            return True, msg
    return False, ""


# ══════════════════════════════════════════════════════════════════════════════
# STOCK LIST — READ FROM stocks.csv
# ══════════════════════════════════════════════════════════════════════════════

def load_tickers_from_csv(csv_path="stocks.csv"):
    """
    Reads the stock universe from a CSV file.

    Supported formats (auto-detected):
      1. Tab-separated with headers: "Stock Name" and "Symbol"
           Stock Name\tSymbol
           EMS Ltd\tEMSLIMITED
      2. Comma-separated with headers: "ticker" and optionally "name", "notes", "sector"
           ticker,name,notes
           INFY,Infosys,IT services

    The 'Symbol' / 'ticker' column is the NSE symbol (without .NS suffix).
    The 'Stock Name' / 'name' column is used as the display name.

    Returns:
      tickers  : list of 'TICKER.NS' strings for yfinance
      meta_df  : DataFrame with standardised columns: ticker, name (+ any extras)
    """
    try:
        # ── Auto-detect separator ──────────────────────────────────────────────
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            first_line = f.readline()

        sep = "\t" if "\t" in first_line else ","
        df  = pd.read_csv(csv_path, sep=sep, engine="python")

        # Strip whitespace from all column names and values
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()

        # ── Normalise column names to standard internal names ──────────────────
        # Map known variants → standard names
        col_map = {}
        for col in df.columns:
            low = col.lower()
            if low in ("symbol", "ticker", "nse symbol", "nse_symbol", "scrip"):
                col_map[col] = "ticker"
            elif low in ("stock name", "stock_name", "name", "company", "company name",
                         "company_name", "scrip name", "scrip_name"):
                col_map[col] = "name"
            elif low in ("sector", "industry"):
                col_map[col] = "sector"
            elif low in ("notes", "note", "remarks"):
                col_map[col] = "notes"

        df.rename(columns=col_map, inplace=True)

        if "ticker" not in df.columns:
            logger.error(
                f"Could not find a Symbol/ticker column in '{csv_path}'.\n"
                f"  Columns found: {list(df.columns)}\n"
                f"  Supported column names: Symbol, ticker, NSE Symbol, Scrip\n"
                f"  Supported separators  : tab (\\t) or comma (,)"
            )
            return [], pd.DataFrame()

        # ── Clean ticker column ────────────────────────────────────────────────
        df["ticker"] = (df["ticker"]
                        .astype(str)
                        .str.strip()
                        .str.upper()
                        .str.replace(r"\.NS$", "", regex=True))  # strip .NS if already present

        # Drop blank / NaN tickers
        df = df[df["ticker"].notna() & (df["ticker"] != "") & (df["ticker"] != "NAN")]
        df = df.drop_duplicates(subset="ticker").reset_index(drop=True)

        # Ensure a 'name' column exists (fallback to ticker if missing)
        if "name" not in df.columns:
            df["name"] = df["ticker"]

        tickers = [f"{t}.NS" for t in df["ticker"].tolist()]
        logger.info(
            f"Loaded {len(tickers)} tickers from '{csv_path}' "
            f"(separator: {'TAB' if sep == chr(9) else 'COMMA'})"
        )
        return tickers, df

    except FileNotFoundError:
        logger.error(
            f"File not found: '{csv_path}'\n"
            f"  Place your CSV in the repo root and pass its name via --csv.\n"
            f"  Supported formats:\n"
            f"    Tab-separated : Stock Name\\tSymbol\n"
            f"    Comma-separated: ticker,name,notes"
        )
        return [], pd.DataFrame()
    except Exception as e:
        logger.error(f"Error reading '{csv_path}': {e}")
        return [], pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# HISTORICAL DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_historical_price(tk, target_date):
    """Closing price on or before target_date (looks back up to 10 calendar days)."""
    start = target_date - timedelta(days=10)
    end   = target_date + timedelta(days=1)
    try:
        hist = tk.history(start=str(start), end=str(end), auto_adjust=True)
        if hist.empty:
            return None
        hist.index = hist.index.date
        valid = hist[hist.index <= target_date]
        return float(valid["Close"].iloc[-1]) if not valid.empty else None
    except Exception:
        return None


def _get_historical_52w(tk, scan_date):
    """52-week high and low ending on scan_date."""
    start = scan_date - timedelta(days=370)
    end   = scan_date + timedelta(days=1)
    try:
        hist = tk.history(start=str(start), end=str(end), auto_adjust=True)
        if hist.empty:
            return None, None
        hist.index = hist.index.date
        valid = hist[hist.index <= scan_date]
        if valid.empty:
            return None, None
        return float(valid["High"].max()), float(valid["Low"].min())
    except Exception:
        return None, None


def _latest_col_before(df_fin, scan_date):
    """Most recent column (as Series) from a yfinance financials DataFrame <= scan_date."""
    if df_fin is None or df_fin.empty:
        return None
    try:
        dates = pd.to_datetime(df_fin.columns).date
        valid = [(d, col) for d, col in zip(dates, df_fin.columns) if d <= scan_date]
        if not valid:
            return None
        _, best_col = max(valid, key=lambda x: x[0])
        return df_fin[best_col]
    except Exception:
        return None


def _val(series, key):
    """Safely extract a float from a pandas Series by partial key match."""
    if series is None:
        return None
    try:
        matches = [k for k in series.index if key.lower() in str(k).lower()]
        if matches:
            v = series[matches[0]]
            f = float(v)
            return f if np.isfinite(f) else None
    except Exception:
        pass
    return None


def _growth(df_fin, metric_key, scan_date):
    """YoY growth for a metric across two most recent filings before scan_date."""
    if df_fin is None or df_fin.empty:
        return None
    try:
        dates = sorted(pd.to_datetime(df_fin.columns).date, reverse=True)
        valid = [d for d in dates if d <= scan_date]
        if len(valid) < 2:
            return None
        cols = list(pd.to_datetime(df_fin.columns).date)
        c0   = df_fin[df_fin.columns[cols.index(valid[0])]]
        c1   = df_fin[df_fin.columns[cols.index(valid[1])]]
        cur  = _val(c0, metric_key)
        prev = _val(c1, metric_key)
        if cur is not None and prev and prev != 0:
            return (cur - prev) / abs(prev)
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE STOCK FETCH
# ══════════════════════════════════════════════════════════════════════════════

def fetch_stock_data(ticker, scan_date=None, csv_meta=None):
    """
    Fetches all valuation parameters for one ticker.

    ticker    : 'TICKER.NS' format
    scan_date : None = live mode | date = backtest mode
    csv_meta  : dict of extra columns from stocks.csv (name, notes, sector etc.)
    """
    is_backtest = scan_date is not None
    csv_meta    = csv_meta or {}

    try:
        tk   = yf.Ticker(ticker)
        info = tk.info

        # Use CSV metadata as fallback for name/sector
        name   = info.get("longName") or info.get("shortName") or csv_meta.get("name") or ticker
        sector = info.get("sector") or csv_meta.get("sector") or "Unknown"
        industry = info.get("industry") or csv_meta.get("notes") or "Unknown"
        shares = info.get("sharesOutstanding")

        # ── Price ──────────────────────────────────────────────────────────────
        if is_backtest:
            price = _get_historical_price(tk, scan_date)
        else:
            price = (info.get("currentPrice")
                     or info.get("regularMarketPrice")
                     or info.get("previousClose")
                     or info.get("open"))

        if not price or price <= 0:
            logger.debug(f"  {ticker}: no price data — skipping")
            return None

        # ── Market cap (informational only — no filtering) ─────────────────────
        raw_mcap = info.get("marketCap")
        if is_backtest and shares and shares > 0:
            market_cap_inr = price * shares
        elif raw_mcap and raw_mcap > 0:
            market_cap_inr = raw_mcap
        elif shares and shares > 0:
            market_cap_inr = price * shares
        else:
            market_cap_inr = 0

        market_cap_cr = round(market_cap_inr / CR_TO_INR, 1) if market_cap_inr else None

        # ── Financials ─────────────────────────────────────────────────────────
        if is_backtest:
            fin_col = _latest_col_before(tk.financials,    scan_date)
            bal_col = _latest_col_before(tk.balance_sheet, scan_date)
            cf_col  = _latest_col_before(tk.cashflow,      scan_date)

            total_revenue = _val(fin_col, "Total Revenue")
            net_income    = _val(fin_col, "Net Income")
            gross_profit  = _val(fin_col, "Gross Profit")
            ebit          = _val(fin_col, "Operating Income") or _val(fin_col, "Ebit")

            gross_margin  = (gross_profit / total_revenue) if gross_profit  and total_revenue else None
            op_margin     = (ebit         / total_revenue) if ebit          and total_revenue else None
            net_margin    = (net_income   / total_revenue) if net_income    and total_revenue else None

            total_assets  = _val(bal_col, "Total Assets")
            total_debt    = (_val(bal_col, "Long Term Debt") or 0) + (_val(bal_col, "Short Long Term Debt") or 0)
            cash          = _val(bal_col, "Cash And Cash Equivalents") or _val(bal_col, "Cash")
            equity        = _val(bal_col, "Total Stockholder Equity") or _val(bal_col, "Stockholders Equity")
            curr_liab     = _val(bal_col, "Total Current Liabilities")
            curr_assets   = _val(bal_col, "Total Current Assets")

            eps_ttm    = (net_income / shares) if net_income and shares and shares > 0 else None
            eps_fwd    = info.get("forwardEps")
            book_value = (equity     / shares) if equity     and shares and shares > 0 else None

            op_cashflow = (_val(cf_col, "Total Cash From Operating Activities")
                           or _val(cf_col, "Operating Cash Flow"))
            capex       = _val(cf_col, "Capital Expenditures")
            fcf         = (op_cashflow + capex) if op_cashflow and capex else op_cashflow

            current_ratio  = (curr_assets / curr_liab)     if curr_assets and curr_liab  and curr_liab  > 0 else None
            debt_equity    = (total_debt  / equity * 100)  if total_debt  and equity     and equity     > 0 else None
            roe            = (net_income  / equity)        if net_income  and equity     and equity     > 0 else None
            roa            = (net_income  / total_assets)  if net_income  and total_assets and total_assets > 0 else None

            roce = None
            if ebit and total_assets and curr_liab:
                cap_emp = total_assets - curr_liab
                if cap_emp > 0:
                    roce = ebit / cap_emp

            enterprise_val = market_cap_inr + (total_debt or 0) - (cash or 0)
            pe_ttm    = (price / eps_ttm)          if eps_ttm    and eps_ttm    > 0 else None
            pe_fwd    = info.get("forwardPE")
            pb        = (price / book_value)        if book_value and book_value > 0 else None
            ps        = (market_cap_inr / total_revenue) if total_revenue and total_revenue > 0 else None

            dep       = _val(cf_col, "Depreciation") if cf_col is not None else None
            ebitda    = (ebit + dep) if ebit and dep else ebit
            ev_ebitda = (enterprise_val / ebitda)        if ebitda        and ebitda        > 0 else None
            ev_revenue= (enterprise_val / total_revenue) if total_revenue and total_revenue > 0 else None

            revenue_growth  = _growth(tk.financials, "Total Revenue", scan_date)
            earnings_growth = _growth(tk.financials, "Net Income",    scan_date)
            peg = (pe_ttm / (earnings_growth * 100)) if pe_ttm and earnings_growth and earnings_growth > 0 else None

            div_yield    = info.get("dividendYield")
            payout_ratio = info.get("payoutRatio")
            week52_high, week52_low = _get_historical_52w(tk, scan_date)
            beta         = info.get("beta")
            target_mean  = info.get("targetMeanPrice")
            target_high  = info.get("targetHighPrice")
            target_low   = info.get("targetLowPrice")
            recommend    = info.get("recommendationKey", "")
            num_analysts = info.get("numberOfAnalystOpinions", 0)

        else:
            # ── Live mode ──────────────────────────────────────────────────────
            pe_ttm          = info.get("trailingPE")
            pe_fwd          = info.get("forwardPE")
            pb              = info.get("priceToBook")
            ps              = info.get("priceToSalesTrailing12Months")
            ev_ebitda       = info.get("enterpriseToEbitda")
            ev_revenue      = info.get("enterpriseToRevenue")
            peg             = info.get("pegRatio")
            eps_ttm         = info.get("trailingEps")
            eps_fwd         = info.get("forwardEps")
            revenue_growth  = info.get("revenueGrowth")
            earnings_growth = info.get("earningsGrowth")
            gross_margin    = info.get("grossMargins")
            op_margin       = info.get("operatingMargins")
            net_margin      = info.get("profitMargins")
            roe             = info.get("returnOnEquity")
            roa             = info.get("returnOnAssets")
            debt_equity     = info.get("debtToEquity")
            current_ratio   = info.get("currentRatio")
            book_value      = info.get("bookValue")
            fcf             = info.get("freeCashflow")
            op_cashflow     = info.get("operatingCashflow")
            total_revenue   = info.get("totalRevenue")
            div_yield       = info.get("dividendYield")
            payout_ratio    = info.get("payoutRatio")
            week52_high     = info.get("fiftyTwoWeekHigh")
            week52_low      = info.get("fiftyTwoWeekLow")
            beta            = info.get("beta")
            target_mean     = info.get("targetMeanPrice")
            target_high     = info.get("targetHighPrice")
            target_low      = info.get("targetLowPrice")
            recommend       = info.get("recommendationKey", "")
            num_analysts    = info.get("numberOfAnalystOpinions", 0)
            enterprise_val  = info.get("enterpriseValue")
            total_debt      = info.get("totalDebt")
            cash            = info.get("totalCash")
            total_assets    = info.get("totalAssets")
            curr_liab       = info.get("currentLiabilities")
            ebit            = info.get("ebit")
            roce = None
            if ebit and total_assets and curr_liab:
                cap_emp = total_assets - curr_liab
                if cap_emp > 0:
                    roce = ebit / cap_emp

        return {
            "ticker":          ticker.replace(".NS", ""),
            "name":            name,
            "sector":          sector,
            "industry":        industry,
            "notes":           csv_meta.get("notes", ""),
            "scan_date":       str(scan_date) if scan_date else str(TODAY),
            "mode":            "backtest" if is_backtest else "live",
            "price":           round(price, 2),
            "market_cap_cr":   market_cap_cr,
            "pe_ttm":          pe_ttm,         "pe_fwd":         pe_fwd,
            "pb":              pb,             "ps":             ps,
            "ev_ebitda":       ev_ebitda,      "ev_revenue":     ev_revenue,
            "peg":             peg,
            "eps_ttm":         eps_ttm,        "eps_fwd":        eps_fwd,
            "revenue_growth":  revenue_growth, "earnings_growth":earnings_growth,
            "gross_margin":    gross_margin,   "op_margin":      op_margin,
            "net_margin":      net_margin,
            "roe":             roe,            "roa":            roa,
            "roce":            roce,
            "debt_equity":     debt_equity,    "current_ratio":  current_ratio,
            "book_value":      book_value,
            "fcf":             fcf,            "op_cashflow":    op_cashflow,
            "total_revenue":   total_revenue,
            "div_yield":       div_yield,      "payout_ratio":   payout_ratio,
            "week52_high":     week52_high,    "week52_low":     week52_low,
            "beta":            beta,
            "target_mean":     target_mean,    "target_high":    target_high,
            "target_low":      target_low,
            "analyst_rec":     recommend,      "num_analysts":   num_analysts,
            "enterprise_val":  enterprise_val, "total_debt":     total_debt,
            "cash":            cash,           "shares_out":     shares,
        }

    except Exception as e:
        logger.debug(f"  {ticker}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# BATCH FETCH  (sequential or parallel)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_all_stocks(tickers, scan_date=None, delay=0.4,
                     workers=1, csv_meta_df=None):
    """
    Fetches all tickers from stocks.csv, sequential or parallel.

    tickers     : list of 'TICKER.NS' strings
    scan_date   : None = live | date = backtest
    delay       : seconds between calls (sequential) or jitter max (parallel)
    workers     : 1 = sequential; 2-4 = parallel (recommended max 4)
    csv_meta_df : DataFrame from load_tickers_from_csv with extra metadata
    """
    # Build per-ticker metadata lookup from CSV
    meta_lookup = {}
    if csv_meta_df is not None and not csv_meta_df.empty:
        for _, row in csv_meta_df.iterrows():
            t = str(row.get("ticker", "")).upper()
            meta_lookup[t] = row.to_dict()

    mode = f"BACKTEST ({scan_date})" if scan_date else "LIVE (today)"
    logger.info(
        f"Fetching {len(tickers)} stocks from stocks.csv | "
        f"Mode: {mode} | Workers: {workers}"
    )

    results = []

    if workers <= 1:
        # ── Sequential ────────────────────────────────────────────────────────
        for ticker in tqdm(tickers, desc="Fetching", unit="stock"):
            sym  = ticker.replace(".NS", "")
            meta = meta_lookup.get(sym, {})
            data = fetch_stock_data(ticker, scan_date=scan_date, csv_meta=meta)
            if data:
                results.append(data)
            time.sleep(delay)

    else:
        # ── Parallel ──────────────────────────────────────────────────────────
        import random

        def _fetch(ticker):
            time.sleep(random.uniform(0.05, delay))
            sym  = ticker.replace(".NS", "")
            meta = meta_lookup.get(sym, {})
            return fetch_stock_data(ticker, scan_date=scan_date, csv_meta=meta)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fetch, t): t for t in tickers}
            for future in tqdm(as_completed(futures), total=len(tickers),
                               desc=f"Fetching ({workers} workers)", unit="stock"):
                try:
                    data = future.result(timeout=30)
                    if data:
                        results.append(data)
                except Exception as e:
                    logger.debug(f"  Worker error: {e}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    logger.info(f"Successfully fetched: {len(df)} / {len(tickers)} stocks")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MONTHLY RETURN ENRICHMENT  (backtest)
# ══════════════════════════════════════════════════════════════════════════════

def enrich_with_monthly_returns(df, scan_date, workers=1):
    """
    Fetches prices at 30,60,...,365 days after scan_date for each stock.
    Adds columns: price_30d, return_30d, price_60d, return_60d, ... price_365d, return_365d
    Uses ONE history call per stock (efficient).
    """
    fetchable = [
        (days, scan_date + timedelta(days=days))
        for days in RETURN_INTERVALS
        if scan_date + timedelta(days=days) <= TODAY
    ]

    for days in RETURN_INTERVALS:
        df[f"price_{days}d"]  = np.nan
        df[f"return_{days}d"] = np.nan

    if not fetchable:
        logger.info("All return intervals are in the future — skipping.")
        df["actual_return_pct"] = np.nan
        df["fwd_date"]          = str(scan_date + timedelta(days=365))
        return df

    logger.info(
        f"Fetching prices at {len(fetchable)} intervals "
        f"({', '.join([f'+{d}d' for d,_ in fetchable])}) ..."
    )

    tickers    = df["ticker"].tolist()
    price_map  = {t: {} for t in tickers}

    def _fetch_history(ticker):
        try:
            tk    = yf.Ticker(f"{ticker}.NS")
            start = scan_date - timedelta(days=5)
            end   = scan_date + timedelta(days=370)
            hist  = tk.history(start=str(start), end=str(end), auto_adjust=True)
            if hist.empty:
                return ticker, {}
            hist.index = hist.index.date
            result = {}
            for days, target_date in fetchable:
                valid = hist[hist.index <= target_date]
                if not valid.empty:
                    result[days] = float(valid["Close"].iloc[-1])
            return ticker, result
        except Exception as e:
            logger.debug(f"  History error {ticker}: {e}")
            return ticker, {}

    if workers <= 1:
        for ticker in tqdm(tickers, desc="Monthly prices", unit="stock"):
            _, data = _fetch_history(ticker)
            price_map[ticker] = data
            time.sleep(0.2)
    else:
        import random
        def _fetch_with_jitter(ticker):
            time.sleep(random.uniform(0.05, 0.3))
            return _fetch_history(ticker)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fetch_with_jitter, t): t for t in tickers}
            for future in tqdm(as_completed(futures), total=len(tickers),
                               desc=f"Monthly prices ({workers} workers)", unit="stock"):
                try:
                    ticker, data = future.result(timeout=30)
                    price_map[ticker] = data
                except Exception as e:
                    logger.debug(f"  Worker error: {e}")

    fetchable_days = {d for d, _ in fetchable}
    for days in RETURN_INTERVALS:
        prices = df["ticker"].map(lambda t: price_map.get(t, {}).get(days))
        df[f"price_{days}d"] = prices
        if days in fetchable_days:
            df[f"return_{days}d"] = ((prices - df["price"]) / df["price"] * 100).round(2)

    df["actual_return_pct"] = df.get("return_365d", np.nan)
    df["fwd_date"]          = str(scan_date + timedelta(days=365))

    logger.info("Monthly return enrichment complete.")
    return df


def parse_scan_date(scan_date_str):
    """Parse and validate a scan date string (YYYY-MM-DD). Returns date or None."""
    if not scan_date_str:
        return None
    try:
        d = datetime.strptime(scan_date_str.strip(), "%Y-%m-%d").date()
        if d >= TODAY:
            logger.info("scan_date is today or future — running in LIVE mode.")
            return None
        return d
    except ValueError:
        logger.error(f"Invalid scan_date '{scan_date_str}'. Use YYYY-MM-DD format.")
        return None
