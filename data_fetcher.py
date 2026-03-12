"""
data_fetcher.py
---------------
Fetches NSE stock data from yfinance.

ARCHITECTURE — why we do it this way:
  tk.info        : Broken in GitHub Actions CI since yfinance 0.2.38+.
                   Returns a stub dict with almost no data due to cookie/crumb
                   auth changes. We use it ONLY for supplementary analyst ratios
                   (PE, PB, margins etc) and accept that these may be None in CI.

  tk.fast_info   : Works in CI. Provides: last_price, market_cap, shares,
                   52w high/low, currency. This is our PRIMARY price source.

  tk.history()   : Works in CI. Provides OHLCV history. Used for:
                   - Price confirmation/fallback
                   - Backtest historical prices
                   - Monthly return intervals

  tk.financials / .balance_sheet / .cashflow:
                   Work in CI (different endpoint from .info).
                   Used for all fundamental data in both live and backtest modes.
"""

import time
import random
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CURL_CFFI SESSION SETUP — bypasses Yahoo Finance bot detection in CI
# ══════════════════════════════════════════════════════════════════════════════

def setup_yfinance_session():
    """
    Configures yfinance to use curl_cffi with Chrome impersonation.
    This bypasses Yahoo Finance's TLS fingerprint bot-detection that blocks
    GitHub Actions (and other CI) IP ranges.

    curl_cffi must be installed: pip install curl_cffi>=0.7.0
    yfinance>=0.2.54 picks it up automatically when present.

    Call once at startup before any yf.Ticker() calls.
    """
    try:
        from curl_cffi import requests as cffi_requests
        session = cffi_requests.Session(impersonate="chrome")
        yf.set_tz_cache_location("/tmp/yf_tz_cache")
        # Monkey-patch yfinance's default session
        import yfinance.utils as yf_utils
        yf_utils.requests = cffi_requests
        logger.info("curl_cffi session active — Chrome impersonation enabled")
        return session
    except ImportError:
        logger.warning(
            "curl_cffi not installed — yfinance will use default requests session.\n"
            "  If you get rate-limit errors, install it: pip install curl_cffi>=0.7.0"
        )
        return None
    except Exception as e:
        logger.warning(f"curl_cffi setup failed ({e}) — falling back to default session")
        return None

CR_TO_INR        = 1e7
TODAY            = date.today()
RETURN_INTERVALS = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365]

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
                f"  Risk   : Financial data will be mixed across companies.\n"
                f"\n"
                f"  SAFE SCAN DATES:\n"
                f"    Nov 15 – Dec 31  → Q1 (Apr-Jun) fully settled\n"
                f"    Feb 15 – Mar 31  → Q2 (Jul-Sep) fully settled\n"
                f"    May 15 – Jun 30  → Q3 (Oct-Dec) fully settled\n"
                f"    Aug 15 – Sep 30  → Q4 (Jan-Mar) fully settled\n"
                f"{'=' * 65}\n"
            )
            return True, msg
    return False, ""


# ══════════════════════════════════════════════════════════════════════════════
# CSV LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_tickers_from_csv(csv_path="stocks.csv"):
    """
    Reads stock universe from CSV. Auto-detects tab or comma separator.
    Supports: 'Symbol'/'ticker' column and 'Stock Name'/'name' column.
    Returns (tickers_list, meta_dataframe).
    """
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            first_line = f.readline()

        sep = "\t" if "\t" in first_line else ","
        df  = pd.read_csv(csv_path, sep=sep, engine="python")
        df.columns = df.columns.str.strip()
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()

        col_map = {}
        for col in df.columns:
            low = col.lower()
            if low in ("symbol", "ticker", "nse symbol", "nse_symbol", "scrip"):
                col_map[col] = "ticker"
            elif low in ("stock name", "stock_name", "name", "company",
                         "company name", "company_name"):
                col_map[col] = "name"
            elif low in ("sector", "industry"):
                col_map[col] = "sector"
            elif low in ("notes", "note", "remarks"):
                col_map[col] = "notes"
        df.rename(columns=col_map, inplace=True)

        if "ticker" not in df.columns:
            logger.error(
                f"No Symbol/ticker column in '{csv_path}'.\n"
                f"  Columns found: {list(df.columns)}\n"
                f"  Expected: 'Symbol' or 'ticker'"
            )
            return [], pd.DataFrame()

        df["ticker"] = (df["ticker"].astype(str).str.strip().str.upper()
                        .str.replace(r"\.NS$", "", regex=True))
        df = df[df["ticker"].notna() & (df["ticker"] != "") & (df["ticker"] != "NAN")]
        df = df.drop_duplicates(subset="ticker").reset_index(drop=True)

        if "name" not in df.columns:
            df["name"] = df["ticker"]

        tickers = [f"{t}.NS" for t in df["ticker"].tolist()]
        logger.info(
            f"Loaded {len(tickers)} tickers from '{csv_path}' "
            f"(sep: {'TAB' if sep == chr(9) else 'COMMA'})"
        )
        return tickers, df

    except FileNotFoundError:
        logger.error(
            f"'{csv_path}' not found.\n"
            f"  Required column: Symbol (or ticker)\n"
            f"  Optional: Stock Name (or name), sector, notes"
        )
        return [], pd.DataFrame()
    except Exception as e:
        logger.error(f"Error reading '{csv_path}': {e}")
        return [], pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# CORE DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_float(val):
    """Convert to float, return None if invalid."""
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _get_fast_info(tk):
    """
    Fetch fast_info safely. Returns a dict of guaranteed-reliable fields.
    fast_info works in GitHub Actions where tk.info often fails.
    """
    result = {}
    try:
        fi = tk.fast_info
        result["price"]      = _safe_float(getattr(fi, "last_price",       None))
        result["market_cap"] = _safe_float(getattr(fi, "market_cap",       None))
        result["shares"]     = _safe_float(getattr(fi, "shares",           None))
        result["week52_high"]= _safe_float(getattr(fi, "year_high",        None))
        result["week52_low"] = _safe_float(getattr(fi, "year_low",         None))
        result["currency"]   = getattr(fi, "currency", "INR")
    except Exception as e:
        logger.info(f"  fast_info error for {ticker if "ticker" in dir() else "?"}: {e}")
    return result


def _get_recent_price_from_history(tk, ticker):
    """
    Gets the most recent closing price via tk.history().
    Most reliable method in CI environments.
    """
    try:
        hist = tk.history(period="5d", auto_adjust=True)
        if not hist.empty:
            close = hist["Close"].dropna()
            if not close.empty:
                return _safe_float(close.iloc[-1])
    except Exception as e:
        logger.info(f"  history price error for {ticker}: {e}")
    return None


def _get_supplementary_info(tk):
    """
    Attempt tk.info for supplementary ratios (PE, PB, margins etc).
    These are NICE-TO-HAVE — if info is stubbed out in CI, we get None
    for these fields but the stock still gets included (price from fast_info).
    Never blocks a stock from being included.
    """
    try:
        info = tk.info
        # Detect stub: yfinance returns {'trailingPegRatio': None} when rate-limited
        meaningful_keys = {
            "trailingPE", "forwardPE", "priceToBook", "pegRatio",
            "trailingEps", "forwardEps", "returnOnEquity", "profitMargins",
            "revenueGrowth", "earningsGrowth", "debtToEquity",
            "operatingMargins", "currentRatio", "dividendYield",
            "freeCashflow", "operatingCashflow", "totalRevenue",
            "beta", "targetMeanPrice", "recommendationKey",
            "numberOfAnalystOpinions", "longName", "shortName",
            "sector", "industry", "sharesOutstanding",
        }
        has_data = any(
            info.get(k) is not None
            for k in meaningful_keys
        )
        if has_data:
            return info
    except Exception as e:
        logger.info(f"  info error: {e}")
    return {}


def _get_historical_price(tk, target_date):
    """Price on or before target_date, looking back up to 10 calendar days."""
    start = target_date - timedelta(days=10)
    end   = target_date + timedelta(days=1)
    try:
        hist = tk.history(start=str(start), end=str(end), auto_adjust=True)
        if not hist.empty:
            hist.index = hist.index.date
            valid = hist[hist.index <= target_date]
            if not valid.empty:
                return _safe_float(valid["Close"].iloc[-1])
    except Exception as e:
        logger.debug(f"historical price error: {e}")
    return None


def _get_full_year_history(tk, scan_date):
    """
    Fetches one full year of OHLCV history ending at scan_date + 365 days.
    Returns a DataFrame indexed by date, or empty DataFrame on failure.
    Used for backtest 52w stats and monthly return intervals.
    """
    start = scan_date - timedelta(days=370)
    end   = scan_date + timedelta(days=370)
    try:
        hist = tk.history(start=str(start), end=str(end), auto_adjust=True)
        if not hist.empty:
            hist.index = hist.index.date
        return hist
    except Exception as e:
        logger.debug(f"full year history error: {e}")
        return pd.DataFrame()


def _latest_col_before(df_fin, scan_date):
    """Most recent column from a yfinance financials DataFrame on or before scan_date."""
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
    """Extract float from Series by partial key match."""
    if series is None:
        return None
    try:
        matches = [k for k in series.index if key.lower() in str(k).lower()]
        if matches:
            return _safe_float(series[matches[0]])
    except Exception:
        pass
    return None


def _growth(df_fin, metric_key, scan_date):
    """YoY growth across two most recent filings before scan_date."""
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

    Data priority (most to least reliable in CI):
      1. tk.fast_info   → price, market_cap, shares, 52w high/low
      2. tk.history()   → price confirmation, historical prices
      3. tk.financials / .balance_sheet / .cashflow → fundamentals
      4. tk.info        → supplementary ratios only (PE, PB etc), may be empty in CI
    """
    is_backtest = scan_date is not None
    csv_meta    = csv_meta or {}

    try:
        tk = yf.Ticker(ticker)

        # ── Step 1: fast_info — primary, works in CI ───────────────────────────
        fi = _get_fast_info(tk)

        # ── Step 2: price ──────────────────────────────────────────────────────
        if is_backtest:
            price = _get_historical_price(tk, scan_date)
        else:
            price = fi.get("price")
            if not price or price <= 0:
                # Fallback to history
                price = _get_recent_price_from_history(tk, ticker)

        if not price or price <= 0:
            logger.info(f"  SKIP {ticker} — no price (fast_info={fi.get('price')}, history=None)")
            return None

        # ── Step 3: supplementary info (nice-to-have, non-blocking) ───────────
        info = _get_supplementary_info(tk)
        info_ok = bool(info)

        # Name / sector — CSV metadata as fallback
        name     = info.get("longName") or info.get("shortName") or csv_meta.get("name") or ticker
        sector   = info.get("sector")   or csv_meta.get("sector")   or "Unknown"
        industry = info.get("industry") or csv_meta.get("notes")    or "Unknown"

        # Shares — fast_info is more reliable than info
        shares = fi.get("shares") or _safe_float(info.get("sharesOutstanding"))

        # ── Step 4: market cap ─────────────────────────────────────────────────
        raw_mcap = fi.get("market_cap") or _safe_float(info.get("marketCap"))
        if is_backtest and shares and shares > 0:
            market_cap_inr = price * shares
        elif raw_mcap and raw_mcap > 0:
            market_cap_inr = raw_mcap
        elif shares and shares > 0:
            market_cap_inr = price * shares
        else:
            market_cap_inr = 0
        market_cap_cr = round(market_cap_inr / CR_TO_INR, 1) if market_cap_inr else None

        logger.info(
            f"  OK   {ticker:20s} price={price:>8.2f}  "
            f"mcap={str(market_cap_cr)+'cr':>12}  "
            f"info={'YES' if info_ok else 'STUB'}"
        )

        # ── Step 5: Fundamentals from financials / balance_sheet / cashflow ────
        # These use a different yfinance endpoint from .info — work reliably in CI

        # --- Common to both live and backtest modes ---
        try:
            fin   = tk.financials
            bal   = tk.balance_sheet
            cf    = tk.cashflow
        except Exception:
            fin = bal = cf = None

        if is_backtest:
            fin_col = _latest_col_before(fin, scan_date)
            bal_col = _latest_col_before(bal, scan_date)
            cf_col  = _latest_col_before(cf,  scan_date)
        else:
            # Live mode: use the most recent column
            fin_col = fin.iloc[:, 0] if fin is not None and not fin.empty else None
            bal_col = bal.iloc[:, 0] if bal is not None and not bal.empty else None
            cf_col  = cf.iloc[:,  0] if cf  is not None and not cf.empty  else None

        total_revenue  = _val(fin_col, "Total Revenue")
        net_income     = _val(fin_col, "Net Income")
        gross_profit   = _val(fin_col, "Gross Profit")
        ebit           = _val(fin_col, "Operating Income") or _val(fin_col, "Ebit")

        gross_margin   = (gross_profit / total_revenue) if gross_profit and total_revenue else None
        op_margin_calc = (ebit         / total_revenue) if ebit         and total_revenue else None
        net_margin_calc= (net_income   / total_revenue) if net_income   and total_revenue else None

        total_assets   = _val(bal_col, "Total Assets")
        ltd            = _val(bal_col, "Long Term Debt")      or 0
        std            = _val(bal_col, "Short Long Term Debt") or 0
        total_debt     = ltd + std
        cash           = (_val(bal_col, "Cash And Cash Equivalents")
                          or _val(bal_col, "Cash"))
        equity         = (_val(bal_col, "Total Stockholder Equity")
                          or _val(bal_col, "Stockholders Equity"))
        curr_liab      = _val(bal_col, "Total Current Liabilities")
        curr_assets    = _val(bal_col, "Total Current Assets")

        op_cashflow    = (_val(cf_col, "Total Cash From Operating Activities")
                          or _val(cf_col, "Operating Cash Flow"))
        capex          = _val(cf_col, "Capital Expenditures")
        fcf_calc       = (op_cashflow + capex) if op_cashflow and capex else op_cashflow
        dep            = _val(cf_col, "Depreciation")
        ebitda         = (ebit + dep) if ebit and dep else ebit

        # Derived ratios from financials
        eps_ttm_calc   = (net_income / shares) if net_income and shares and shares > 0 else None
        book_val_calc  = (equity     / shares) if equity     and shares and shares > 0 else None
        current_ratio_calc = (curr_assets / curr_liab)    if curr_assets and curr_liab and curr_liab > 0 else None
        debt_equity_calc   = (total_debt  / equity * 100) if total_debt  and equity   and equity   > 0 else None
        roe_calc           = (net_income  / equity)       if net_income  and equity   and equity   > 0 else None
        roa_calc           = (net_income  / total_assets) if net_income  and total_assets and total_assets > 0 else None

        roce_calc = None
        if ebit and total_assets and curr_liab:
            cap_emp = total_assets - curr_liab
            if cap_emp > 0:
                roce_calc = ebit / cap_emp

        enterprise_val = market_cap_inr + total_debt - (cash or 0) if market_cap_inr else None
        ev_ebitda_calc = (enterprise_val / ebitda)        if enterprise_val and ebitda        and ebitda        > 0 else None
        ev_revenue_calc= (enterprise_val / total_revenue) if enterprise_val and total_revenue and total_revenue > 0 else None

        # Revenue/earnings growth
        revenue_growth  = _growth(fin, "Total Revenue", scan_date if is_backtest else TODAY)
        earnings_growth = _growth(fin, "Net Income",    scan_date if is_backtest else TODAY)

        # ── Step 6: Supplementary from info (PE, PB, analyst data etc) ─────────
        # Use calculated values from financials first, fall back to info
        pe_ttm   = (price / eps_ttm_calc)  if eps_ttm_calc  and eps_ttm_calc  > 0 else _safe_float(info.get("trailingPE"))
        pe_fwd   = _safe_float(info.get("forwardPE"))
        pb       = (price / book_val_calc) if book_val_calc and book_val_calc > 0 else _safe_float(info.get("priceToBook"))
        ps       = _safe_float(info.get("priceToSalesTrailing12Months"))
        peg      = _safe_float(info.get("pegRatio"))
        eps_ttm  = eps_ttm_calc  or _safe_float(info.get("trailingEps"))
        eps_fwd  = _safe_float(info.get("forwardEps"))
        book_value = book_val_calc or _safe_float(info.get("bookValue"))

        # Margins — prefer calculated, fallback to info
        op_margin  = op_margin_calc   or _safe_float(info.get("operatingMargins"))
        net_margin = net_margin_calc  or _safe_float(info.get("profitMargins"))
        roe        = roe_calc         or _safe_float(info.get("returnOnEquity"))
        roa        = roa_calc         or _safe_float(info.get("returnOnAssets"))
        debt_equity= debt_equity_calc
        current_ratio = current_ratio_calc

        ev_ebitda  = ev_ebitda_calc  or _safe_float(info.get("enterpriseToEbitda"))
        ev_revenue = ev_revenue_calc or _safe_float(info.get("enterpriseToRevenue"))

        peg_calc   = (pe_ttm / (earnings_growth * 100)) if pe_ttm and earnings_growth and earnings_growth > 0 else None
        peg        = peg_calc or peg

        # Info-only fields
        div_yield    = _safe_float(info.get("dividendYield"))
        payout_ratio = _safe_float(info.get("payoutRatio"))
        beta         = _safe_float(info.get("beta"))
        target_mean  = _safe_float(info.get("targetMeanPrice"))
        target_high  = _safe_float(info.get("targetHighPrice"))
        target_low   = _safe_float(info.get("targetLowPrice"))
        recommend    = info.get("recommendationKey", "")
        num_analysts = info.get("numberOfAnalystOpinions", 0)

        # 52w high/low — fast_info is most reliable
        week52_high = fi.get("week52_high") or _safe_float(info.get("fiftyTwoWeekHigh"))
        week52_low  = fi.get("week52_low")  or _safe_float(info.get("fiftyTwoWeekLow"))

        # For backtest: compute 52w from history if fast_info gave live values
        if is_backtest:
            hist = _get_full_year_history(tk, scan_date)
            if not hist.empty:
                hist_to_date = hist[hist.index <= scan_date]
                if not hist_to_date.empty:
                    week52_high = _safe_float(hist_to_date["High"].max())
                    week52_low  = _safe_float(hist_to_date["Low"].min())

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
            "pe_ttm":          pe_ttm,          "pe_fwd":          pe_fwd,
            "pb":              pb,              "ps":              ps,
            "ev_ebitda":       ev_ebitda,       "ev_revenue":      ev_revenue,
            "peg":             peg,
            "eps_ttm":         eps_ttm,         "eps_fwd":         eps_fwd,
            "revenue_growth":  revenue_growth,  "earnings_growth": earnings_growth,
            "gross_margin":    gross_margin,    "op_margin":       op_margin,
            "net_margin":      net_margin,
            "roe":             roe,             "roa":             roa,
            "roce":            roce_calc,
            "debt_equity":     debt_equity,     "current_ratio":   current_ratio,
            "book_value":      book_value,
            "fcf":             fcf_calc,        "op_cashflow":     op_cashflow,
            "total_revenue":   total_revenue,
            "div_yield":       div_yield,       "payout_ratio":    payout_ratio,
            "week52_high":     week52_high,     "week52_low":      week52_low,
            "beta":            beta,
            "target_mean":     target_mean,     "target_high":     target_high,
            "target_low":      target_low,
            "analyst_rec":     recommend,       "num_analysts":    num_analysts,
            "enterprise_val":  enterprise_val,  "total_debt":      total_debt,
            "cash":            cash,            "shares_out":      shares,
        }

    except Exception as e:
        logger.info(f"  SKIP {ticker} — unexpected error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# BATCH FETCH
# ══════════════════════════════════════════════════════════════════════════════

def fetch_all_stocks(tickers, scan_date=None, delay=0.5,
                     workers=1, csv_meta_df=None):
    """
    Fetches all tickers. workers=1 is safest; workers=2-4 for speed.
    Each OK/SKIP is logged at INFO level so GitHub Actions logs show progress.
    """
    meta_lookup = {}
    if csv_meta_df is not None and not csv_meta_df.empty:
        for _, row in csv_meta_df.iterrows():
            t = str(row.get("ticker", "")).upper()
            meta_lookup[t] = row.to_dict()

    mode = f"BACKTEST ({scan_date})" if scan_date else "LIVE (today)"
    logger.info(f"Fetching {len(tickers)} stocks | {mode} | workers={workers} | delay={delay}s")

    # ── Quick connectivity check before full scan ──────────────────────────────
    logger.info("  Running connectivity check (INFY.NS) ...")
    _probe_ticker = tickers[0] if tickers else "INFY.NS"
    try:
        _probe = yf.Ticker(_probe_ticker)
        _fi    = _probe.fast_info
        _price = getattr(_fi, "last_price", None)
        if _price and float(_price) > 0:
            logger.info(f"  Connectivity OK — {_probe_ticker} last_price={_price:.2f}")
        else:
            # Try history as fallback
            _hist = _probe.history(period="3d", auto_adjust=True)
            if not _hist.empty:
                logger.info(f"  Connectivity OK via history — {_probe_ticker} close={_hist['Close'].iloc[-1]:.2f}")
            else:
                logger.warning(
                    f"  Connectivity WARNING — {_probe_ticker} returned no price data.\n"
                    f"  fast_info.last_price={_price}, history empty.\n"
                    f"  yfinance may be rate-limiting. Proceeding anyway ..."
                )
    except Exception as _e:
        logger.warning(f"  Connectivity check error: {_e}. Proceeding anyway ...")

    results = []

    if workers <= 1:
        for ticker in tqdm(tickers, desc="Fetching", unit="stock"):
            sym  = ticker.replace(".NS", "")
            data = fetch_stock_data(ticker, scan_date=scan_date,
                                    csv_meta=meta_lookup.get(sym, {}))
            if data:
                results.append(data)
            time.sleep(delay)

    else:
        def _fetch(ticker):
            # Minimum 0.5s even if user passes a lower delay, to avoid CI rate-limits
            jitter = random.uniform(max(0.3, delay * 0.5), max(0.8, delay))
            time.sleep(jitter)
            sym = ticker.replace(".NS", "")
            return fetch_stock_data(ticker, scan_date=scan_date,
                                    csv_meta=meta_lookup.get(sym, {}))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fetch, t): t for t in tickers}
            for future in tqdm(as_completed(futures), total=len(tickers),
                               desc=f"Fetching ({workers} workers)", unit="stock"):
                try:
                    data = future.result(timeout=60)
                    if data:
                        results.append(data)
                except Exception as e:
                    logger.info(f"  Worker error: {e}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    logger.info(f"Fetched: {len(df)} / {len(tickers)} stocks successfully")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MONTHLY RETURN ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════════

def enrich_with_monthly_returns(df, scan_date, workers=1):
    """
    Fetches prices at 30,60,...,365 days after scan_date via tk.history().
    One history call per stock covers the full range efficiently.
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
        logger.info("All return intervals are future — skipping.")
        df["actual_return_pct"] = np.nan
        df["fwd_date"]          = str(scan_date + timedelta(days=365))
        return df

    logger.info(
        f"Fetching prices at {len(fetchable)} intervals: "
        f"{', '.join(f'+{d}d' for d,_ in fetchable)} ..."
    )

    tickers   = df["ticker"].tolist()
    price_map = {t: {} for t in tickers}

    def _fetch_hist(ticker):
        try:
            tk    = yf.Ticker(f"{ticker}.NS")
            hist  = _get_full_year_history(tk, scan_date)
            if hist.empty:
                return ticker, {}
            result = {}
            for days, target_date in fetchable:
                valid = hist[hist.index <= target_date]
                if not valid.empty:
                    result[days] = _safe_float(valid["Close"].iloc[-1])
            return ticker, result
        except Exception as e:
            logger.debug(f"  Return history error {ticker}: {e}")
            return ticker, {}

    if workers <= 1:
        for ticker in tqdm(tickers, desc="Monthly prices", unit="stock"):
            _, data = _fetch_hist(ticker)
            price_map[ticker] = data
            time.sleep(0.3)
    else:
        def _fetch_with_jitter(ticker):
            time.sleep(random.uniform(0.1, 0.4))
            return _fetch_hist(ticker)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fetch_with_jitter, t): t for t in tickers}
            for future in tqdm(as_completed(futures), total=len(tickers),
                               desc=f"Monthly prices ({workers} workers)", unit="stock"):
                try:
                    ticker, data = future.result(timeout=60)
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
    if not scan_date_str:
        return None
    try:
        d = datetime.strptime(scan_date_str.strip(), "%Y-%m-%d").date()
        if d >= TODAY:
            logger.info("scan_date is today or future — switching to LIVE mode.")
            return None
        return d
    except ValueError:
        logger.error(f"Invalid scan_date '{scan_date_str}'. Use YYYY-MM-DD.")
        return None
