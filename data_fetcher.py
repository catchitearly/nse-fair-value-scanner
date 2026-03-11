"""
data_fetcher.py
---------------
Fetches NSE stock data from free sources (yfinance + NSE India CSV).

KEY FEATURES:
  - Live mode    : today's price + latest financials
  - Backtest mode: historical price on scan_date + filings before scan_date
  - Parallel fetch via ThreadPoolExecutor (--workers flag)
  - Monthly return intervals: 30,60,90,120,150,180,210,240,270,300,330,365 days
  - Results season warning: flags grey-zone dates where financials may be mixed
"""

import time
import logging
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from io import StringIO

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
CR_TO_INR = 1e7
TODAY     = date.today()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Monthly return checkpoints (days from scan_date)
RETURN_INTERVALS = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365]

# Results season grey zones (month, day_start, day_end) — when filings are mixed
# Q1 (Apr-Jun): results Oct 1 – Nov 14
# Q2 (Jul-Sep): results Jan 1 – Feb 14
# Q3 (Oct-Dec): results Apr 1 – May 14
# Q4 (Jan-Mar): results Jul 1 – Aug 14
GREY_ZONES = [
    (10,  1, 14, "Q1 FY results (Apr-Jun quarter) — some companies may not have filed yet"),
    (11,  1, 14, "Q1 FY results (Apr-Jun quarter) — stragglers still filing"),
    ( 1,  1, 14, "Q2 FY results (Jul-Sep quarter) — some companies may not have filed yet"),
    ( 2,  1, 14, "Q2 FY results (Jul-Sep quarter) — stragglers still filing"),
    ( 4,  1, 14, "Q3 FY results (Oct-Dec quarter) — some companies may not have filed yet"),
    ( 5,  1, 14, "Q3 FY results (Oct-Dec quarter) — stragglers still filing"),
    ( 7,  1, 14, "Q4/Full-Year results (Jan-Mar quarter) — some companies may not have filed yet"),
    ( 8,  1, 14, "Q4/Full-Year results (Jan-Mar quarter) — stragglers still filing"),
]


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS SEASON WARNING
# ══════════════════════════════════════════════════════════════════════════════

def check_results_season(scan_date):
    """
    Checks if scan_date falls in a quarterly results grey zone.
    Returns (is_grey_zone: bool, warning_message: str)
    """
    if scan_date is None:
        return False, ""

    m, d = scan_date.month, scan_date.day
    for gz_month, gz_start, gz_end, gz_reason in GREY_ZONES:
        if m == gz_month and gz_start <= d <= gz_end:
            msg = (
                f"\n{'⚠️ ' * 20}\n"
                f"  RESULTS SEASON WARNING\n"
                f"  Your scan_date {scan_date} falls in a quarterly results grey zone!\n"
                f"  Reason : {gz_reason}\n"
                f"  Risk   : Some companies have filed new quarterly results while others\n"
                f"           have not. Your scanner will MIX old and new financial data,\n"
                f"           making comparisons between stocks UNRELIABLE.\n"
                f"\n"
                f"  RECOMMENDED SAFE SCAN DATES (fully settled financials):\n"
                f"    Q1 settled : 15-Nov to 31-Dec  (Q1 Apr-Jun results all filed)\n"
                f"    Q2 settled : 15-Feb to 31-Mar  (Q2 Jul-Sep results all filed)\n"
                f"    Q3 settled : 15-May to 30-Jun  (Q3 Oct-Dec results all filed)\n"
                f"    Q4 settled : 15-Aug to 30-Sep  (Full-year results all filed)\n"
                f"{'⚠️ ' * 20}\n"
            )
            return True, msg

    return False, ""


# ══════════════════════════════════════════════════════════════════════════════
# NSE STOCK LIST
# ══════════════════════════════════════════════════════════════════════════════

def get_nse_stock_list():
    """Returns all NSE ticker symbols (TICKER.NS). Falls back to curated list."""
    try:
        logger.info("Fetching NSE equity list from NSE India ...")
        resp = requests.get(
            "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
            headers=HEADERS, timeout=15
        )
        if resp.status_code == 200:
            df = pd.read_csv(StringIO(resp.text))

            # Drop SME / extremely illiquid tickers that yfinance cannot price:
            #   - Symbols starting with digits (20MICRONS, 21STCENMGM etc.)
            #   - Symbols with hyphens or special chars
            df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
            df = df[~df["SYMBOL"].str.match(r"^\d")]        # drop digit-prefixed
            df = df[~df["SYMBOL"].str.contains(r"[^A-Z0-9&-]", regex=True)]  # keep clean

            # Sort: prefer mainboard large symbols (shorter names tend to be more liquid)
            df = df.sort_values("SYMBOL")

            symbols = df["SYMBOL"].dropna().tolist()
            logger.info(f"  NSE list: {len(symbols)} symbols loaded (after SME filter)")
            return [f"{s}.NS" for s in symbols]
    except Exception as e:
        logger.warning(f"NSE equity list fetch failed: {e}")

    logger.info("Using curated NSE mid-cap fallback list ...")
    fallback = [
        "KALYANKJIL","RKFORGE","MAHLIFE","CENTURYPLY","GREENPANEL",
        "KPITTECH","PERSISTENT","MPHASIS","COFORGE","LTTS","TATAELXSI",
        "SONACOMS","CRAFTSMAN","PRICOL","SUPRAJIT","ENDURANCE",
        "JKCEMENT","RAMCOCEM","HEIDELBERG","ORIENTCEM","BIRLACORPN",
        "KANSAINER","AKZOINDIA","GNFC","DEEPAKNTR","AARTI","VINATI",
        "FINEORG","GALAXYSURF","PIDILITIND","SUPREMEIND","ASTRAL",
        "APOLLOPIPE","HATHWAY","SUNTV","ZEEL","NAZARA","NETWORK18",
        "CHOLAFIN","MUTHOOTFIN","MANAPPURAM","IIFL","SPANDANA",
        "CREDITACC","ARMANFIN","UJJIVANSFB","EQUITASBNK","SURYODAY",
        "METROPOLIS","THYROCARE","KRSNAA","YATHARTH","MAXHEALTH",
        "RAINBOW","POLYMED","NEULANDLAB","SOLARA","GRANULES",
        "JUBLPHARMA","ALEMBICLTD","SEQUENT","RPGLIFE","TARSONS",
        "DIVI","NATCOPHARM","GLAND","BIOCON","LAURUS","NAVINFLUOR",
        "CLEAN","SIS","TEAMLEASE","QUESS","CAMPUS","BATA","RELAXO",
        "SHOPERSTOP","VMART","TRENT","ABFRL","MANYAVAR",
        "RBLBANK","FEDERALBNK","KTKBANK","DCBBANK","CSBBANK",
        "AUBANK","BANDHANBNK","IDFCFIRSTB","SOUTHBANK",
        "NIACL","STARHEALTH","ICICIGI","HDFCLIFE","SBILIFE",
        "GRINDWELL","CARBORUNIV","AIAENG","GRAPHITE","HEG",
        "ORIENTELEC","CROMPTON","HAVELLS","POLYCAB","KEI",
        "FINOLEX","HBLPOWER","EXIDEIND","GRAVITA","CUMMINSIND",
        "THERMAX","GREAVES","MAHINDCIE","SCHAEFFLER","IGARASHI",
        "PNBHOUSING","CANFINHOME","REPCO","AAVAS","HOMEFIRST","APTUS",
        "IRCTC","RVNL","RAILTEL","IRFC","CONCOR","HUDCO","NBCC","BHEL",
        "SAIL","NMDC","MOIL","NALCO","VEDL","HINDCOPPER","GMRINFRA",
        "SUZLON","INOXWIND","CESC","TORNTPOWER","KPRMILL","WELSPUNIND",
        "TRIDENT","RAYMOND","PAGEIND","VARDHMAN","TANLA","ROUTE",
        "INTELLECT","MASTEK","ZENSAR","CYIENT","DATAMATICS","BIRLASOFT",
    ]
    return [f"{s}.NS" for s in fallback]


# ══════════════════════════════════════════════════════════════════════════════
# HISTORICAL PRICE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_historical_price(tk, target_date):
    """Returns closing price on or before target_date (looks back up to 10 days)."""
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
    """Returns (52w_high, 52w_low) ending on scan_date."""
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
    """Returns the most recent column (as Series) from a yfinance DataFrame <= scan_date."""
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
    """Computes YoY growth for a metric across two most recent filings before scan_date."""
    if df_fin is None or df_fin.empty:
        return None
    try:
        dates = sorted(pd.to_datetime(df_fin.columns).date, reverse=True)
        valid = [d for d in dates if d <= scan_date]
        if len(valid) < 2:
            return None
        cols  = list(pd.to_datetime(df_fin.columns).date)
        c0    = df_fin[df_fin.columns[cols.index(valid[0])]]
        c1    = df_fin[df_fin.columns[cols.index(valid[1])]]
        cur   = _val(c0, metric_key)
        prev  = _val(c1, metric_key)
        if cur is not None and prev and prev != 0:
            return (cur - prev) / abs(prev)
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE STOCK FETCH
# ══════════════════════════════════════════════════════════════════════════════

def fetch_stock_data(ticker, scan_date=None, mcap_min=500, mcap_max=10000):
    """
    Fetches all valuation parameters for one ticker.
    scan_date=None  -> live mode (current data)
    scan_date=date  -> backtest mode (historical price + filings before that date)
    """
    is_backtest = scan_date is not None
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info

        name     = info.get("longName") or info.get("shortName") or ticker
        sector   = info.get("sector",   "Unknown")
        industry = info.get("industry", "Unknown")
        shares   = info.get("sharesOutstanding")

        # ── Price ─────────────────────────────────────────────────────────────
        if is_backtest:
            price = _get_historical_price(tk, scan_date)
        else:
            price = (info.get("currentPrice")
                     or info.get("regularMarketPrice")
                     or info.get("previousClose")
                     or info.get("open"))
        if not price or price <= 0:
            logger.debug(f"  {ticker}: no price — skipping")
            return None

        # ── Market cap ────────────────────────────────────────────────────────
        # Try multiple sources in order of reliability
        raw_mcap = info.get("marketCap") or info.get("market_cap")
        if is_backtest and shares and shares > 0:
            market_cap_inr = price * shares
        elif raw_mcap and raw_mcap > 0:
            market_cap_inr = raw_mcap
        elif shares and shares > 0:
            market_cap_inr = price * shares
        else:
            logger.debug(f"  {ticker}: no market cap — skipping")
            return None

        market_cap_cr = market_cap_inr / CR_TO_INR

        # mcap_min=0 means no lower bound
        if mcap_min > 0 and market_cap_cr < mcap_min:
            return None
        if market_cap_cr > mcap_max:
            return None

        # ── Financials ────────────────────────────────────────────────────────
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
            book_value = (equity / shares)     if equity     and shares and shares > 0 else None

            op_cashflow = _val(cf_col, "Total Cash From Operating Activities") or _val(cf_col, "Operating Cash Flow")
            capex       = _val(cf_col, "Capital Expenditures")
            fcf         = (op_cashflow + capex) if op_cashflow and capex else op_cashflow

            current_ratio  = (curr_assets / curr_liab)     if curr_assets and curr_liab and curr_liab > 0 else None
            debt_equity    = (total_debt  / equity * 100)  if total_debt  and equity    and equity   > 0 else None
            roe            = (net_income  / equity)        if net_income  and equity    and equity   > 0 else None
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
            "scan_date":       str(scan_date) if scan_date else str(TODAY),
            "mode":            "backtest" if is_backtest else "live",
            "price":           round(price, 2),
            "market_cap_cr":   round(market_cap_cr, 1),
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
                     mcap_min=500, mcap_max=10000, workers=1):
    """
    Fetches all tickers — sequential (workers=1) or parallel (workers>1).

    workers=1  : safe, polite, no rate-limit risk
    workers=4  : ~4x faster; recommended max for yfinance free tier
    workers=8  : fastest but risk of temporary IP block from yfinance
    """
    mode = f"BACKTEST ({scan_date})" if scan_date else "LIVE (today)"
    logger.info(
        f"Scanning {len(tickers)} tickers | Mode: {mode} | "
        f"MCap: Rs{mcap_min}-{mcap_max} cr | Workers: {workers}"
    )

    results = []

    if workers <= 1:
        # ── Sequential ────────────────────────────────────────────────────────
        for ticker in tqdm(tickers, desc="Fetching", unit="stock"):
            data = fetch_stock_data(ticker, scan_date=scan_date,
                                    mcap_min=mcap_min, mcap_max=mcap_max)
            if data:
                results.append(data)
            time.sleep(delay)

    else:
        # ── Parallel via ThreadPoolExecutor ───────────────────────────────────
        # Add small jitter to avoid thundering-herd on yfinance
        import random

        def _fetch_with_delay(ticker):
            time.sleep(random.uniform(0.1, delay))
            return fetch_stock_data(ticker, scan_date=scan_date,
                                    mcap_min=mcap_min, mcap_max=mcap_max)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fetch_with_delay, t): t for t in tickers}
            for future in tqdm(as_completed(futures), total=len(tickers),
                               desc=f"Fetching ({workers} workers)", unit="stock"):
                try:
                    data = future.result(timeout=30)
                    if data:
                        results.append(data)
                except Exception as e:
                    logger.debug(f"Worker error for {futures[future]}: {e}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    logger.info(f"Qualifying stocks found: {len(df)}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MONTHLY RETURN ENRICHMENT  (backtest)
# ══════════════════════════════════════════════════════════════════════════════

def enrich_with_monthly_returns(df, scan_date, workers=1):
    """
    For each stock, fetches prices at 30,60,90,...,365 days after scan_date
    and computes the return % at each interval.

    Only fetches intervals where scan_date + N days <= TODAY.
    Adds columns: return_30d, return_60d, ..., return_365d
    Also adds: price_30d, price_60d, ..., price_365d
    """
    # Determine which intervals are fetchable (date already past)
    fetchable = []
    for days in RETURN_INTERVALS:
        target = scan_date + timedelta(days=days)
        if target <= TODAY:
            fetchable.append((days, target))
        else:
            logger.info(f"  Interval +{days}d ({target}) is future — skipping")

    if not fetchable:
        logger.info("No return intervals are in the past — skipping return enrichment.")
        for days in RETURN_INTERVALS:
            df[f"price_{days}d"]  = np.nan
            df[f"return_{days}d"] = np.nan
        return df

    logger.info(
        f"Fetching prices at {len(fetchable)} intervals "
        f"({', '.join([f'+{d}d' for d,_ in fetchable])}) for {len(df)} stocks ..."
    )

    tickers = df["ticker"].tolist()

    # Build a dict: ticker -> {days: price}
    # Fetch one full year of history per stock in ONE call (efficient)
    price_map = {t: {} for t in tickers}

    def _fetch_ticker_history(ticker):
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
            logger.debug(f"  History fetch failed for {ticker}: {e}")
            return ticker, {}

    if workers <= 1:
        for ticker in tqdm(tickers, desc="Monthly prices", unit="stock"):
            _, data = _fetch_ticker_history(ticker)
            price_map[ticker] = data
            time.sleep(0.25)
    else:
        import random
        def _fetch_with_jitter(ticker):
            time.sleep(random.uniform(0.05, 0.3))
            return _fetch_ticker_history(ticker)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fetch_with_jitter, t): t for t in tickers}
            for future in tqdm(as_completed(futures), total=len(tickers),
                               desc=f"Monthly prices ({workers} workers)", unit="stock"):
                try:
                    ticker, data = future.result(timeout=30)
                    price_map[ticker] = data
                except Exception as e:
                    logger.debug(f"  Worker error: {e}")

    # ── Attach price and return columns to DataFrame ───────────────────────────
    for days in RETURN_INTERVALS:
        prices  = df["ticker"].map(lambda t: price_map.get(t, {}).get(days))
        df[f"price_{days}d"]  = prices
        if days in [d for d, _ in fetchable]:
            df[f"return_{days}d"] = ((prices - df["price"]) / df["price"] * 100).round(2)
        else:
            df[f"return_{days}d"] = np.nan

    # Convenience alias for final return (used in accuracy summary)
    df["actual_return_pct"] = df["return_365d"] if "return_365d" in df.columns else np.nan
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
