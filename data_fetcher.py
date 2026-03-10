"""
data_fetcher.py
---------------
Fetches NSE stock data from free sources:
  - yfinance  (financials, price, market cap)
  - NSE India (stock list)

Supports HISTORICAL DATE mode for backtesting:
  - Price is pulled as the closing price on/before scan_date
  - Financials (EPS, book value, margins, etc.) are pulled from the
    most-recent quarterly/annual filing BEFORE scan_date using
    yfinance .financials / .balance_sheet / .cashflow history
  - Market cap is reconstructed as: historical_price × shares_outstanding
"""

import time
import logging
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, datetime, timedelta
from tqdm import tqdm
from io import StringIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MARKET_CAP_MIN_CR = 500
MARKET_CAP_MAX_CR = 10_000
CR_TO_INR         = 1e7

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

TODAY = date.today()


# ══════════════════════════════════════════════════════════════════════════════
# NSE STOCK LIST
# ══════════════════════════════════════════════════════════════════════════════

def get_nse_stock_list() -> list:
    """
    Returns all NSE ticker symbols in Yahoo Finance format (TICKER.NS).
    Falls back to a curated mid-cap universe if NSE CSV is unavailable.
    """
    try:
        logger.info("Fetching NSE equity list from NSE India ...")
        resp = requests.get(
            "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
            headers=HEADERS, timeout=15
        )
        if resp.status_code == 200:
            df = pd.read_csv(StringIO(resp.text))
            symbols = df["SYMBOL"].dropna().tolist()
            logger.info(f"  NSE list: {len(symbols)} symbols loaded")
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
# HISTORICAL DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_scan_date(scan_date_str):
    """Parses scan date string (YYYY-MM-DD). Returns None for today/live mode."""
    if not scan_date_str:
        return None
    try:
        d = datetime.strptime(scan_date_str.strip(), "%Y-%m-%d").date()
        if d >= TODAY:
            logger.info("scan_date is today or future -- running in LIVE mode.")
            return None
        return d
    except ValueError:
        logger.error(f"Invalid scan_date '{scan_date_str}'. Use YYYY-MM-DD format.")
        return None


def _get_historical_price(tk, scan_date):
    """
    Returns closing price on scan_date (or nearest prior trading day).
    Looks back up to 10 calendar days to find a valid close.
    """
    start = scan_date - timedelta(days=10)
    end   = scan_date + timedelta(days=1)
    hist  = tk.history(start=str(start), end=str(end), auto_adjust=True)
    if hist.empty:
        return None
    hist.index = hist.index.date
    valid = hist[hist.index <= scan_date]
    if valid.empty:
        return None
    return float(valid["Close"].iloc[-1])


def _get_historical_52w(tk, scan_date):
    """Returns 52-week high and low ending on scan_date."""
    start = scan_date - timedelta(days=370)
    end   = scan_date + timedelta(days=1)
    hist  = tk.history(start=str(start), end=str(end), auto_adjust=True)
    if hist.empty:
        return None, None
    hist.index = hist.index.date
    valid = hist[hist.index <= scan_date]
    if valid.empty:
        return None, None
    return float(valid["High"].max()), float(valid["Low"].min())


def _latest_col_before(df_fin, scan_date):
    """
    Given a yfinance financials DataFrame (rows=metrics, columns=dates),
    returns the column Series for the most recent date <= scan_date.
    """
    if df_fin is None or df_fin.empty:
        return None
    dates = pd.to_datetime(df_fin.columns).date
    valid = [(d, col) for d, col in zip(dates, df_fin.columns) if d <= scan_date]
    if not valid:
        return None
    # pick most recent
    _, best_col = max(valid, key=lambda x: x[0])
    return df_fin[best_col]


def _val(series, key):
    """Safely extract a numeric value from a pandas Series by partial key match."""
    if series is None:
        return None
    try:
        matches = [k for k in series.index if key.lower() in str(k).lower()]
        if matches:
            v = series[matches[0]]
            return float(v) if pd.notna(v) and np.isfinite(float(v)) else None
    except Exception:
        pass
    return None


def _growth(df_fin, metric_key, scan_date):
    """Computes YoY growth for a metric from financial history."""
    if df_fin is None or df_fin.empty:
        return None
    dates = sorted(pd.to_datetime(df_fin.columns).date, reverse=True)
    valid = [d for d in dates if d <= scan_date]
    if len(valid) < 2:
        return None
    try:
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
# SINGLE STOCK FETCH  (live or historical)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_stock_data(ticker, scan_date=None):
    """
    Fetches all required valuation parameters for one ticker.

    scan_date=None  -> live/current data (default)
    scan_date=date  -> historical mode: price on that date, financials from
                       most recent filing before that date.
    """
    is_backtest = scan_date is not None

    try:
        tk   = yf.Ticker(ticker)
        info = tk.info

        name     = info.get("longName") or info.get("shortName") or ticker
        sector   = info.get("sector",   "Unknown")
        industry = info.get("industry", "Unknown")
        shares   = info.get("sharesOutstanding")

        # ── PRICE ─────────────────────────────────────────────────────────────
        if is_backtest:
            price = _get_historical_price(tk, scan_date)
        else:
            price = info.get("currentPrice") or info.get("regularMarketPrice")

        if not price or price <= 0:
            return None

        # ── MARKET CAP ────────────────────────────────────────────────────────
        if is_backtest and shares and shares > 0:
            market_cap_inr = price * shares
        else:
            market_cap_inr = info.get("marketCap", 0) or 0

        market_cap_cr = market_cap_inr / CR_TO_INR
        if not (MARKET_CAP_MIN_CR <= market_cap_cr <= MARKET_CAP_MAX_CR):
            return None

        # ── FINANCIALS ────────────────────────────────────────────────────────
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

            eps_ttm    = (net_income / shares)  if net_income and shares and shares > 0 else None
            eps_fwd    = info.get("forwardEps")
            book_value = (equity     / shares)  if equity     and shares and shares > 0 else None

            op_cashflow = _val(cf_col, "Total Cash From Operating Activities") or _val(cf_col, "Operating Cash Flow")
            capex       = _val(cf_col, "Capital Expenditures")
            # capex is usually stored as negative in yfinance
            if op_cashflow and capex:
                fcf = op_cashflow + capex   # capex negative => subtraction
            else:
                fcf = op_cashflow

            current_ratio  = (curr_assets / curr_liab) if curr_assets and curr_liab and curr_liab > 0 else None
            debt_equity    = (total_debt  / equity * 100) if total_debt and equity and equity > 0 else None
            roe            = (net_income  / equity)       if net_income and equity  and equity  > 0 else None
            roa            = (net_income  / total_assets) if net_income and total_assets and total_assets > 0 else None

            roce = None
            if ebit and total_assets and curr_liab:
                cap_emp = total_assets - curr_liab
                if cap_emp > 0:
                    roce = ebit / cap_emp

            enterprise_val = market_cap_inr + (total_debt or 0) - (cash or 0)

            pe_ttm    = (price / eps_ttm)         if eps_ttm    and eps_ttm    > 0 else None
            pe_fwd    = info.get("forwardPE")
            pb        = (price / book_value)       if book_value and book_value > 0 else None
            ps        = (market_cap_inr / total_revenue) if total_revenue and total_revenue > 0 else None

            dep       = _val(cf_col, "Depreciation") if cf_col is not None else None
            ebitda    = (ebit + dep) if ebit and dep else ebit
            ev_ebitda = (enterprise_val / ebitda)      if ebitda and ebitda > 0 else None
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
            quick_ratio  = None   # not easily derived from annual statements

        else:
            # ── LIVE MODE ─────────────────────────────────────────────────────
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
            quick_ratio     = info.get("quickRatio")
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
            "ticker":           ticker.replace(".NS", ""),
            "name":             name,
            "sector":           sector,
            "industry":         industry,
            "scan_date":        str(scan_date) if scan_date else str(TODAY),
            "mode":             "backtest" if is_backtest else "live",
            "price":            round(price, 2),
            "market_cap_cr":    round(market_cap_cr, 1),
            "pe_ttm":           pe_ttm,
            "pe_fwd":           pe_fwd,
            "pb":               pb,
            "ps":               ps,
            "ev_ebitda":        ev_ebitda,
            "ev_revenue":       ev_revenue,
            "peg":              peg,
            "eps_ttm":          eps_ttm,
            "eps_fwd":          eps_fwd,
            "revenue_growth":   revenue_growth,
            "earnings_growth":  earnings_growth,
            "gross_margin":     gross_margin,
            "op_margin":        op_margin,
            "net_margin":       net_margin,
            "roe":              roe,
            "roa":              roa,
            "roce":             roce,
            "debt_equity":      debt_equity,
            "current_ratio":    current_ratio,
            "book_value":       book_value,
            "fcf":              fcf,
            "op_cashflow":      op_cashflow,
            "total_revenue":    total_revenue,
            "div_yield":        div_yield,
            "payout_ratio":     payout_ratio,
            "week52_high":      week52_high,
            "week52_low":       week52_low,
            "beta":             beta,
            "target_mean":      target_mean,
            "target_high":      target_high,
            "target_low":       target_low,
            "analyst_rec":      recommend,
            "num_analysts":     num_analysts,
            "enterprise_val":   enterprise_val,
            "total_debt":       total_debt,
            "cash":             cash,
            "shares_out":       shares,
        }

    except Exception as e:
        logger.debug(f"  {ticker}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST RETURN ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════════

def enrich_with_forward_price(df, scan_date, forward_days=365):
    """
    For backtesting: fetches the actual price N days after scan_date
    and computes actual_return vs the verdict at scan_date.
    Only calculates returns if scan_date + forward_days <= today.
    """
    fwd_date = scan_date + timedelta(days=forward_days)
    df["fwd_date"] = str(fwd_date)

    if fwd_date > TODAY:
        logger.info(f"Forward date {fwd_date} is still in the future -- skipping return calc.")
        df["price_fwd"]         = np.nan
        df["actual_return_pct"] = np.nan
        return df

    logger.info(f"Fetching forward prices as of {fwd_date} for actual return calculation ...")
    fwd_prices = {}
    for ticker in tqdm(df["ticker"].tolist(), desc="Fwd prices", unit="stock"):
        try:
            tk = yf.Ticker(f"{ticker}.NS")
            fwd_prices[ticker] = _get_historical_price(tk, fwd_date)
            time.sleep(0.2)
        except Exception:
            fwd_prices[ticker] = None

    df["price_fwd"]         = df["ticker"].map(fwd_prices)
    df["actual_return_pct"] = ((df["price_fwd"] - df["price"]) / df["price"] * 100).round(2)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# BATCH FETCH
# ══════════════════════════════════════════════════════════════════════════════

def fetch_all_stocks(tickers, scan_date=None, delay=0.4):
    """
    Iterates over all tickers, fetches data (live or historical),
    and filters by reconstructed market cap at scan_date.
    """
    results = []
    mode    = f"BACKTEST ({scan_date})" if scan_date else "LIVE (today)"
    logger.info(f"Scanning {len(tickers)} tickers | Mode: {mode} | MCap filter: Rs{MARKET_CAP_MIN_CR}-{MARKET_CAP_MAX_CR} cr")

    for ticker in tqdm(tickers, desc="Fetching", unit="stock"):
        data = fetch_stock_data(ticker, scan_date=scan_date)
        if data:
            results.append(data)
        time.sleep(delay)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    logger.info(f"Qualifying stocks found: {len(df)}")
    return df


def parse_scan_date(scan_date_str):
    """Public wrapper to parse and validate a scan date string."""
    return _parse_scan_date(scan_date_str)
