"""
valuation_engine.py
--------------------
Computes fair value for each stock using a blended multi-model approach:
  1. Relative Valuation   (35 %)
  2. Graham Number        (20 %)
  3. DCF (simplified)     (25 %)
  4. Analyst Target       (20 %)

Then scores each stock on quality metrics and produces a
final verdict: Strong Buy / Buy / Hold / Avoid / Strong Avoid.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ── Sector median P/E benchmarks (NSE context) ────────────────────────────────
SECTOR_PE = {
    "Technology":              28,
    "Financial Services":      18,
    "Healthcare":              30,
    "Consumer Cyclical":       25,
    "Consumer Defensive":      35,
    "Industrials":             22,
    "Basic Materials":         14,
    "Energy":                  12,
    "Real Estate":             20,
    "Communication Services":  20,
    "Utilities":               16,
    "Unknown":                 20,
}

SECTOR_PB = {
    "Technology":        6,
    "Financial Services":2,
    "Healthcare":        5,
    "Consumer Cyclical": 4,
    "Consumer Defensive":7,
    "Industrials":       3,
    "Basic Materials":   2,
    "Energy":            2,
    "Real Estate":       2,
    "Communication Services": 3,
    "Utilities":         2,
    "Unknown":           3,
}

RISK_FREE_RATE  = 0.072   # 10-yr Indian G-Sec yield (~7.2 %)
EQUITY_PREMIUM  = 0.055   # India ERP
WACC_DEFAULT    = 0.12    # fallback WACC

# ── Model weights ─────────────────────────────────────────────────────────────
W_RELATIVE   = 0.35
W_GRAHAM     = 0.20
W_DCF        = 0.25
W_ANALYST    = 0.20


# ── Helper: safe float ────────────────────────────────────────────────────────
def sf(v, default=np.nan):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RELATIVE VALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def relative_fair_value(row: pd.Series) -> float:
    """
    Blends P/E-implied and P/B-implied fair values using sector medians.
    """
    sector      = row.get("sector", "Unknown")
    sector_pe   = SECTOR_PE.get(sector, 20)
    sector_pb   = SECTOR_PB.get(sector, 3)

    eps         = sf(row.get("eps_fwd")) or sf(row.get("eps_ttm"))
    book_value  = sf(row.get("book_value"))
    price       = sf(row.get("price"), 0)

    estimates = []

    # P/E implied
    if eps and eps > 0:
        pe_fair = eps * sector_pe
        estimates.append(pe_fair)

    # P/B implied
    if book_value and book_value > 0:
        pb_fair = book_value * sector_pb
        estimates.append(pb_fair)

    # EV/EBITDA implied (rough)
    ev_ebitda   = sf(row.get("ev_ebitda"))
    shares      = sf(row.get("shares_out"))
    market_cap  = sf(row.get("market_cap_cr", 0)) * 1e7
    if ev_ebitda and shares and shares > 0 and market_cap > 0:
        # back into EBITDA from EV/EBITDA and enterprise value
        ev = sf(row.get("enterprise_val"))
        if ev and ev > 0:
            ebitda = ev / ev_ebitda
            # target EV at 10x EBITDA for mid-caps
            fair_ev    = ebitda * 10
            debt       = sf(row.get("total_debt"), 0)
            cash       = sf(row.get("cash"), 0)
            fair_equity= fair_ev - debt + cash
            if fair_equity > 0:
                per_share  = fair_equity / shares
                estimates.append(per_share)

    return float(np.mean(estimates)) if estimates else np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GRAHAM NUMBER
# ═══════════════════════════════════════════════════════════════════════════════

def graham_fair_value(row: pd.Series) -> float:
    """
    Graham Number = sqrt(22.5 × EPS × Book Value per Share)
    Classic intrinsic value floor — conservative.
    """
    eps    = sf(row.get("eps_ttm"))
    bv     = sf(row.get("book_value"))
    if eps and bv and eps > 0 and bv > 0:
        return float(np.sqrt(22.5 * eps * bv))
    return np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIMPLIFIED DCF
# ═══════════════════════════════════════════════════════════════════════════════

def dcf_fair_value(row: pd.Series) -> float:
    """
    Simplified DCF:
      - Uses FCF/share as base cash flow
      - Projects 5 years at observed growth rate (capped 5–25 %)
      - Terminal value at 3 % perpetual growth
      - Discount at WACC (beta-adjusted)
    """
    shares   = sf(row.get("shares_out"))
    fcf      = sf(row.get("fcf"))
    beta     = sf(row.get("beta"), 1.0)

    if not shares or shares <= 0 or not fcf or fcf <= 0:
        return np.nan

    fcf_per_share = fcf / shares

    # Growth rate: use earnings_growth, cap between 5 % and 25 %
    g_raw    = sf(row.get("earnings_growth")) or sf(row.get("revenue_growth")) or 0.10
    g        = max(0.05, min(0.25, g_raw))

    # WACC (CAPM)
    beta     = max(0.5, min(2.5, beta))
    wacc     = RISK_FREE_RATE + beta * EQUITY_PREMIUM
    wacc     = max(0.10, min(0.20, wacc))

    # Terminal growth
    g_term   = 0.03

    # 5-year DCF
    pv = 0.0
    cf = fcf_per_share
    for yr in range(1, 6):
        cf   *= (1 + g)
        pv   += cf / (1 + wacc) ** yr

    # Terminal value (Gordon Growth)
    tv   = (cf * (1 + g_term)) / (wacc - g_term)
    pv  += tv / (1 + wacc) ** 5

    return float(pv)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ANALYST CONSENSUS TARGET
# ═══════════════════════════════════════════════════════════════════════════════

def analyst_fair_value(row: pd.Series) -> float:
    """
    Uses analyst mean target price. Falls back to low/high midpoint.
    Only used if ≥ 2 analyst opinions exist.
    """
    n    = sf(row.get("num_analysts"), 0)
    mean = sf(row.get("target_mean"))
    low  = sf(row.get("target_low"))
    high = sf(row.get("target_high"))

    if n >= 2 and mean and mean > 0:
        return float(mean)
    if low and high and low > 0 and high > 0:
        return float((low + high) / 2)
    return np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# BLENDED FAIR VALUE
# ═══════════════════════════════════════════════════════════════════════════════

def blended_fair_value(row: pd.Series) -> dict:
    """
    Weighted blend of all four models.
    Returns bear/base/bull case and final blended value.
    """
    rv  = relative_fair_value(row)
    gv  = graham_fair_value(row)
    dcf = dcf_fair_value(row)
    av  = analyst_fair_value(row)

    models = {
        "relative":  (rv,  W_RELATIVE),
        "graham":    (gv,  W_GRAHAM),
        "dcf":       (dcf, W_DCF),
        "analyst":   (av,  W_ANALYST),
    }

    valid_vals  = [(v, w) for v, w in models.values() if np.isfinite(v) and v > 0]
    if not valid_vals:
        return {
            "fair_value_bear": np.nan,
            "fair_value_base": np.nan,
            "fair_value_bull": np.nan,
            "models_used":     0,
            "rv": np.nan, "gv": np.nan, "dcf": np.nan, "av": np.nan,
        }

    # Re-normalise weights for available models
    total_w = sum(w for _, w in valid_vals)
    base    = sum(v * w / total_w for v, w in valid_vals)
    vals    = [v for v, _ in valid_vals]

    bear    = min(vals)
    bull    = max(vals)

    return {
        "fair_value_bear": round(bear,  2),
        "fair_value_base": round(base,  2),
        "fair_value_bull": round(bull,  2),
        "models_used":     len(valid_vals),
        "rv":  round(rv,  2) if np.isfinite(rv)  else np.nan,
        "gv":  round(gv,  2) if np.isfinite(gv)  else np.nan,
        "dcf": round(dcf, 2) if np.isfinite(dcf) else np.nan,
        "av":  round(av,  2) if np.isfinite(av)  else np.nan,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY SCORE  (0 – 100)
# ═══════════════════════════════════════════════════════════════════════════════

def quality_score(row: pd.Series) -> float:
    """
    Scores stock quality across 5 pillars (each 0–20):
      1. Profitability  (ROE, net margin, ROCE)
      2. Growth         (revenue & earnings growth)
      3. Balance Sheet  (D/E, current ratio)
      4. Cash Flow      (FCF positive, op cashflow)
      5. Valuation      (PEG, P/E vs sector)
    """
    score = 0.0

    # ── 1. Profitability ─────────────────────────────────────────
    roe        = sf(row.get("roe"), 0)
    net_margin = sf(row.get("net_margin"), 0)
    roce       = sf(row.get("roce"), 0)

    if roe   > 0.20: score += 7
    elif roe > 0.12: score += 4
    elif roe > 0.05: score += 1

    if net_margin > 0.15: score += 7
    elif net_margin > 0.08: score += 4
    elif net_margin > 0.02: score += 1

    if roce > 0.20: score += 6
    elif roce > 0.12: score += 3

    # ── 2. Growth ────────────────────────────────────────────────
    rev_g = sf(row.get("revenue_growth"),  0)
    ear_g = sf(row.get("earnings_growth"), 0)

    if rev_g > 0.20: score += 10
    elif rev_g > 0.10: score += 6
    elif rev_g > 0.05: score += 3

    if ear_g > 0.20: score += 10
    elif ear_g > 0.10: score += 6
    elif ear_g > 0.05: score += 3

    # ── 3. Balance Sheet ─────────────────────────────────────────
    de = sf(row.get("debt_equity"), 999)   # yfinance gives % (e.g. 50 = 0.5x)
    cr = sf(row.get("current_ratio"), 0)

    de_ratio = de / 100 if de > 10 else de  # normalise
    if de_ratio < 0.3:   score += 10
    elif de_ratio < 0.7: score += 6
    elif de_ratio < 1.5: score += 2

    if cr > 2.0:   score += 10
    elif cr > 1.5: score += 6
    elif cr > 1.0: score += 3

    # ── 4. Cash Flow ─────────────────────────────────────────────
    fcf     = sf(row.get("fcf"), 0)
    op_cf   = sf(row.get("op_cashflow"), 0)

    if fcf > 0:   score += 10
    if op_cf > 0: score += 10

    # ── 5. Valuation ─────────────────────────────────────────────
    peg    = sf(row.get("peg"))
    pe_fwd = sf(row.get("pe_fwd"))
    sector = row.get("sector", "Unknown")
    s_pe   = SECTOR_PE.get(sector, 20)

    if peg and 0 < peg < 1:    score += 10
    elif peg and 1 <= peg < 2: score += 5

    if pe_fwd and pe_fwd > 0:
        discount = (s_pe - pe_fwd) / s_pe
        if discount > 0.30:   score += 10
        elif discount > 0.10: score += 5

    return min(100.0, round(score, 1))


# ═══════════════════════════════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════════════════════════════

def compute_verdict(price: float, base: float, bear: float, bull: float,
                    q_score: float, models_used: int) -> tuple[str, float]:
    """
    Returns (verdict_label, margin_of_safety_%).
    """
    if not (np.isfinite(base) and base > 0) or models_used < 2:
        return ("Insufficient Data", np.nan)

    mos = (base - price) / base * 100   # +ve = undervalued

    # Adjust for quality
    bonus = (q_score - 50) / 50 * 5     # ±5 % quality tilt on MoS threshold

    if   mos > (30 + bonus) and q_score >= 60:  verdict = "⭐ Strong Buy"
    elif mos > (15 + bonus) and q_score >= 45:  verdict = "✅ Buy"
    elif mos > (-5 + bonus):                     verdict = "🟡 Hold"
    elif mos > (-20 + bonus):                    verdict = "🔴 Avoid"
    else:                                         verdict = "🚫 Strong Avoid"

    return (verdict, round(mos, 1))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

def run_valuation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all models to the DataFrame and returns enriched results.
    """
    logger.info("Running valuation models …")
    records = []

    for _, row in df.iterrows():
        fv      = blended_fair_value(row)
        qs      = quality_score(row)
        price   = sf(row.get("price"), 0)

        verdict, mos = compute_verdict(
            price,
            fv["fair_value_base"],
            fv["fair_value_bear"],
            fv["fair_value_bull"],
            qs,
            fv["models_used"],
        )

        rec = row.to_dict()
        rec.update(fv)
        rec["quality_score"] = qs
        rec["verdict"]       = verdict
        rec["margin_of_safety_pct"] = mos

        # % from 52-week low/high
        w52h = sf(row.get("week52_high"))
        w52l = sf(row.get("week52_low"))
        if w52h and w52h > 0:
            rec["pct_from_52w_high"] = round((price - w52h) / w52h * 100, 1)
        if w52l and w52l > 0:
            rec["pct_from_52w_low"]  = round((price - w52l) / w52l * 100, 1)

        records.append(rec)

    result = pd.DataFrame(records)
    logger.info("Valuation complete.")
    return result
