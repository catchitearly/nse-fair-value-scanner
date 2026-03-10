"""
report_generator.py
--------------------
Generates a nicely formatted Excel report + console summary
from the valuation results DataFrame.
"""

import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VERDICT_ORDER = [
    "⭐ Strong Buy",
    "✅ Buy",
    "🟡 Hold",
    "🔴 Avoid",
    "🚫 Strong Avoid",
    "Insufficient Data",
]

# Columns for display
DISPLAY_COLS = [
    "ticker", "name", "sector",
    "scan_date", "mode",
    "price", "market_cap_cr",
    "fair_value_bear", "fair_value_base", "fair_value_bull",
    "margin_of_safety_pct", "quality_score", "verdict",
    "pe_ttm", "pe_fwd", "pb", "peg", "ev_ebitda",
    "roe", "roce", "net_margin", "op_margin",
    "revenue_growth", "earnings_growth",
    "debt_equity", "current_ratio",
    "div_yield", "beta",
    "week52_low", "week52_high",
    "pct_from_52w_high", "pct_from_52w_low",
    "rv", "gv", "dcf", "av", "models_used",
    "analyst_rec", "num_analysts", "target_mean",
    # backtest columns (NaN in live mode)
    "fwd_date", "price_fwd", "actual_return_pct",
]


def pct_fmt(v):
    if pd.isna(v): return ""
    return f"{v:.1f}%"

def num_fmt(v, dec=2):
    if pd.isna(v): return ""
    return f"{v:,.{dec}f}"


def print_summary(df: pd.DataFrame):
    """Prints a grouped summary table to console."""
    from tabulate import tabulate

    print("\n" + "═" * 90)
    print(f"  NSE MID-CAP FAIR VALUE SCANNER  |  {datetime.now().strftime('%d %b %Y  %H:%M')}")
    print(f"  Market cap range: ₹500 cr – ₹10,000 cr  |  Stocks scanned: {len(df)}")
    print("═" * 90)

    for verdict in VERDICT_ORDER:
        subset = df[df["verdict"] == verdict].copy()
        if subset.empty:
            continue

        subset = subset.sort_values("margin_of_safety_pct", ascending=False)

        print(f"\n{'─'*90}")
        print(f"  {verdict}  ({len(subset)} stocks)")
        print(f"{'─'*90}")

        rows = []
        for _, r in subset.iterrows():
            rows.append([
                r.get("ticker", ""),
                (r.get("name", "") or "")[:28],
                (r.get("sector", "") or "")[:18],
                num_fmt(r.get("price")),
                num_fmt(r.get("market_cap_cr"), 0),
                num_fmt(r.get("fair_value_base")),
                pct_fmt(r.get("margin_of_safety_pct")),
                num_fmt(r.get("quality_score"), 1),
                num_fmt(r.get("pe_fwd"), 1),
                pct_fmt((r.get("roe") or 0) * 100),
            ])

        headers = [
            "Ticker", "Name", "Sector",
            "Price ₹", "MCap cr", "Fair Val ₹", "MoS%",
            "Quality", "Fwd P/E", "ROE%"
        ]
        print(tabulate(rows, headers=headers, tablefmt="simple"))

    print("\n" + "═" * 90)
    print("  Legend: MoS = Margin of Safety  |  Fair Val = Blended Base Case")
    print("═" * 90 + "\n")


def save_excel(df: pd.DataFrame, output_dir: str = "output",
               scan_date=None, is_backtest: bool = False,
               forward_days: int = 365) -> str:
    """
    Saves a multi-sheet Excel workbook:
      - Summary (all stocks, sorted by verdict)
      - One sheet per verdict category
      - Metadata sheet
    """
    os.makedirs(output_dir, exist_ok=True)
    date_str  = datetime.now().strftime("%Y%m%d_%H%M")
    filepath  = os.path.join(output_dir, f"nse_fair_value_{date_str}.xlsx")

    # Subset to available display columns
    avail_cols = [c for c in DISPLAY_COLS if c in df.columns]
    df_out     = df[avail_cols].copy()

    # Format pct columns
    for col in ["roe", "roa", "roce", "net_margin", "op_margin",
                "gross_margin", "revenue_growth", "earnings_growth",
                "div_yield", "payout_ratio"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].apply(
                lambda x: round(x * 100, 2) if pd.notna(x) else x
            )

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        # ── Summary sheet ─────────────────────────────────────────────────────
        df_summary = df_out.copy()
        df_summary["_order"] = df_summary["verdict"].apply(
            lambda v: VERDICT_ORDER.index(v) if v in VERDICT_ORDER else 99
        )
        df_summary = df_summary.sort_values(
            ["_order", "margin_of_safety_pct"], ascending=[True, False]
        ).drop(columns=["_order"])

        df_summary.to_excel(writer, sheet_name="📊 Summary", index=False)
        _style_sheet(writer, "📊 Summary", df_summary)

        # ── Per-verdict sheets ────────────────────────────────────────────────
        for verdict in VERDICT_ORDER:
            subset = df_out[df_out["verdict"] == verdict].sort_values(
                "margin_of_safety_pct", ascending=False
            )
            if subset.empty:
                continue
            sheet_name = verdict[:31]   # Excel limit
            subset.to_excel(writer, sheet_name=sheet_name, index=False)
            _style_sheet(writer, sheet_name, subset)

        # ── Backtest accuracy sheet ───────────────────────────────────────────
        if is_backtest and "actual_return_pct" in df_out.columns:
            acc_rows = []
            for verdict in VERDICT_ORDER:
                subset = df_out[df_out["verdict"] == verdict]
                if subset.empty:
                    continue
                valid = subset[subset["actual_return_pct"].notna()]
                acc_rows.append({
                    "Verdict":          verdict,
                    "Stock Count":      len(subset),
                    "With Return Data": len(valid),
                    "Avg Return %":     round(valid["actual_return_pct"].mean(), 2) if len(valid) else "",
                    "Median Return %":  round(valid["actual_return_pct"].median(), 2) if len(valid) else "",
                    "Best Return %":    round(valid["actual_return_pct"].max(), 2) if len(valid) else "",
                    "Worst Return %":   round(valid["actual_return_pct"].min(), 2) if len(valid) else "",
                    "% Positive":       round((valid["actual_return_pct"] > 0).mean() * 100, 1) if len(valid) else "",
                    "% Beat Nifty (est)": "",   # placeholder for manual fill
                })
            if acc_rows:
                acc_df = pd.DataFrame(acc_rows)
                acc_df.to_excel(writer, sheet_name="📈 Backtest Accuracy", index=False)
                _style_sheet(writer, "📈 Backtest Accuracy", acc_df)

        # ── Metadata ─────────────────────────────────────────────────────────
        scan_date_str = str(scan_date) if scan_date else "Today (live)"
        fwd_date_str  = ""
        if is_backtest and "fwd_date" in df.columns:
            fwd_dates = df["fwd_date"].dropna().unique()
            fwd_date_str = fwd_dates[0] if len(fwd_dates) else ""

        meta = pd.DataFrame({
            "Parameter": [
                "Scan Date", "Scan Time", "Mode",
                "Forward Date (backtest)", "Forward Window (days)",
                "Market Cap Min (Rs cr)", "Market Cap Max (Rs cr)",
                "Total Stocks Scanned",
                "Strong Buy", "Buy", "Hold", "Avoid", "Strong Avoid",
                "Insufficient Data",
                "Risk-Free Rate", "Equity Risk Premium",
                "Model Weights (Relative / Graham / DCF / Analyst)",
            ],
            "Value": [
                scan_date_str,
                datetime.now().strftime("%H:%M:%S"),
                "BACKTEST" if is_backtest else "LIVE",
                fwd_date_str,
                forward_days if is_backtest else "N/A",
                500, 10000,
                len(df),
                len(df[df["verdict"].str.contains("Strong Buy",  na=False)]),
                len(df[df["verdict"].str.contains(r"^✅",        na=False, regex=True)]),
                len(df[df["verdict"].str.contains("Hold",        na=False)]),
                len(df[df["verdict"].str.contains(r"^🔴",        na=False, regex=True)]),
                len(df[df["verdict"].str.contains("Strong Avoid",na=False)]),
                len(df[df["verdict"].str.contains("Insufficient",na=False)]),
                "7.2%", "5.5%",
                "35% / 20% / 25% / 20%",
            ]
        })
        meta.to_excel(writer, sheet_name="ℹ️ Metadata", index=False)

    logger.info(f"Excel report saved → {filepath}")
    return filepath


def _style_sheet(writer, sheet_name: str, df: pd.DataFrame):
    """Applies basic column widths and header formatting."""
    try:
        from openpyxl.styles import PatternFill, Font, Alignment
        ws = writer.sheets[sheet_name]

        # Header row
        header_fill = PatternFill("solid", fgColor="1F3864")
        header_font = Font(color="FFFFFF", bold=True)
        for cell in ws[1]:
            cell.fill     = header_fill
            cell.font     = header_font
            cell.alignment= Alignment(horizontal="center", wrap_text=True)

        # Verdict colour coding
        fills = {
            "⭐ Strong Buy":  PatternFill("solid", fgColor="C6EFCE"),
            "✅ Buy":         PatternFill("solid", fgColor="E2EFDA"),
            "🟡 Hold":        PatternFill("solid", fgColor="FFEB9C"),
            "🔴 Avoid":       PatternFill("solid", fgColor="FFC7CE"),
            "🚫 Strong Avoid":PatternFill("solid", fgColor="FF4444"),
        }
        # Find verdict column index
        verdict_col = None
        for i, cell in enumerate(ws[1], 1):
            if cell.value == "verdict":
                verdict_col = i
                break

        if verdict_col:
            for row in ws.iter_rows(min_row=2):
                verdict_cell = row[verdict_col - 1]
                fill = fills.get(str(verdict_cell.value))
                if fill:
                    for cell in row:
                        cell.fill = fill

        # Auto column width
        for col in ws.columns:
            max_len = max(
                (len(str(c.value)) for c in col if c.value), default=10
            )
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 30)

        # Freeze header row
        ws.freeze_panes = "A2"

    except Exception as e:
        logger.debug(f"Styling error: {e}")


def generate_report(df: pd.DataFrame, output_dir: str = "output",
                    scan_date=None, is_backtest: bool = False,
                    forward_days: int = 365) -> str:
    """Main entry — prints console summary and saves Excel."""
    print_summary(df)
    return save_excel(df, output_dir,
                      scan_date=scan_date,
                      is_backtest=is_backtest,
                      forward_days=forward_days)
