# 📈 NSE Fair Value Scanner

A professional-grade stock valuation scanner for **NSE mid-cap stocks (₹500 cr – ₹10,000 cr market cap)**.  
Runs entirely on **free data sources** (yfinance). No API keys required.

---

## 🎯 What It Does

Scans all NSE-listed stocks in the mid-cap range and computes a **blended fair value** using 4 models:

| Model | Weight | Description |
|---|---|---|
| **Relative Valuation** | 35% | Sector P/E, P/B, EV/EBITDA vs benchmarks |
| **DCF (Discounted Cash Flow)** | 25% | FCF-based, CAPM WACC, 5-yr projection + terminal value |
| **Graham Number** | 20% | `√(22.5 × EPS × Book Value)` — classic safety floor |
| **Analyst Consensus** | 20% | Mean analyst price target (when ≥2 analysts cover stock) |

Each stock is also scored on a **0–100 Quality Score** across:
- 📊 Profitability (ROE, Net Margin, ROCE)
- 🚀 Growth (Revenue & Earnings growth)
- 🏦 Balance Sheet (Debt/Equity, Current Ratio)
- 💵 Cash Flow (FCF positive, Operating CF)
- 💎 Valuation (PEG ratio, P/E vs sector)

---

## 🏷️ Verdict Categories

| Verdict | Margin of Safety | Quality Score |
|---|---|---|
| ⭐ **Strong Buy** | > 30% undervalued | ≥ 60 |
| ✅ **Buy** | > 15% undervalued | ≥ 45 |
| 🟡 **Hold** | Fair value ± 5% | Any |
| 🔴 **Avoid** | < 20% overvalued | Any |
| 🚫 **Strong Avoid** | > 20% overvalued | Any |

---

## 🚀 Running on GitHub (Recommended)

### Step 1 — Fork this repository
Click the **Fork** button on GitHub.

### Step 2 — Run the scanner manually
1. Go to **Actions** tab in your forked repo
2. Click **NSE Fair Value Scanner** in the left panel
3. Click **Run workflow** → fill in parameters → click green **Run workflow**
4. Download the **Excel report** from **Artifacts** when complete

---

### ⚙️ Workflow Parameters

| Parameter | Default | Description |
|---|---|---|
| `scan_date` | *(blank = live)* | **Backtest date** in `YYYY-MM-DD` format. Leave blank for today's live scan. E.g. `2023-01-01` |
| `forward_days` | `365` | Days after scan_date to measure actual return (backtest accuracy). E.g. `180`, `365`, `730` |
| `tickers` | *(blank = all NSE)* | Space-separated specific stocks. E.g. `INFY TCS HCLTECH` |
| `limit` | *(blank = all)* | First N tickers only. E.g. `50` for a quick test |
| `delay` | `0.5` | Seconds between API calls. Lower = faster but risk rate-limit |

---

### 📅 Backtest Examples

| Goal | scan_date | forward_days |
|---|---|---|
| Pre-COVID recovery picks | `2020-04-01` | `365` |
| Post-rate-hike undervalued stocks | `2022-10-01` | `365` |
| 2-year return validation | `2022-01-01` | `730` |
| 6-month short-term backtest | `2023-06-01` | `180` |

> **How backtest works:**
> 1. Fetches historical **closing price** on `scan_date`
> 2. Reconstructs market cap and pulls financials from the **most recent filing before** `scan_date`
> 3. Runs all 4 valuation models → assigns verdicts
> 4. Fetches actual price `forward_days` later
> 5. Adds **"📈 Backtest Accuracy"** sheet showing how each verdict category actually performed

---

## 💻 Running Locally

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/nse-fair-value-scanner.git
cd nse-fair-value-scanner

# Install dependencies
pip install -r requirements.txt

# Full scan
python scanner.py

# Quick test (first 30 tickers)
python scanner.py --limit 30

# Specific stocks
python scanner.py --tickers INFY TCS HCLTECH WIPRO

# Custom output folder
python scanner.py --output my_results/
```

---

## 📂 Output

The scanner produces:
1. **Excel report** (`output/nse_fair_value_YYYYMMDD_HHMM.xlsx`) with:
   - `📊 Summary` — all stocks sorted by verdict
   - `⭐ Strong Buy`, `✅ Buy`, etc. — individual sheets per category
   - `ℹ️ Metadata` — scan parameters & counts
2. **Console summary** — grouped table printed to terminal
3. **Log file** (`logs/scan_*.log`) — detailed execution log

### Excel columns explained
| Column | Description |
|---|---|
| `fair_value_base` | Blended fair value (base case) |
| `fair_value_bear` | Most conservative model's output |
| `fair_value_bull` | Most optimistic model's output |
| `margin_of_safety_pct` | `(Fair Value − Price) / Fair Value × 100` |
| `quality_score` | 0–100 composite quality score |
| `rv` | Relative valuation model output |
| `gv` | Graham Number |
| `dcf` | DCF fair value |
| `av` | Analyst consensus target |
| `models_used` | How many models had enough data |
| `roe / roce / net_margin` | Shown as **%** in Excel |

---

## ⚠️ Disclaimer

> This tool is for **educational and research purposes only**.  
> It is **not** financial advice. Always do your own due diligence before investing.  
> Fair value estimates are based on publicly available data and simplified models.  
> Past performance and model outputs do not guarantee future results.

---

## 🛠 Architecture

```
nse-fair-value-scanner/
├── scanner.py            ← Main entry point (CLI)
├── data_fetcher.py       ← NSE stock list + yfinance data pull
├── valuation_engine.py   ← All 4 models + blending + quality score
├── report_generator.py   ← Console table + Excel workbook
├── requirements.txt
├── .github/
│   └── workflows/
│       └── scanner.yml   ← GitHub Actions manual trigger
├── output/               ← Generated Excel reports
└── logs/                 ← Scan logs
```

---

## 📦 Dependencies (all free)

| Package | Purpose |
|---|---|
| `yfinance` | Stock data (price, financials, ratios) |
| `pandas` | Data manipulation |
| `numpy` | Numerical computations |
| `openpyxl` | Excel report generation |
| `tabulate` | Console table formatting |
| `tqdm` | Progress bar |
| `requests` | NSE stock list download |
| `beautifulsoup4` | HTML parsing (fallback) |
