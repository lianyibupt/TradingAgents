"""Financial Modeling Prep (FMP) data provider for TradingAgents.

Provides stock price data, fundamentals, and financial statements via
FMP's stable API (https://financialmodelingprep.com/stable/...).

API Key: set FMP_API_KEY in .env or environment.
"""

from __future__ import annotations

import os
import csv
import io
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests

from .symbol_utils import NoMarketDataError
import re
import pandas as pd


logger = logging.getLogger(__name__)


def _ak_module():
    import akshare as ak

    return ak

# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

_FMP_BASE = "https://financialmodelingprep.com/stable"
_FMP_API_KEY_ENV = "FMP_API_KEY"

# Simple in-memory rate limiter: 5 calls per 60 seconds (free tier)
_last_call_timestamps: list[float] = []


def _get_api_key() -> str:
    key = os.environ.get(_FMP_API_KEY_ENV, "")
    if not key:
        raise ValueError(
            f"FMP API key not set. Set {_FMP_API_KEY_ENV}=your_key in .env"
        )
    return key


def _rate_limit():
    """Ensure we don't exceed FMP free-tier rate limits (~5 req/min)."""
    global _last_call_timestamps
    now = time.monotonic()
    # Prune timestamps older than 60 seconds
    _last_call_timestamps = [t for t in _last_call_timestamps if now - t < 60]
    if len(_last_call_timestamps) >= 5:
        sleep_for = 60 - (now - _last_call_timestamps[0])
        if sleep_for > 0:
            logger.debug("FMP rate limit: sleeping %.1fs", sleep_for)
            time.sleep(sleep_for)
    _last_call_timestamps.append(time.monotonic())


def _fmp_get(endpoint: str, params: dict | None = None) -> list | dict:
    """Make a GET request to the FMP stable API with rate limiting and retry."""
    _rate_limit()
    url = f"{_FMP_BASE}/{endpoint.lstrip('/')}"
    req_params = {"apikey": _get_api_key()}
    if params:
        req_params.update(params)

    max_retries = 3
    for attempt in range(max_retries):
        resp = requests.get(url, params=req_params, timeout=30)
        if resp.status_code == 429:
            wait = 60 * (attempt + 1)
            logger.warning("FMP rate limit hit (attempt %d/%d), waiting %ds...",
                           attempt + 1, max_retries, wait)
            time.sleep(wait)
            continue
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        return resp.json()

    # All retries exhausted
    logger.error("FMP rate limit: all %d retries exhausted for %s", max_retries, endpoint)
    return []


def _to_csv_string(data: list[dict], header_lines: list[str] | None = None) -> str:
    """Convert a list of dicts to a CSV string, matching yfinance output format."""
    if not data:
        return ""

    output = io.StringIO()
    if header_lines:
        for line in header_lines:
            output.write(f"# {line}\n")
        output.write("\n")

    writer = csv.DictWriter(output, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue()


# ---------------------------------------------------------------------------
# Stock price data
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Akshare fallback (when FMP is rate-limited or unavailable)
# ---------------------------------------------------------------------------

def _akshare_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
) -> str:
    """Fetch stock data via akshare as fallback.

    Supports US stocks (stock_us_daily), A-shares (stock_zh_a_hist),
    and indices (stock_zh_index_daily_em).
    """
    sym = symbol.upper()
    df = pd.DataFrame()

    # US stocks: 1-5 letter symbols
    if re.match(r'^[A-Z]{1,5}$', sym):
        try:
            ak = _ak_module()
            df = ak.stock_us_daily(symbol=sym)
        except Exception:
            pass

    # A-shares: 6-digit codes
    elif re.match(r'^\d{6}$', sym):
        try:
            ak = _ak_module()
            df = ak.stock_zh_a_hist(
                symbol=sym, period="daily",
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust="",
            )
        except Exception:
            pass

    # Try generic index function
    if df.empty:
        try:
            ak = _ak_module()
            df = ak.stock_zh_index_daily_em(symbol=sym)
        except Exception:
            pass

    if df.empty:
        raise NoMarketDataError(symbol, sym, "no data from akshare fallback")

    # Normalize columns
    col_map = {
        'date': 'Date', '日期': 'Date',
        'open': 'Open', '开盘': 'Open',
        'high': 'High', '最高': 'High',
        'low': 'Low', '最低': 'Low',
        'close': 'Close', '收盘': 'Close',
        'volume': 'Volume', '成交量': 'Volume',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Filter by date range
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # Round numeric columns
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col].round(2)

    csv_string = df.to_csv(index=False)
    header = (
        f"# Stock data for {sym} from {start_date} to {end_date}\n"
        f"# Source: akshare (fallback)\n"
        f"# Total records: {len(df)}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string

def get_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
) -> str:
    """Retrieve OHLCV stock price data from FMP, with akshare fallback.

    Uses the /stable/historical-price/<symbol> endpoint first.
    Falls back to akshare when FMP is rate-limited or unavailable.
    Returns data as CSV string matching the yfinance output format.
    """
    # Try FMP first
    try:
        data = _fmp_get(
            f"historical-price/{symbol.upper()}",
            {"from": start_date, "to": end_date},
        )
        if data and isinstance(data, list) and len(data) > 0:
            # FMP returns: date, open, high, low, close, volume, adjClose, etc.
            rows = []
            for row in data:
                rows.append({
                    "Date": row.get("date", ""),
                    "Open": round(float(row.get("open", 0)), 2),
                    "High": round(float(row.get("high", 0)), 2),
                    "Low": round(float(row.get("low", 0)), 2),
                    "Close": round(float(row.get("close", 0)), 2),
                    "Adj Close": round(float(row.get("adjClose", row.get("close", 0))), 2),
                    "Volume": int(row.get("volume", 0)),
                })

            header = [
                f"Stock data for {symbol.upper()} from {start_date} to {end_date}",
                f"Source: Financial Modeling Prep",
                f"Total records: {len(rows)}",
                f"Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ]
            return _to_csv_string(rows, header)
    except Exception:
        pass

    # Fallback to akshare
    logger.info("FMP unavailable for %s, falling back to akshare", symbol)
    return _akshare_stock_data(symbol, start_date, end_date)


# ---------------------------------------------------------------------------
# Technical indicators (computed from FMP price data)
# ---------------------------------------------------------------------------

def get_indicators(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int = 30,
) -> str:
    """Compute a single technical indicator from FMP price data.

    FMP doesn't provide indicator endpoints on the free tier, so we
    fetch historical prices and compute basic indicators locally.
    """
    # Calculate date range
    try:
        end = datetime.strptime(curr_date, "%Y-%m-%d")
    except ValueError:
        end = datetime.now()
    start = (end - timedelta(days=look_back_days + 10)).strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    # Fetch price data
    try:
        data = _fmp_get(
            f"historical-price/{symbol.upper()}",
            {"from": start, "to": end_str},
        )
    except Exception as exc:
        raise NoMarketDataError(symbol, symbol.upper(), str(exc)) from exc

    if not data or not isinstance(data, list) or len(data) < 2:
        return f"# Insufficient price data to compute {indicator} for {symbol}"

    # Sort by date ascending
    data.sort(key=lambda r: r.get("date", ""))

    closes = [float(r["close"]) for r in data if r.get("close")]
    highs = [float(r["high"]) for r in data if r.get("high")]
    lows = [float(r["low"]) for r in data if r.get("low")]
    dates = [r["date"] for r in data if r.get("date")]

    ind = indicator.strip().lower()
    result_lines = [f"# {indicator.upper()} for {symbol.upper()}"]
    result_lines.append(f"# Data from FMP, computed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    result_lines.append(f"# Period: {start} to {end_str}")
    result_lines.append("")

    if ind in ("close_50_sma", "50_sma", "sma_50"):
        if len(closes) >= 50:
            sma = sum(closes[-50:]) / 50
            result_lines.append(f"close_50_sma: {sma:.2f}")
        else:
            result_lines.append(f"close_50_sma: insufficient data (need 50, have {len(closes)})")

    elif ind in ("close_200_sma", "200_sma", "sma_200"):
        if len(closes) >= 200:
            sma = sum(closes[-200:]) / 200
            result_lines.append(f"close_200_sma: {sma:.2f}")
        else:
            result_lines.append(f"close_200_sma: insufficient data (need 200, have {len(closes)})")

    elif ind in ("close_10_ema", "10_ema", "ema_10"):
        if len(closes) >= 10:
            k = 2 / (10 + 1)
            ema = closes[-10]
            for price in closes[-9:]:
                ema = price * k + ema * (1 - k)
            result_lines.append(f"close_10_ema: {ema:.2f}")
        else:
            result_lines.append(f"close_10_ema: insufficient data")

    elif ind in ("rsi", "rsi_14"):
        if len(closes) >= 15:
            gains, losses = [], []
            for i in range(1, 15):
                diff = closes[-i] - closes[-i - 1]
                gains.append(max(diff, 0))
                losses.append(max(-diff, 0))
            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            result_lines.append(f"rsi_14: {rsi:.2f}")
        else:
            result_lines.append(f"rsi_14: insufficient data")

    elif ind in ("macd",):
        if len(closes) >= 26:
            # Fast EMA (12)
            k12 = 2 / (12 + 1)
            ema12 = closes[-12]
            for price in closes[-11:]:
                ema12 = price * k12 + ema12 * (1 - k12)
            # Slow EMA (26)
            k26 = 2 / (26 + 1)
            ema26 = closes[-26]
            for price in closes[-25:]:
                ema26 = price * k26 + ema26 * (1 - k26)
            macd_line = ema12 - ema26
            # Signal (9-day EMA of MACD)
            k9 = 2 / (9 + 1)
            signal = macd_line  # simplified
            histogram = macd_line - signal
            result_lines.append(f"macd: {macd_line:.2f}")
            result_lines.append(f"macds (signal): {signal:.2f}")
            result_lines.append(f"macdh (histogram): {histogram:.2f}")
        else:
            result_lines.append(f"macd: insufficient data")

    elif ind in ("boll", "bollinger", "bollinger_bands"):
        if len(closes) >= 20:
            sma20 = sum(closes[-20:]) / 20
            variance = sum((c - sma20) ** 2 for c in closes[-20:]) / 20
            std = variance ** 0.5
            result_lines.append(f"boll (middle): {sma20:.2f}")
            result_lines.append(f"boll_ub (upper): {sma20 + 2 * std:.2f}")
            result_lines.append(f"boll_lb (lower): {sma20 - 2 * std:.2f}")
        else:
            result_lines.append(f"bollinger_bands: insufficient data")

    elif ind in ("volume",):
        volumes = [int(r.get("volume", 0)) for r in data if r.get("volume")]
        if volumes:
            avg_vol = sum(volumes) / len(volumes)
            result_lines.append(f"average_volume: {avg_vol:.0f}")
            result_lines.append(f"latest_volume: {volumes[-1]}")
        else:
            result_lines.append("volume: no data")

    else:
        result_lines.append(f"Indicator '{indicator}' not directly available from FMP.")
        result_lines.append(f"Available: rsi, macd, close_50_sma, close_200_sma, close_10_ema, boll, volume")

    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

def _get_financial_statement(endpoint: str, symbol: str, freq: str = "quarterly") -> list:
    """Fetch a financial statement from FMP."""
    period = "Q4" if freq.lower() == "annual" else "Q2"  # Q2 = latest quarter
    data = _fmp_get(
        f"{endpoint}",
        {"symbol": symbol.upper(), "period": "quarter" if freq.lower() == "quarterly" else "annual"},
    )
    if not isinstance(data, list):
        return []
    return data


def get_fundamentals(symbol: str) -> str:
    """Get company fundamentals from FMP profile + key metrics."""
    symbol = symbol.upper()

    # Get profile
    profile_data = _fmp_get("profile", {"symbol": symbol})
    if not profile_data or not isinstance(profile_data, list) or not profile_data:
        raise NoMarketDataError(symbol, symbol, "no profile data")

    profile = profile_data[0]

    # Get key metrics
    metrics_data = _fmp_get("key-metrics", {"symbol": symbol})

    lines = [
        f"# Company Fundamentals for {symbol}",
        f"# Source: Financial Modeling Prep",
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Company Name:** {profile.get('companyName', 'N/A')}",
        f"**Sector:** {profile.get('sector', 'N/A')}",
        f"**Industry:** {profile.get('industry', 'N/A')}",
        f"**Exchange:** {profile.get('exchangeFullName', profile.get('exchange', 'N/A'))}",
        f"**Currency:** {profile.get('currency', 'USD')}",
        f"**Country:** {profile.get('country', 'N/A')}",
        f"**Description:** {profile.get('description', 'N/A')}",
        f"**Market Cap:** {profile.get('marketCap', 'N/A')}",
        f"**Beta:** {profile.get('beta', 'N/A')}",
        f"**Price:** {profile.get('price', 'N/A')}",
        f"**Volume (avg):** {profile.get('averageVolume', 'N/A')}",
        f"**52W High:** {profile.get('range', 'N/A')}",
        f"**Last Dividend:** {profile.get('lastDividend', 'N/A')}",
        f"**CEO:** {profile.get('ceo', 'N/A')}",
        f"**Employees:** {profile.get('fullTimeEmployees', 'N/A')}",
        f"**Website:** {profile.get('website', 'N/A')}",
    ]

    if metrics_data and isinstance(metrics_data, list) and metrics_data:
        m = metrics_data[0]
        lines.extend([
            "",
            "**Key Metrics (latest):**",
            f"  Revenue per Share: {m.get('revenuePerShare', 'N/A')}",
            f"  Net Income per Share: {m.get('netIncomePerShare', 'N/A')}",
            f"  Free Cash Flow per Share: {m.get('freeCashFlowPerShare', 'N/A')}",
            f"  P/E Ratio: {m.get('peRatio', 'N/A')}",
            f"  P/B Ratio: {m.get('pbRatio', 'N/A')}",
            f"  P/S Ratio: {m.get('psRatio', 'N/A')}",
            f"  ROE: {m.get('roe', 'N/A')}",
            f"  ROA: {m.get('roa', 'N/A')}",
            f"  Debt to Equity: {m.get('debtToEquity', 'N/A')}",
            f"  Current Ratio: {m.get('currentRatio', 'N/A')}",
            f"  Dividend Yield: {m.get('dividendYield', 'N/A')}",
            f"  Earnings Yield: {m.get('earningsYield', 'N/A')}",
        ])

    return "\n".join(lines)


def get_balance_sheet(symbol: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """Get balance sheet from FMP."""
    data = _get_financial_statement("balance-sheet-statement", symbol, freq)
    if not data:
        raise NoMarketDataError(symbol, symbol.upper(), "no balance sheet data")

    rows = []
    for entry in data[:8]:  # Last 8 quarters
        rows.append({
            "Date": entry.get("date", ""),
            "Cash and Equivalents": entry.get("cashAndCashEquivalents", 0),
            "Short Term Investments": entry.get("shortTermInvestments", 0),
            "Accounts Receivable": entry.get("netReceivables", 0),
            "Inventory": entry.get("inventory", 0),
            "Total Current Assets": entry.get("totalCurrentAssets", 0),
            "Property Plant Equipment": entry.get("propertyPlantEquipmentNet", 0),
            "Goodwill": entry.get("goodwill", 0),
            "Total Assets": entry.get("totalAssets", 0),
            "Accounts Payable": entry.get("accountPayables", 0),
            "Short Term Debt": entry.get("shortTermDebt", 0),
            "Total Current Liabilities": entry.get("totalCurrentLiabilities", 0),
            "Long Term Debt": entry.get("longTermDebt", 0),
            "Total Liabilities": entry.get("totalLiabilities", 0),
            "Total Stockholders Equity": entry.get("totalStockholdersEquity", 0),
            "Total Equity": entry.get("totalEquity", 0),
        })

    header = [
        f"Balance Sheet data for {symbol.upper()} ({freq})",
        f"Source: Financial Modeling Prep",
        f"Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    return _to_csv_string(rows, header)


def get_cashflow(symbol: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """Get cash flow statement from FMP."""
    data = _get_financial_statement("cash-flow-statement", symbol, freq)
    if not data:
        raise NoMarketDataError(symbol, symbol.upper(), "no cash flow data")

    rows = []
    for entry in data[:8]:
        rows.append({
            "Date": entry.get("date", ""),
            "Net Income": entry.get("netIncome", 0),
            "Depreciation & Amortization": entry.get("depreciationAndAmortization", 0),
            "Stock Based Compensation": entry.get("stockBasedCompensation", 0),
            "Operating Cash Flow": entry.get("operatingCashFlow", 0),
            "Capital Expenditure": entry.get("capitalExpenditure", 0),
            "Free Cash Flow": entry.get("freeCashFlow", 0),
            "Dividends Paid": entry.get("dividendsPaid", 0),
            "Stock Repurchase": entry.get("commonStockRepurchased", 0),
            "Investing Cash Flow": entry.get("investingCashFlow", 0),
            "Financing Cash Flow": entry.get("financingCashFlow", 0),
            "Cash at End of Period": entry.get("cashAndCashEquivalents", 0),
        })

    header = [
        f"Cash Flow data for {symbol.upper()} ({freq})",
        f"Source: Financial Modeling Prep",
        f"Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    return _to_csv_string(rows, header)


def get_income_statement(symbol: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """Get income statement from FMP."""
    data = _get_financial_statement("income-statement", symbol, freq)
    if not data:
        raise NoMarketDataError(symbol, symbol.upper(), "no income statement data")

    rows = []
    for entry in data[:8]:
        rows.append({
            "Date": entry.get("date", ""),
            "Revenue": entry.get("revenue", 0),
            "Cost of Revenue": entry.get("costOfRevenue", 0),
            "Gross Profit": entry.get("grossProfit", 0),
            "R&D Expenses": entry.get("researchAndDevelopmentExpenses", 0),
            "SG&A Expenses": entry.get("sellingGeneralAndAdministrativeExpenses", 0),
            "Operating Expenses": entry.get("operatingExpenses", 0),
            "Operating Income": entry.get("operatingIncome", 0),
            "Interest Expense": entry.get("interestExpense", 0),
            "Income Before Tax": entry.get("incomeBeforeTax", 0),
            "Income Tax": entry.get("incomeTaxExpense", 0),
            "Net Income": entry.get("netIncome", 0),
            "EPS (Diluted)": entry.get("epsdiluted", entry.get("eps", 0)),
            "Weighted Avg Shares": entry.get("weightedAverageShsOut", 0),
        })

    header = [
        f"Income Statement data for {symbol.upper()} ({freq})",
        f"Source: Financial Modeling Prep",
        f"Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    return _to_csv_string(rows, header)


# ---------------------------------------------------------------------------
# News & insider transactions (not available on free FMP tier)
# ---------------------------------------------------------------------------

def get_news(symbol: str) -> str:
    """FMP free tier does not include news. Return informative message."""
    return (
        f"# News data not available from FMP on the current subscription plan.\n"
        f"# Upgrade your FMP plan or use yfinance for news data."
    )


def get_global_news() -> str:
    """FMP free tier does not include global news."""
    return (
        f"# Global news not available from FMP on the current subscription plan.\n"
        f"# Upgrade your FMP plan or use yfinance for news data."
    )


def get_insider_transactions(symbol: str) -> str:
    """FMP free tier does not include insider transactions."""
    return (
        f"# Insider transactions not available from FMP on the current subscription plan.\n"
        f"# Upgrade your FMP plan or use yfinance for insider data."
    )
