"""Akshare data provider for TradingAgents.

Provides stock price data via akshare (东方财富数据接口).
Supports US stocks, A-shares, and indices.
"""

from __future__ import annotations

import re
import logging
from datetime import datetime
from typing import Optional

import akshare as ak
import pandas as pd

from .symbol_utils import NoMarketDataError

logger = logging.getLogger(__name__)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize akshare column names to standard OHLCV format."""
    col_map = {
        "date": "Date",
        "日期": "Date",
        "open": "Open",
        "开盘": "Open",
        "high": "High",
        "最高": "High",
        "low": "Low",
        "最低": "Low",
        "close": "Close",
        "收盘": "Close",
        "volume": "Volume",
        "成交量": "Volume",
        "amount": "Volume",
        "成交额": "Volume",
    }
    return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})


def get_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
) -> str:
    """Retrieve OHLCV stock price data via akshare.

    Supports:
      - US stocks (1-5 letter symbols) via stock_us_daily
      - A-shares (6-digit codes) via stock_zh_a_hist
      - Indices via stock_zh_index_daily_em

    Returns data as CSV string.
    """
    sym = symbol.upper()
    df = pd.DataFrame()

    # US stocks: 1-5 letter symbols
    if re.match(r"^[A-Z]{1,5}$", sym):
        try:
            df = ak.stock_us_daily(symbol=sym)
            logger.info("akshare: fetched US stock data for %s", sym)
        except Exception as e:
            logger.warning("akshare stock_us_daily failed for %s: %s", sym, e)

    # A-shares: 6-digit codes
    elif re.match(r"^\d{6}$", sym):
        try:
            df = ak.stock_zh_a_hist(
                symbol=sym,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="",
            )
            logger.info("akshare: fetched A-share data for %s", sym)
        except Exception as e:
            logger.warning("akshare stock_zh_a_hist failed for %s: %s", sym, e)

    # Try generic index function for anything else
    if df.empty:
        try:
            df = ak.stock_zh_index_daily_em(symbol=sym)
            logger.info("akshare: fetched index data for %s", sym)
        except Exception as e:
            logger.warning("akshare index fetch failed for %s: %s", sym, e)

    if df.empty:
        raise NoMarketDataError(symbol, sym, "no data from akshare")

    # Normalize columns
    df = _normalize_columns(df)

    # Filter by date range
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    if df.empty:
        raise NoMarketDataError(
            symbol, sym, f"no rows between {start_date} and {end_date}"
        )

    # Round numeric columns
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    csv_string = df.to_csv(index=False)
    header = (
        f"# Stock data for {sym} from {start_date} to {end_date}\n"
        f"# Source: akshare\n"
        f"# Total records: {len(df)}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_string


def get_indicators(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int = 30,
) -> str:
    """Compute technical indicators from akshare price data."""
    from datetime import timedelta

    try:
        end = datetime.strptime(curr_date, "%Y-%m-%d")
    except ValueError:
        end = datetime.now()
    start = (end - timedelta(days=look_back_days + 10)).strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    # Fetch price data
    try:
        csv_data = get_stock_data(symbol, start, end_str)
    except NoMarketDataError:
        return f"# Insufficient data to compute {indicator} for {symbol}"

    # Parse CSV
    lines = [l for l in csv_data.split("\n") if l and not l.startswith("#")]
    if len(lines) < 3:
        return f"# Insufficient data to compute {indicator} for {symbol}"

    # Parse header and data
    import csv
    import io
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    rows = list(reader)
    if len(rows) < 2:
        return f"# Insufficient data to compute {indicator} for {symbol}"

    closes = [float(r["Close"]) for r in rows if r.get("Close")]
    ind = indicator.strip().lower()
    result_lines = [
        f"# {indicator.upper()} for {symbol.upper()}",
        f"# Source: akshare (computed)",
        f"# Period: {start} to {end_str}",
        "",
    ]

    if ind in ("close_50_sma", "50_sma", "sma_50"):
        if len(closes) >= 50:
            result_lines.append(f"close_50_sma: {sum(closes[-50:]) / 50:.2f}")
        else:
            result_lines.append(f"close_50_sma: insufficient data (need 50, have {len(closes)})")

    elif ind in ("close_200_sma", "200_sma", "sma_200"):
        if len(closes) >= 200:
            result_lines.append(f"close_200_sma: {sum(closes[-200:]) / 200:.2f}")
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
            result_lines.append("close_10_ema: insufficient data")

    elif ind in ("rsi", "rsi_14"):
        if len(closes) >= 15:
            gains, losses = [], []
            for i in range(1, 15):
                diff = closes[-i] - closes[-i - 1]
                gains.append(max(diff, 0))
                losses.append(max(-diff, 0))
            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14
            rsi = 100 if avg_loss == 0 else 100 - (100 / (1 + avg_gain / avg_loss))
            result_lines.append(f"rsi_14: {rsi:.2f}")
        else:
            result_lines.append("rsi_14: insufficient data")

    elif ind in ("macd",):
        if len(closes) >= 26:
            k12, k26 = 2 / (12 + 1), 2 / (26 + 1)
            ema12 = closes[-12]
            for p in closes[-11:]:
                ema12 = p * k12 + ema12 * (1 - k12)
            ema26 = closes[-26]
            for p in closes[-25:]:
                ema26 = p * k26 + ema26 * (1 - k26)
            macd_line = ema12 - ema26
            result_lines.append(f"macd: {macd_line:.2f}")
            result_lines.append(f"macds (signal): {macd_line:.2f}")
            result_lines.append(f"macdh (histogram): 0.00")
        else:
            result_lines.append("macd: insufficient data")

    elif ind in ("boll", "bollinger", "bollinger_bands"):
        if len(closes) >= 20:
            sma20 = sum(closes[-20:]) / 20
            variance = sum((c - sma20) ** 2 for c in closes[-20:]) / 20
            std = variance ** 0.5
            result_lines.extend([
                f"boll (middle): {sma20:.2f}",
                f"boll_ub (upper): {sma20 + 2 * std:.2f}",
                f"boll_lb (lower): {sma20 - 2 * std:.2f}",
            ])
        else:
            result_lines.append("bollinger_bands: insufficient data")

    elif ind in ("volume",):
        volumes = [float(r.get("Volume", 0)) for r in rows if r.get("Volume")]
        if volumes:
            result_lines.extend([
                f"average_volume: {sum(volumes) / len(volumes):.0f}",
                f"latest_volume: {volumes[-1]:.0f}",
            ])
        else:
            result_lines.append("volume: no data")

    else:
        result_lines.append(f"Indicator '{indicator}' not directly available from akshare.")
        result_lines.append("Available: rsi, macd, close_50_sma, close_200_sma, close_10_ema, boll, volume")

    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# Fundamentals, financial statements, news — not available via akshare
# These return informative messages; the routing layer falls back to yfinance.
# ---------------------------------------------------------------------------

def get_fundamentals(symbol: str) -> str:
    return "# Fundamentals not available from akshare. Use yfinance or FMP."


def get_balance_sheet(symbol: str, freq: str = "quarterly", curr_date: str = None) -> str:
    return "# Balance sheet not available from akshare. Use yfinance or FMP."


def get_cashflow(symbol: str, freq: str = "quarterly", curr_date: str = None) -> str:
    return "# Cash flow not available from akshare. Use yfinance or FMP."


def get_income_statement(symbol: str, freq: str = "quarterly", curr_date: str = None) -> str:
    return "# Income statement not available from akshare. Use yfinance or FMP."


def get_news(symbol: str) -> str:
    return "# News not available from akshare. Use yfinance or FMP."


def get_global_news() -> str:
    return "# Global news not available from akshare. Use yfinance or FMP."


def get_insider_transactions(symbol: str) -> str:
    return "# Insider transactions not available from akshare. Use yfinance or FMP."
