"""Fetch OHLCV data for the strategy.

Reads ``data/data.csv`` first; if missing, pulls daily bars from the
FTShare MCP gateway (https://market.ft.tech/gateway/mcp).

Exposes ``fetch(config)`` which returns a pandas DataFrame indexed by
``date`` with columns ``open, high, low, close, volume``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _csv_path() -> Path:
    return Path(__file__).resolve().parent / "data.csv"


def _symbol_and_type(config: dict) -> tuple[str, str]:
    symbols = config.get("symbols") or ["000300.SH"]
    symbol = symbols[0] if isinstance(symbols, list) else symbols
    market = str(config.get("market", "A股"))
    if "港" in market or str(symbol).endswith(".HK"):
        return symbol, "hk_stock"
    if "美" in market or (isinstance(symbol, str) and symbol.isalpha()):
        return symbol, "us_stock"
    return symbol, "stock"


def fetch(config: dict) -> pd.DataFrame:
    """Return OHLCV data for the configured symbol."""
    csv_file = _csv_path()
    if csv_file.exists():
        df = pd.read_csv(csv_file, parse_dates=["date"])
        return df.set_index("date").sort_index()

    from autostrategy.data.ftshare import fetch_daily_ohlc

    symbol, type_ = _symbol_and_type(config)
    period = config.get("period", {})
    return fetch_daily_ohlc(
        symbol,
        limit=int(config.get("data_limit", 500)),
        type_=type_,
        start_date=period.get("start"),
        end_date=period.get("end"),
    )
