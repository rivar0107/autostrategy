"""Fetch OHLCV data for the strategy from FTShare MCP gateway.

Exposes ``fetch(config)`` which returns a pandas DataFrame indexed by
``date`` with columns ``open, high, low, close, volume``.

Data is ALWAYS fetched from the FTShare MCP gateway
(https://market.ft.tech/gateway/mcp). No local CSV fallback.
"""

from __future__ import annotations

import pandas as pd


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
    """Return OHLCV data for the configured symbol from FTShare MCP."""
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
