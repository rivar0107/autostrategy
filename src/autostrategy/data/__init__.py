"""Data source integrations for autostrategy strategies."""

from autostrategy.data.ftshare import (
    FtshareClient,
    FtshareIntradayClient,
    aggregate_ten_minute_bars,
    fetch_daily_ohlc,
    fetch_intraday_prices,
)

__all__ = [
    "FtshareClient",
    "FtshareIntradayClient",
    "aggregate_ten_minute_bars",
    "fetch_daily_ohlc",
    "fetch_intraday_prices",
]
