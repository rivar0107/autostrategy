"""Data source integrations for autostrategy strategies."""

from autostrategy.data.ftshare import FtshareClient, fetch_daily_ohlc

__all__ = ["FtshareClient", "fetch_daily_ohlc"]
