"""FTShare MCP data source client.

Talks to the FTShare read-only financial data MCP gateway over
streamable HTTP (JSON-RPC + SSE). The default endpoint is
https://market.ft.tech/gateway/mcp and can be overridden with the
``AUTOSTRATEGY_FTSHARE_URL`` environment variable.

The primary entry point is :func:`fetch_daily_ohlc`, which returns a
``pandas.DataFrame`` indexed by ``date`` with OHLCV columns, ready for
Backtrader's ``PandasData`` feed.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any

DEFAULT_ENDPOINT = "https://market.ft.tech/gateway/mcp"
_PROTOCOL_VERSION = "2025-03-26"
_TIMEOUT = 60


class FtshareError(RuntimeError):
    """Raised when the FTShare gateway returns an error or bad payload."""


class FtshareClient:
    """Minimal streamable-HTTP MCP client for the FTShare gateway."""

    def __init__(self, endpoint: str | None = None, timeout: int = _TIMEOUT) -> None:
        self.endpoint = endpoint or os.environ.get(
            "AUTOSTRATEGY_FTSHARE_URL", DEFAULT_ENDPOINT
        )
        self.timeout = timeout
        self._session_id: str | None = None
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        request = urllib.request.Request(self.endpoint, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                session = response.headers.get("Mcp-Session-Id")
                if session:
                    self._session_id = session
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise FtshareError(f"FTShare gateway request failed: {exc}") from exc
        if not raw.strip():
            return {}
        return _parse_sse_json(raw)

    def initialize(self) -> None:
        """Perform the MCP handshake and open a session."""
        result = self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "autostrategy", "version": "0.1.0"},
                },
            }
        )
        if "error" in result:
            raise FtshareError(f"MCP initialize failed: {result['error']}")
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call an FTShare tool and return ``structuredContent.data``."""
        if self._session_id is None:
            self.initialize()
        result = self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        if "error" in result:
            raise FtshareError(f"FTShare tool {name} failed: {result['error']}")
        payload = result.get("result", {})
        if payload.get("isError"):
            raise FtshareError(f"FTShare tool {name} returned error: {payload}")
        structured = payload.get("structuredContent") or {}
        return structured.get("data")

    def daily_ohlc(
        self,
        symbol: str,
        limit: int = 500,
        type_: str = "stock",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch daily OHLC rows for a symbol.

        ``symbol`` uses the FTShare external code format, e.g.
        ``600519.SH`` (A股), ``00700.HK`` (港股). ``type_`` selects the
        FTShare ``daily_ohlc``口径: ``stock``, ``hk_stock``, ``us_stock``,
        ``global_index``, etc. ``start_date``/``end_date`` use
        ``YYYY-MM-DD`` and apply where the口径 supports them.
        """
        arguments: dict[str, Any] = {"type": type_, "limit": limit}
        if type_ == "us_stock":
            arguments["stock_code"] = symbol
        elif type_ == "global_index":
            arguments["secid"] = symbol
        elif type_ == "hk_index":
            arguments["index_code"] = symbol
        else:
            arguments["symbol"] = symbol
        if type_ == "hk_stock":
            # hk_stock requires until_date and does not accept start_date;
            # the date window is applied client-side below.
            arguments["until_date"] = end_date or datetime.now(
                timezone.utc
            ).strftime("%Y-%m-%d")
        elif type_ in ("us_stock", "hk_index", "global_index", "eastmoney_board"):
            if start_date:
                arguments["start_date"] = start_date
            if end_date:
                arguments["end_date"] = end_date
        data = self.call_tool("daily_ohlc", arguments)
        if data is None:
            return []
        if isinstance(data, dict):
            data = [data]
        rows = list(data)
        # stock/hk_stock return the earliest `limit` rows; apply the
        # requested date window client-side so callers always get rows
        # inside [start_date, end_date].
        if start_date or end_date:
            rows = [
                row
                for row in rows
                if _row_in_window(row, start_date, end_date)
            ]
        return rows


def _parse_sse_json(raw: str) -> dict[str, Any]:
    """Parse a streamable-HTTP MCP response body.

    The gateway responds either with plain JSON or with an SSE stream
    whose ``data:`` lines carry JSON-RPC messages. Return the first
    JSON-RPC message that has a ``result`` or ``error`` key.
    """
    raw = raw.strip()
    if not raw:
        raise FtshareError("Empty response from FTShare gateway")
    if raw.startswith("{"):
        return json.loads(raw)
    fallback: dict[str, Any] | None = None
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        chunk = line[len("data:"):].strip()
        if not chunk:
            continue
        try:
            message = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict):
            continue
        if "result" in message or "error" in message:
            return message
        fallback = fallback or message
    if fallback is not None:
        return fallback
    raise FtshareError("No JSON-RPC message found in FTShare SSE response")


def _row_date_str(row: dict[str, Any]) -> str | None:
    date = row.get("date")
    if date:
        return str(date)[:10]
    ts = row.get("ts_millis") or row.get("ts_millis_open")
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _row_in_window(
    row: dict[str, Any], start_date: str | None, end_date: str | None
) -> bool:
    date = _row_date_str(row)
    if date is None:
        return False
    if start_date and date < start_date:
        return False
    if end_date and date > end_date:
        return False
    return True


def _rows_to_dataframe(rows: list[dict[str, Any]]):
    """Convert FTShare daily OHLC rows to a Backtrader-ready DataFrame."""
    import pandas as pd

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    records = []
    for row in rows:
        date = row.get("date")
        if date is None:
            ts = row.get("ts_millis") or row.get("ts_millis_open")
            if ts is None:
                continue
            date = datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc).date()
        records.append(
            {
                "date": date,
                "open": float(row.get("open", 0) or 0),
                "high": float(row.get("high", 0) or 0),
                "low": float(row.get("low", 0) or 0),
                "close": float(row.get("close", 0) or 0),
                "volume": float(row.get("volume", 0) or 0),
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return df


def fetch_daily_ohlc(
    symbol: str,
    limit: int = 500,
    type_: str = "stock",
    start_date: str | None = None,
    end_date: str | None = None,
    client: FtshareClient | None = None,
):
    """Fetch daily OHLCV data from FTShare and return a DataFrame.

    The DataFrame is indexed by ``date`` (ascending) with columns
    ``open, high, low, close, volume`` — compatible with
    ``backtrader.feeds.PandasData``.
    """
    client = client or FtshareClient()
    rows = client.daily_ohlc(
        symbol=symbol,
        limit=limit,
        type_=type_,
        start_date=start_date,
        end_date=end_date,
    )
    return _rows_to_dataframe(rows)
