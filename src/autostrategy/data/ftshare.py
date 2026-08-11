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
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

DEFAULT_ENDPOINT = "https://market.ft.tech/gateway/mcp"
DEFAULT_MARKET_ENDPOINT = "https://market.ft.tech/app/api/v2"
_PROTOCOL_VERSION = "2025-03-26"
_TIMEOUT = 60
_SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


class FtshareError(RuntimeError):
    """Raised when the FTShare gateway returns an error or bad payload."""


class FtshareClient:
    """Minimal streamable-HTTP MCP client for the FTShare gateway."""

    def __init__(self, endpoint: str | None = None, timeout: int = _TIMEOUT) -> None:
        self.endpoint = endpoint or os.environ.get("AUTOSTRATEGY_FTSHARE_URL", DEFAULT_ENDPOINT)
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
            arguments["until_date"] = end_date or datetime.now(UTC).strftime("%Y-%m-%d")
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
            rows = [row for row in rows if _row_in_window(row, start_date, end_date)]
        return rows


class FtshareIntradayClient:
    """Small REST client for FTShare's current-day one-minute prices."""

    def __init__(self, endpoint: str | None = None, timeout: int = _TIMEOUT) -> None:
        self.endpoint = (
            endpoint
            or os.environ.get("AUTOSTRATEGY_FTSHARE_MARKET_URL")
            or DEFAULT_MARKET_ENDPOINT
        ).rstrip("/")
        self.timeout = timeout

    def prices(self, symbol: str, *, asset_type: str) -> list[dict[str, Any]]:
        collection = {
            "stock": "stocks",
            "etf": "etfs",
            "index": "indices",
        }.get(asset_type)
        if collection is None:
            raise FtshareError(f"Unsupported intraday asset type: {asset_type}")
        external_symbol = to_ftshare_price_symbol(symbol)
        query = urllib.parse.urlencode({"since": "TODAY"})
        request = urllib.request.Request(
            f"{self.endpoint}/{collection}/{external_symbol}/prices?{query}",
            headers={"Accept": "application/json", "X-Client-Name": "ft-web"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise FtshareError(f"FTShare intraday request failed: {exc}") from exc
        rows = payload.get("prices") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise FtshareError("FTShare intraday response does not contain a prices list")
        return [row for row in rows if isinstance(row, dict)]


def to_ftshare_price_symbol(symbol: str) -> str:
    """Convert ``600519.SH`` style symbols to FTShare REST symbols."""
    normalized = str(symbol).strip().upper()
    if "." not in normalized:
        raise FtshareError(f"Exchange-qualified symbol required: {symbol}")
    code, exchange = normalized.rsplit(".", 1)
    suffix = {"SH": "XSHG", "SZ": "XSHE"}.get(exchange)
    if len(code) != 6 or not code.isdigit() or suffix is None:
        raise FtshareError(f"Unsupported Shanghai/Shenzhen symbol: {symbol}")
    return f"{code}.{suffix}"


def fetch_intraday_prices(
    symbol: str,
    *,
    type_: str,
    client: FtshareIntradayClient | None = None,
) -> list[dict[str, Any]]:
    """Fetch current-day one-minute price points for a mainland symbol."""
    return (client or FtshareIntradayClient()).prices(symbol, asset_type=type_)


def aggregate_ten_minute_bars(
    rows: list[dict[str, Any]],
    *,
    symbol: str,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Aggregate FTShare minute price points into completed CN-session bars."""
    current = now or datetime.now(_SHANGHAI_TZ)
    if current.tzinfo is None:
        current = current.replace(tzinfo=_SHANGHAI_TZ)
    current = current.astimezone(_SHANGHAI_TZ)
    buckets: dict[datetime, list[tuple[datetime, float, float]]] = {}

    for row in rows:
        timestamp = _parse_intraday_timestamp(row.get("tm"))
        if timestamp is None:
            continue
        session_start = _session_start(timestamp)
        if session_start is None:
            continue
        elapsed_minutes = int((timestamp - session_start).total_seconds() // 60)
        bucket_start = session_start + timedelta(minutes=(elapsed_minutes // 10) * 10)
        bucket_end = bucket_start + timedelta(minutes=10)
        if bucket_end > current:
            continue
        price = row.get("p")
        if price is None:
            continue
        buckets.setdefault(bucket_end, []).append(
            (timestamp, float(price), float(row.get("v", 0) or 0))
        )

    bars: list[dict[str, Any]] = []
    for bucket_end, points in sorted(buckets.items()):
        points.sort(key=lambda item: item[0])
        prices = [point[1] for point in points]
        bars.append(
            {
                "at": bucket_end.isoformat(),
                "symbol": symbol,
                "open": prices[0],
                "high": max(prices),
                "low": min(prices),
                "close": prices[-1],
                "volume": sum(point[2] for point in points),
            }
        )
    return bars


def _parse_intraday_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) or str(value).isdigit():
        numeric = float(value)
        if numeric > 100_000_000_000:
            numeric /= 1000
        try:
            return datetime.fromtimestamp(numeric, tz=UTC).astimezone(_SHANGHAI_TZ)
        except (OverflowError, OSError, ValueError):
            return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_SHANGHAI_TZ)
    return parsed.astimezone(_SHANGHAI_TZ)


def _session_start(timestamp: datetime) -> datetime | None:
    local_time = timestamp.timetz().replace(tzinfo=None)
    day = timestamp.date()
    if time(9, 30) <= local_time < time(11, 30):
        return datetime.combine(day, time(9, 30), tzinfo=_SHANGHAI_TZ)
    if time(13, 0) <= local_time < time(15, 0):
        return datetime.combine(day, time(13, 0), tzinfo=_SHANGHAI_TZ)
    return None


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
        chunk = line[len("data:") :].strip()
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
    return datetime.fromtimestamp(int(ts) / 1000, tz=UTC).strftime("%Y-%m-%d")


def _row_in_window(row: dict[str, Any], start_date: str | None, end_date: str | None) -> bool:
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
            date = datetime.fromtimestamp(int(ts) / 1000, tz=UTC).date()
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
    if type_ in {"etf", "index"}:
        rows = _fetch_candlestick_rows(
            client,
            symbol=symbol,
            limit=limit,
            asset_type=type_,
            start_date=start_date,
            end_date=end_date,
        )
        return _rows_to_dataframe(rows)
    rows = client.daily_ohlc(
        symbol=symbol,
        limit=limit,
        type_=type_,
        start_date=start_date,
        end_date=end_date,
    )
    return _rows_to_dataframe(rows)


def _fetch_candlestick_rows(
    client: FtshareClient,
    *,
    symbol: str,
    limit: int,
    asset_type: str,
    start_date: str | None,
    end_date: str | None,
) -> list[dict[str, Any]]:
    """Fetch ETF/index bars in small reverse pages to avoid MCP response truncation."""
    tool_name = {
        "etf": "ft_etf_candlesticks",
        "index": "ft_index_candlesticks",
    }[asset_type]
    end_day = (
        datetime.fromisoformat(end_date).replace(tzinfo=UTC) if end_date else datetime.now(UTC)
    )
    cursor = int((end_day + timedelta(days=1) - timedelta(milliseconds=1)).timestamp() * 1000)
    since = (
        int(datetime.fromisoformat(start_date).replace(tzinfo=UTC).timestamp() * 1000)
        if start_date
        else None
    )
    remaining = max(int(limit), 0)
    rows: list[dict[str, Any]] = []

    while remaining > 0 and (since is None or cursor >= since):
        page_limit = min(150, remaining)
        arguments: dict[str, Any] = {
            "symbol": symbol,
            "interval_unit": "Day",
            "until_ts_millis": cursor,
            "limit": page_limit,
            "adjust_kind": "Forward" if asset_type == "etf" else "None",
        }
        if since is not None:
            arguments["since_ts_millis"] = since
        page = client.call_tool(tool_name, arguments) or []
        if isinstance(page, dict):
            page = [page]
        page_rows = [row for row in page if isinstance(row, dict)]
        if not page_rows:
            break
        rows.extend(page_rows)
        remaining -= len(page_rows)
        timestamps = [
            int(row.get("ts_millis") or row.get("ts_millis_open"))
            for row in page_rows
            if row.get("ts_millis") is not None or row.get("ts_millis_open") is not None
        ]
        if not timestamps:
            break
        next_cursor = min(timestamps) - 1
        if next_cursor >= cursor:
            break
        cursor = next_cursor
        if len(page_rows) < page_limit:
            break

    deduplicated: dict[int, dict[str, Any]] = {}
    for row in rows:
        timestamp = row.get("ts_millis") or row.get("ts_millis_open")
        if timestamp is not None:
            deduplicated[int(timestamp)] = row
    return list(deduplicated.values())
