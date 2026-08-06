"""Unit tests for the FTShare MCP data source."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from autostrategy.data.ftshare import (
    FtshareClient,
    FtshareError,
    _parse_sse_json,
    _rows_to_dataframe,
    fetch_daily_ohlc,
)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n"


def _tool_response(rows):
    return {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "structuredContent": {"data": rows},
            "isError": False,
        },
    }


def test_parse_sse_json_plain():
    assert _parse_sse_json('{"jsonrpc":"2.0","id":1,"result":{}}')["id"] == 1


def test_parse_sse_json_sse():
    body = 'data: \n\ndata: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n'
    assert _parse_sse_json(body)["result"]["ok"] is True


def test_parse_sse_json_empty_raises():
    with pytest.raises(FtshareError):
        _parse_sse_json("   ")


def test_rows_to_dataframe_ts_millis():
    rows = [
        {"ts_millis": 1785826800000, "open": "1", "high": "2", "low": "0.5", "close": "1.5", "volume": 100},
        {"ts_millis": 1785913200000, "open": "2", "high": "3", "low": "1.5", "close": "2.5", "volume": 200},
    ]
    df = _rows_to_dataframe(rows)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df.index.is_monotonic_increasing


def test_rows_to_dataframe_date_field():
    rows = [
        {"date": "2026-08-05", "open": "1", "high": "2", "low": "0.5", "close": "1.5", "volume": 100},
        {"date": "2026-08-04", "open": "2", "high": "3", "low": "1.5", "close": "2.5", "volume": 200},
    ]
    df = _rows_to_dataframe(rows)
    assert len(df) == 2
    assert df.index[0].strftime("%Y-%m-%d") == "2026-08-04"


def test_rows_to_dataframe_empty():
    df = _rows_to_dataframe([])
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.empty


def test_client_stock_arguments():
    client = FtshareClient()
    captured = {}

    def fake_post(payload):
        captured.update(payload)
        if payload.get("method") == "tools/call":
            return _tool_response([])
        return {"jsonrpc": "2.0", "id": payload.get("id"), "result": {}}

    client._post = fake_post  # type: ignore[assignment]
    client._session_id = "s"
    client.daily_ohlc("600519.SH", limit=5)
    args = captured["params"]["arguments"]
    assert args["type"] == "stock"
    assert args["symbol"] == "600519.SH"
    assert args["limit"] == 5


def test_client_hk_stock_requires_until_date():
    client = FtshareClient()
    captured = {}

    def fake_post(payload):
        captured.update(payload)
        if payload.get("method") == "tools/call":
            return _tool_response([])
        return {"jsonrpc": "2.0", "id": payload.get("id"), "result": {}}

    client._post = fake_post  # type: ignore[assignment]
    client._session_id = "s"
    client.daily_ohlc("00700.HK", type_="hk_stock")
    args = captured["params"]["arguments"]
    assert args["type"] == "hk_stock"
    assert "until_date" in args
    assert "start_date" not in args


def test_client_us_stock_arguments():
    client = FtshareClient()
    captured = {}

    def fake_post(payload):
        captured.update(payload)
        if payload.get("method") == "tools/call":
            return _tool_response([])
        return {"jsonrpc": "2.0", "id": payload.get("id"), "result": {}}

    client._post = fake_post  # type: ignore[assignment]
    client._session_id = "s"
    client.daily_ohlc("AAPL", type_="us_stock", start_date="2026-07-01", end_date="2026-07-03")
    args = captured["params"]["arguments"]
    assert args["type"] == "us_stock"
    assert args["stock_code"] == "AAPL"
    assert args["start_date"] == "2026-07-01"
    assert args["end_date"] == "2026-07-03"


def test_client_filters_date_window_client_side():
    rows = [
        {"date": "2026-07-01", "open": "1", "high": "1", "low": "1", "close": "1", "volume": 1},
        {"date": "2026-08-01", "open": "2", "high": "2", "low": "2", "close": "2", "volume": 2},
    ]
    client = FtshareClient()

    def fake_post(payload):
        if payload.get("method") == "tools/call":
            return _tool_response(rows)
        return {"jsonrpc": "2.0", "id": payload.get("id"), "result": {}}

    client._post = fake_post  # type: ignore[assignment]
    client._session_id = "s"
    result = client.daily_ohlc("600519.SH", start_date="2026-07-15")
    assert len(result) == 1
    assert result[0]["date"] == "2026-08-01"


def test_fetch_daily_ohlc_returns_dataframe():
    rows = [
        {"ts_millis": 1785826800000, "open": "1", "high": "2", "low": "0.5", "close": "1.5", "volume": 100},
    ]
    client = FtshareClient()

    def fake_post(payload):
        if payload.get("method") == "tools/call":
            return _tool_response(rows)
        return {"jsonrpc": "2.0", "id": payload.get("id"), "result": {}}

    client._post = fake_post  # type: ignore[assignment]
    client._session_id = "s"
    df = fetch_daily_ohlc("600519.SH", client=client)
    assert len(df) == 1
    assert df.iloc[0]["close"] == 1.5


def test_tool_error_raises():
    client = FtshareClient()

    def fake_post(payload):
        if payload.get("method") == "tools/call":
            return {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"isError": True, "content": [{"type": "text", "text": "bad"}]},
            }
        return {"jsonrpc": "2.0", "id": payload.get("id"), "result": {}}

    client._post = fake_post  # type: ignore[assignment]
    client._session_id = "s"
    with pytest.raises(FtshareError):
        client.daily_ohlc("600519.SH")
