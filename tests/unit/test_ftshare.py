"""Unit tests for the FTShare MCP data source."""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from autostrategy.data.ftshare import (
    FtshareClient,
    FtshareError,
    FtshareIntradayClient,
    _parse_sse_json,
    _rows_to_dataframe,
    aggregate_ten_minute_bars,
    fetch_daily_ohlc,
    to_ftshare_price_symbol,
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


def test_ftshare_price_symbol_uses_exchange_qualified_rest_format():
    assert to_ftshare_price_symbol("510500.SH") == "510500.XSHG"
    assert to_ftshare_price_symbol("159915.SZ") == "159915.XSHE"
    assert to_ftshare_price_symbol("000905.SH") == "000905.XSHG"


def test_intraday_client_uses_ftshare_prices_rest_contract(monkeypatch):
    captured = {}

    class FakeResponse:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"prices": [{"tm": "2026-08-11T09:30:00", "p": 1}]}).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["client_name"] = request.get_header("X-client-name")
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    rows = FtshareIntradayClient(endpoint="https://market.example/api", timeout=7).prices(
        "510500.SH", asset_type="etf"
    )

    assert captured == {
        "url": "https://market.example/api/etfs/510500.XSHG/prices?since=TODAY",
        "client_name": "ft-web",
        "timeout": 7,
    }
    assert rows[0]["p"] == 1


def test_ten_minute_bars_only_publish_completed_bucket():
    rows = [
        {
            "tm": f"2026-08-11T09:{minute:02d}:00",
            "p": 10 + (minute - 30) * 0.1,
            "v": 100 + minute,
        }
        for minute in range(30, 42)
    ]

    bars = aggregate_ten_minute_bars(
        rows,
        symbol="510500.SH",
        now=datetime(2026, 8, 11, 9, 45, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert len(bars) == 1
    assert bars[0] == {
        "at": "2026-08-11T09:40:00+08:00",
        "symbol": "510500.SH",
        "open": 10.0,
        "high": 10.9,
        "low": 10.0,
        "close": 10.9,
        "volume": sum(100 + minute for minute in range(30, 40)),
    }


def test_ten_minute_bars_accept_ftshare_millisecond_timestamps():
    zone = ZoneInfo("Asia/Shanghai")
    rows = [
        {
            "tm": int(datetime(2026, 8, 11, 9, minute, tzinfo=zone).timestamp() * 1000),
            "p": 10 + (minute - 30) * 0.1,
            "v": 100,
        }
        for minute in range(30, 40)
    ]

    bars = aggregate_ten_minute_bars(
        rows,
        symbol="510500.SH",
        now=datetime(2026, 8, 11, 9, 40, tzinfo=zone),
    )

    assert len(bars) == 1
    assert bars[0]["at"] == "2026-08-11T09:40:00+08:00"
    assert bars[0]["close"] == 10.9


def test_ten_minute_bars_do_not_cross_lunch_break():
    rows = [
        {"tm": "2026-08-11T11:29:00", "p": 10.0, "v": 100},
        {"tm": "2026-08-11T13:00:00", "p": 11.0, "v": 200},
        {"tm": "2026-08-11T13:09:00", "p": 12.0, "v": 300},
    ]

    bars = aggregate_ten_minute_bars(
        rows,
        symbol="510500.SH",
        now=datetime(2026, 8, 11, 13, 10, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert [bar["at"] for bar in bars] == [
        "2026-08-11T11:30:00+08:00",
        "2026-08-11T13:10:00+08:00",
    ]
    assert bars[0]["close"] == 10.0
    assert bars[1]["open"] == 11.0


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
        {
            "ts_millis": 1785826800000,
            "open": "1",
            "high": "2",
            "low": "0.5",
            "close": "1.5",
            "volume": 100,
        },
        {
            "ts_millis": 1785913200000,
            "open": "2",
            "high": "3",
            "low": "1.5",
            "close": "2.5",
            "volume": 200,
        },
    ]
    df = _rows_to_dataframe(rows)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df.index.is_monotonic_increasing


def test_rows_to_dataframe_date_field():
    rows = [
        {
            "date": "2026-08-05",
            "open": "1",
            "high": "2",
            "low": "0.5",
            "close": "1.5",
            "volume": 100,
        },
        {
            "date": "2026-08-04",
            "open": "2",
            "high": "3",
            "low": "1.5",
            "close": "2.5",
            "volume": 200,
        },
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
        {
            "ts_millis": 1785826800000,
            "open": "1",
            "high": "2",
            "low": "0.5",
            "close": "1.5",
            "volume": 100,
        },
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


def test_fetch_daily_ohlc_pages_candlesticks_without_gateway_truncation():
    client = FtshareClient()
    calls = []

    def fake_call_tool(name, arguments):
        calls.append((name, arguments))
        until = arguments["until_ts_millis"] // 86_400_000 * 86_400_000
        return [
            {
                "ts_millis": until - (arguments["limit"] - index - 1) * 86_400_000,
                "open": "1",
                "high": "2",
                "low": "0.5",
                "close": "1.5",
                "volume": 100,
            }
            for index in range(arguments["limit"])
        ]

    client.call_tool = fake_call_tool  # type: ignore[assignment]
    df = fetch_daily_ohlc(
        "510500.SH",
        limit=300,
        type_="etf",
        start_date="2020-01-01",
        end_date="2024-12-31",
        client=client,
    )

    assert len(calls) == 2
    assert all(name == "ft_etf_candlesticks" for name, _ in calls)
    assert all(call[1]["limit"] == 150 for call in calls)
    assert len(df) == 300
    assert df.index.is_monotonic_increasing


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
