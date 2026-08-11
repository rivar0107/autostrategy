"""Phase 5D local feed tests: bar schema + fixture loading."""

from __future__ import annotations

import json

import pytest

from autostrategy.data.feed import LocalFeed, load_bars, normalize_bar


def test_normalize_bar_schema():
    bar = normalize_bar(
        {
            "date": "2024-01-02",
            "symbol": "000300.SH",
            "open": "10.0",
            "high": 10.5,
            "low": 9.8,
            "close": 10.2,
            "volume": 12345,
        }
    )
    assert bar == {
        "at": "2024-01-02T00:00:00",
        "symbol": "000300.SH",
        "open": 10.0,
        "high": 10.5,
        "low": 9.8,
        "close": 10.2,
        "volume": 12345.0,
    }


def test_normalize_bar_requires_time_and_symbol():
    with pytest.raises(ValueError, match="时间"):
        normalize_bar({"symbol": "X", "open": 1, "high": 1, "low": 1, "close": 1})
    with pytest.raises(ValueError, match="symbol"):
        normalize_bar({"at": "2024-01-02", "open": 1, "high": 1, "low": 1, "close": 1})
    with pytest.raises(ValueError, match="无法解析"):
        normalize_bar(
            {"at": "not-a-date", "symbol": "X", "open": 1, "high": 1, "low": 1, "close": 1}
        )


@pytest.fixture()
def csv_feed(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text(
        "date,symbol,open,high,low,close,volume\n"
        "2024-01-03,000300.SH,11,11.5,10.8,11.2,2000\n"
        "2024-01-02,000300.SH,10,10.5,9.8,10.2,1000\n"
        "2024-01-02,600519.SH,100,101,99,100.5,500\n"
        "2024-01-04,000300.SH,12,12.5,11.8,12.2,3000\n",
        encoding="utf-8",
    )
    return path


def test_load_bars_sorts_by_time(csv_feed):
    bars = load_bars(csv_feed)
    assert [bar["at"] for bar in bars] == [
        "2024-01-02T00:00:00",
        "2024-01-02T00:00:00",
        "2024-01-03T00:00:00",
        "2024-01-04T00:00:00",
    ]


def test_load_bars_filters_symbols(csv_feed):
    bars = load_bars(csv_feed, symbols=["600519.SH"])
    assert len(bars) == 1
    assert bars[0]["symbol"] == "600519.SH"


def test_load_bars_time_window(csv_feed):
    bars = load_bars(csv_feed, start="2024-01-03", end="2024-01-03")
    assert len(bars) == 1
    assert bars[0]["close"] == 11.2


def test_load_bars_jsonl(tmp_path):
    path = tmp_path / "bars.jsonl"
    records = [
        {
            "at": "2024-02-01T09:30:00",
            "symbol": "A",
            "open": 1,
            "high": 2,
            "low": 0.5,
            "close": 1.5,
            "volume": 10,
        },
        {
            "at": "2024-02-02T09:30:00",
            "symbol": "A",
            "open": 2,
            "high": 3,
            "low": 1.5,
            "close": 2.5,
            "volume": 20,
        },
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    bars = load_bars(path)
    assert len(bars) == 2
    assert bars[0]["at"] == "2024-02-01T09:30:00"


def test_load_bars_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("date,symbol,open,high,low,close,volume\n", encoding="utf-8")
    assert load_bars(path) == []


def test_load_bars_missing_file_and_bad_suffix(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bars(tmp_path / "missing.csv")
    bad = tmp_path / "bars.txt"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="不支持"):
        load_bars(bad)


def test_local_feed_metadata(csv_feed):
    feed = LocalFeed(csv_feed, symbols=["000300.SH"], start="2024-01-02", end="2024-01-04")
    assert len(feed) == 3
    meta = feed.metadata()
    assert meta["bar_count"] == 3
    assert meta["symbol_count"] == 1
    assert meta["symbols"] == ["000300.SH"]
    assert meta["start"] == "2024-01-02T00:00:00"
    assert meta["end"] == "2024-01-04T00:00:00"
    assert meta["source"].endswith("bars.csv")
