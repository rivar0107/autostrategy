"""Contract tests for the generated FTShare data adapter template."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

import autostrategy.data.ftshare as ftshare


def test_fetch_data_template_uses_public_ftshare_signature(monkeypatch):
    template = (
        Path(__file__).parents[2]
        / "src"
        / "autostrategy"
        / "templates"
        / "_shared"
        / "fetch_data.py"
    )
    spec = importlib.util.spec_from_file_location("generated_fetch_data_template", template)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = {}

    def fake_fetch_daily_ohlc(symbol, limit=500, type_="stock", start_date=None, end_date=None):
        captured.update(
            symbol=symbol,
            limit=limit,
            type_=type_,
            start_date=start_date,
            end_date=end_date,
        )
        return pd.DataFrame({"close": [1.0]})

    monkeypatch.setattr(ftshare, "fetch_daily_ohlc", fake_fetch_daily_ohlc)

    result = module.fetch(
        {
            "symbols": ["000905.SH"],
            "data_limit": 600,
            "period": {"start": "2024-01-01", "end": "2024-12-31"},
        }
    )

    assert len(result) == 1
    assert captured == {
        "symbol": "000905.SH",
        "limit": 600,
        "type_": "stock",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
    }
