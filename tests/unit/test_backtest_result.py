"""Tests for the strict backtest result contract."""

import math

import pytest
from pydantic import ValidationError

from autostrategy.core.backtest_result import (
    BacktestMetrics,
    assess_research_quality,
)


def _valid_metrics(**overrides):
    metrics = {
        "annual_return": 12.0,
        "max_drawdown": 8.0,
        "sharpe": 1.5,
        "win_rate": 55.0,
        "profit_loss_ratio": 1.8,
        "total_trades": 10,
    }
    metrics.update(overrides)
    return metrics


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_backtest_metrics_reject_non_finite_numbers(value):
    with pytest.raises(ValidationError):
        BacktestMetrics.model_validate(_valid_metrics(sharpe=value))


@pytest.mark.parametrize(
    ("field", "value"),
    [("win_rate", -0.1), ("win_rate", 100.1), ("max_drawdown", -1.0)],
)
def test_backtest_metrics_reject_invalid_ranges(field, value):
    with pytest.raises(ValidationError):
        BacktestMetrics.model_validate(_valid_metrics(**{field: value}))


def test_backtest_metrics_require_at_least_one_trade():
    with pytest.raises(ValidationError):
        BacktestMetrics.model_validate(_valid_metrics(total_trades=0))


def test_backtest_metrics_preserve_valid_legacy_extras():
    metrics = BacktestMetrics.model_validate(
        _valid_metrics(initial_cash=1_000_000, first_half_return=4.2)
    )

    assert metrics.initial_cash == 1_000_000
    assert metrics.model_dump()["first_half_return"] == 4.2


def test_research_quality_reports_limited_sample_and_missing_evidence():
    quality = assess_research_quality(_valid_metrics(total_trades=10))

    assert quality.trade_sample == "limited"
    assert quality.has_equity_curve is False
    assert quality.has_trade_records is False
    assert quality.has_benchmark is False
    assert quality.has_out_of_sample is False
    assert len(quality.warnings) >= 5


def test_research_quality_recognizes_complete_evidence():
    quality = assess_research_quality(
        _valid_metrics(
            total_trades=30,
            equity_curve=[{"date": "2026-01-01", "equity": 1_000_000}],
            trades=[
                {
                    "date": "2026-01-02",
                    "symbol": "510300.SH",
                    "action": "buy",
                    "price": 4.0,
                    "quantity": 100,
                    "cost": 400,
                }
            ],
            benchmark={"symbol": "000300.SH", "annual_return": 8.0},
            out_of_sample={"annual_return": 6.0, "period": "2025"},
        )
    )

    assert quality.trade_sample == "adequate"
    assert quality.has_equity_curve is True
    assert quality.has_trade_records is True
    assert quality.has_benchmark is True
    assert quality.has_out_of_sample is True
    assert quality.warnings == []
