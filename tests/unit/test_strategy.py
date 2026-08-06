"""Tests for strategy model."""

from datetime import datetime

from autostrategy.core.strategy import Strategy, StrategyStatus


def test_strategy_defaults():
    strategy = Strategy(name="dual-ma", market="A股")
    assert strategy.name == "dual-ma"
    assert strategy.market == "A股"
    assert strategy.status == StrategyStatus.DRAFT
    assert isinstance(strategy.created_at, datetime)


def test_strategy_slug():
    strategy = Strategy(name="Dual MA Strategy", market="A股")
    assert strategy.slug == "dual-ma-strategy"
