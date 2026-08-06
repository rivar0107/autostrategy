"""Tests for strategy service."""

from autostrategy.services.exceptions import StrategyNotFoundError
from autostrategy.services.strategy_service import StrategyService


def test_strategy_service_create_list_detail_delete(tmp_path):
    service = StrategyService(workspace_root=tmp_path)

    created = service.create_strategy("demo", market="A股", template="dual-ma")
    strategies = service.list_strategies()
    detail = service.get_strategy_detail("demo")

    assert created.slug == "demo"
    assert len(strategies) == 1
    assert detail.strategy.slug == "demo"
    assert detail.paths.design == tmp_path / "demo" / "STRATEGY_DESIGN.md"

    service.delete_strategy("demo")

    try:
        service.get_strategy("demo")
    except StrategyNotFoundError:
        pass
    else:
        raise AssertionError("Expected StrategyNotFoundError")


def test_strategy_service_lists_templates(tmp_path):
    service = StrategyService(workspace_root=tmp_path)

    templates = service.list_templates()

    assert "dual-ma" in templates
