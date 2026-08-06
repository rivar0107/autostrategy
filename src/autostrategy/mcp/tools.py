"""MCP tool adapters for autostrategy.

The functions in this module are intentionally plain Python wrappers around
services so they can be tested without starting an MCP transport.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autostrategy.services.backtest_service import BacktestService
from autostrategy.services.exceptions import AutostrategyServiceError
from autostrategy.services.strategy_service import StrategyService


def _workspace_root(path: str | None) -> Path | None:
    return Path(path) if path else None


def _ok(data: Any) -> dict[str, Any]:
    return {"ok": True, "data": data}


def _error(exc: AutostrategyServiceError) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
        },
    }


def list_strategies(workspace_root: str | None = None) -> dict[str, Any]:
    """List strategies in the workspace."""
    service = StrategyService(workspace_root=_workspace_root(workspace_root))
    return _ok([strategy.model_dump(mode="json") for strategy in service.list_strategies()])


def get_strategy(slug: str, workspace_root: str | None = None) -> dict[str, Any]:
    """Get a strategy detail by slug."""
    service = StrategyService(workspace_root=_workspace_root(workspace_root))
    try:
        detail = service.get_strategy_detail(slug)
    except AutostrategyServiceError as exc:
        return _error(exc)
    return _ok(detail.model_dump(mode="json"))


def get_strategy_paths(slug: str, workspace_root: str | None = None) -> dict[str, Any]:
    """Get important local file paths for a strategy."""
    service = StrategyService(workspace_root=_workspace_root(workspace_root))
    try:
        paths = service.get_strategy_paths(slug)
    except AutostrategyServiceError as exc:
        return _error(exc)
    return _ok(paths.model_dump(mode="json"))


def list_templates(workspace_root: str | None = None) -> dict[str, Any]:
    """List built-in templates."""
    service = StrategyService(workspace_root=_workspace_root(workspace_root))
    return _ok(service.list_templates())


def get_backtest_result(slug: str, workspace_root: str | None = None) -> dict[str, Any]:
    """Read the latest backtest result for a strategy."""
    service = BacktestService(workspace_root=_workspace_root(workspace_root))
    try:
        result = service.get_backtest_result(slug)
    except AutostrategyServiceError as exc:
        return _error(exc)
    return _ok(result.model_dump(mode="json"))


def create_strategy(
    name: str,
    market: str = "A股",
    template: str | None = None,
    workspace_root: str | None = None,
) -> dict[str, Any]:
    """Create a strategy workspace."""
    service = StrategyService(workspace_root=_workspace_root(workspace_root))
    try:
        strategy = service.create_strategy(name=name, market=market, template=template)
    except AutostrategyServiceError as exc:
        return _error(exc)
    return _ok(strategy.model_dump(mode="json"))


def run_backtest(slug: str, workspace_root: str | None = None) -> dict[str, Any]:
    """Run a local backtest for a strategy."""
    service = BacktestService(workspace_root=_workspace_root(workspace_root))
    try:
        result = service.run_backtest(slug)
    except AutostrategyServiceError as exc:
        return _error(exc)
    return _ok(result.model_dump(mode="json"))
