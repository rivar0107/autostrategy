"""Backtest service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autostrategy.core.backtest_engine import run_backtest_workflow
from autostrategy.core.strategy import StrategyStatus
from autostrategy.persistence.run_store import RunStore
from autostrategy.services.exceptions import BacktestServiceError, StrategyNotFoundError
from autostrategy.services.models import BacktestResult, BacktestRunRecord, BacktestRunSummary
from autostrategy.services.strategy_service import StrategyService
from autostrategy.services.version_service import VersionService


class BacktestService:
    """Application service for running and reading strategy backtests."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.strategy_service = StrategyService(workspace_root=workspace_root)
        self.run_store = RunStore(self.strategy_service.workspace_root)
        self.version_service = VersionService(workspace_root=workspace_root)

    def run_backtest(self, slug: str) -> BacktestResult:
        """Run a backtest for a strategy and persist its result JSON."""
        try:
            strategy_dir = self.strategy_service.workspace.get_strategy_dir(slug)
        except FileNotFoundError as exc:
            raise StrategyNotFoundError(str(exc)) from exc

        version = self.version_service.ensure_live_version(slug)
        result = run_backtest_workflow(strategy_dir)
        if "error" in result:
            raise BacktestServiceError(
                str(result["error"]), details={"score": result.get("score", 0)}
            )

        self.strategy_service.workspace.update_strategy_status(slug, StrategyStatus.BACKTESTED)
        self.run_store.record_backtest(
            strategy_slug=slug,
            strategy_version=version.version,
            strategy_digest=version.content_digest,
            score=float(result.get("score", 0)),
            result=result,
            version_id=version.version_id,
            phase="full",
        )
        return self._build_backtest_result(slug, result)

    def list_backtest_runs(self, slug: str) -> list[BacktestRunSummary]:
        """List immutable backtest history for a strategy."""
        self.strategy_service.get_strategy(slug)
        return self.run_store.list_backtest_runs(slug)

    def get_backtest_run(self, slug: str, run_id: str) -> BacktestRunRecord:
        """Read one immutable backtest snapshot."""
        self.strategy_service.get_strategy(slug)
        record = self.run_store.get_backtest_run(slug, run_id)
        if record is None:
            raise BacktestServiceError(f"Backtest run '{run_id}' not found for strategy '{slug}'.")
        return record

    def get_backtest_result(self, slug: str) -> BacktestResult:
        """Read the latest persisted backtest result for a strategy."""
        paths = self.strategy_service.get_strategy_paths(slug)
        if not paths.backtest_result.exists():
            raise BacktestServiceError(f"Backtest result for strategy '{slug}' not found.")
        with open(paths.backtest_result, encoding="utf-8") as file:
            result: dict[str, Any] = json.load(file)
        return self._build_backtest_result(slug, result)

    def _build_backtest_result(self, slug: str, result: dict[str, Any]) -> BacktestResult:
        paths = self.strategy_service.get_strategy_paths(slug)
        strategy = self.strategy_service.get_strategy(slug)
        return BacktestResult(
            strategy=strategy,
            result_path=paths.backtest_result,
            score=float(result.get("score", 0)),
            result=result,
        )
