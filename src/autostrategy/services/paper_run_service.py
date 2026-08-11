"""Paper run service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autostrategy.core.backtest_engine import run_paper_replay_workflow
from autostrategy.core.strategy import StrategyStatus
from autostrategy.services.exceptions import PaperRunServiceError, StrategyNotFoundError
from autostrategy.services.models import PaperRunResult
from autostrategy.services.strategy_service import StrategyService


class PaperRunService:
    """Application service for running and reading paper replay results."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.strategy_service = StrategyService(workspace_root=workspace_root)

    def run_paper(self, slug: str, stop_requested=None) -> PaperRunResult:
        """Run a replay-first paper run for a strategy."""
        try:
            strategy_dir = self.strategy_service.workspace.get_strategy_dir(slug)
        except FileNotFoundError as exc:
            raise StrategyNotFoundError(str(exc)) from exc

        self.strategy_service.workspace.update_strategy_status(slug, StrategyStatus.PAPER_RUNNING)
        result = run_paper_replay_workflow(strategy_dir, stop_requested=stop_requested)
        if result.get("run_status") == "failed":
            raise PaperRunServiceError(
                str(result.get("error") or "Paper run failed."), details=result
            )
        return self._build_paper_result(slug, result)

    def get_paper_result(self, slug: str) -> PaperRunResult:
        """Read the latest persisted paper run result for a strategy."""
        paths = self.strategy_service.get_strategy_paths(slug)
        if not paths.paper_run_result.exists():
            raise PaperRunServiceError(f"Paper run result for strategy '{slug}' not found.")
        with open(paths.paper_run_result, encoding="utf-8") as file:
            result: dict[str, Any] = json.load(file)
        return self._build_paper_result(slug, result)

    def _build_paper_result(self, slug: str, result: dict[str, Any]) -> PaperRunResult:
        paths = self.strategy_service.get_strategy_paths(slug)
        strategy = self.strategy_service.get_strategy(slug)
        return PaperRunResult(
            strategy=strategy,
            result_path=paths.paper_run_result,
            result=result,
        )
