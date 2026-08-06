"""Strategy workspace service."""

from __future__ import annotations

from pathlib import Path

from autostrategy.core.template_registry import TemplateRegistry
from autostrategy.core.workspace import Workspace
from autostrategy.services.exceptions import StrategyAlreadyExistsError, StrategyNotFoundError
from autostrategy.services.models import StrategyDetail, StrategyPaths, StrategySummary


class StrategyService:
    """Application service for strategy workspace operations."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace = Workspace(root=workspace_root)

    @property
    def workspace_root(self) -> Path:
        """Return the workspace root directory."""
        return self.workspace.root

    def list_strategies(self) -> list[StrategySummary]:
        """List strategies in the workspace."""
        return [StrategySummary.from_strategy(strategy) for strategy in self.workspace.list_strategies()]

    def list_templates(self) -> list[str]:
        """List built-in strategy templates."""
        return TemplateRegistry.list_templates()

    def create_strategy(
        self,
        name: str,
        description: str = "",
        market: str = "A股",
        template: str | None = None,
        tags: list[str] | None = None,
    ) -> StrategySummary:
        """Create a strategy workspace."""
        try:
            strategy = self.workspace.create_strategy(
                name=name,
                description=description,
                market=market,
                template=template,
                tags=tags,
            )
        except FileExistsError as exc:
            raise StrategyAlreadyExistsError(str(exc)) from exc
        return StrategySummary.from_strategy(strategy)

    def get_strategy(self, slug: str) -> StrategySummary:
        """Get a strategy by slug."""
        strategy = self.workspace.get_strategy(slug)
        if strategy is None:
            raise StrategyNotFoundError(f"Strategy '{slug}' not found.")
        return StrategySummary.from_strategy(strategy)

    def get_strategy_detail(self, slug: str) -> StrategyDetail:
        """Get a strategy and its key paths."""
        return StrategyDetail(strategy=self.get_strategy(slug), paths=self.get_strategy_paths(slug))

    def get_strategy_paths(self, slug: str) -> StrategyPaths:
        """Return important local paths for a strategy."""
        try:
            strategy_dir = self.workspace.get_strategy_dir(slug)
        except FileNotFoundError as exc:
            raise StrategyNotFoundError(str(exc)) from exc
        return StrategyPaths(
            workspace=strategy_dir,
            metadata=strategy_dir / "strategy.yaml",
            design=strategy_dir / "STRATEGY_DESIGN.md",
            strategy_code=strategy_dir / "strategy.py",
            config=strategy_dir / "config.yaml",
            readme=strategy_dir / "README.md",
            backtest_result=strategy_dir / "backtest" / "results" / "backtest_result.json",
            paper_run_result=strategy_dir / "paper_run" / "results" / "paper_run_result.json",
            paper_run_events=strategy_dir / "paper_run" / "results" / "paper_run_events.jsonl",
            paper_run_log=strategy_dir / "paper_run" / "logs" / "paper_run.log",
        )

    def delete_strategy(self, slug: str) -> None:
        """Delete a strategy workspace."""
        self.get_strategy(slug)
        self.workspace.delete_strategy(slug)
