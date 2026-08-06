"""Workspace management for strategies."""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from autostrategy.core.strategy import Strategy, StrategyStatus
from autostrategy.core.template_registry import TemplateRegistry


DEFAULT_WORKSPACE_ROOT = Path.home() / ".autostrategy" / "strategies"


class Workspace:
    """Manages a directory of strategy workspaces."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or DEFAULT_WORKSPACE_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    def _strategy_dir(self, slug: str) -> Path:
        return self.root / slug

    def create_strategy(
        self,
        name: str,
        description: str = "",
        market: str = "A股",
        template: str | None = None,
        tags: list[str] | None = None,
    ) -> Strategy:
        """Create a new strategy workspace."""
        strategy = Strategy(
            name=name,
            description=description,
            market=market,
            template=template,
            tags=tags or [],
        )
        strategy_dir = self._strategy_dir(strategy.slug)
        if strategy_dir.exists():
            raise FileExistsError(f"Strategy '{name}' ({strategy.slug}) already exists.")

        strategy_dir.mkdir(parents=True)
        self._write_strategy_meta(strategy_dir, strategy)
        self._write_default_files(strategy_dir)
        if template:
            TemplateRegistry.apply_template(template, strategy_dir)
        return strategy

    def _write_strategy_meta(self, strategy_dir: Path, strategy: Strategy) -> None:
        meta_path = strategy_dir / "strategy.yaml"
        with open(meta_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(strategy.model_dump(mode="json"), f, allow_unicode=True)

    def _write_default_files(self, strategy_dir: Path) -> None:
        config_path = strategy_dir / "config.yaml"
        config_path.write_text(
            "# Strategy configuration\n"
            "market: A股\n"
            "symbols: []\n"
            "period:\n"
            "  start: 2022-01-01\n"
            "  end: 2024-01-01\n",
            encoding="utf-8",
        )
        design_path = strategy_dir / "STRATEGY_DESIGN.md"
        design_path.write_text(
            f"# {strategy_dir.name}\n\n"
            "## 策略概述\n\n"
            "待补充...\n\n"
            "## 买入条件\n\n"
            "- 待补充\n\n"
            "## 卖出条件\n\n"
            "- 待补充\n",
            encoding="utf-8",
        )

    def list_strategies(self) -> list[Strategy]:
        """List all strategies in the workspace."""
        strategies = []
        for entry in sorted(self.root.iterdir()):
            if entry.is_dir():
                strategy = self._load_strategy(entry)
                if strategy:
                    strategies.append(strategy)
        return strategies

    def get_strategy(self, slug: str) -> Strategy | None:
        """Get a strategy by slug."""
        strategy_dir = self._strategy_dir(slug)
        if not strategy_dir.exists():
            return None
        return self._load_strategy(strategy_dir)

    def _load_strategy(self, strategy_dir: Path) -> Strategy | None:
        meta_path = strategy_dir / "strategy.yaml"
        if not meta_path.exists():
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return Strategy(**data)

    def delete_strategy(self, slug: str) -> None:
        """Delete a strategy workspace."""
        strategy_dir = self._strategy_dir(slug)
        if strategy_dir.exists():
            shutil.rmtree(strategy_dir)

    def get_strategy_dir(self, slug: str) -> Path:
        """Return the absolute directory path for an existing strategy."""
        strategy_dir = self._strategy_dir(slug)
        if not strategy_dir.exists():
            raise FileNotFoundError(f"Strategy '{slug}' not found.")
        return strategy_dir

    def resolve_strategy_path(self, slug: str, relative_path: str | Path) -> Path:
        """Resolve a path inside a strategy workspace safely.

        The API/MCP layers may pass user-controlled paths. This method keeps
        file access constrained to the selected strategy directory.
        """
        path = Path(relative_path)
        if path.is_absolute() or str(relative_path).strip() == "":
            raise ValueError(f"Unsafe strategy file path: {relative_path}")

        strategy_dir = self.get_strategy_dir(slug).resolve()
        candidate = (strategy_dir / path).resolve()
        if candidate != strategy_dir and strategy_dir not in candidate.parents:
            raise ValueError(f"Unsafe strategy file path: {relative_path}")
        return candidate

    def read_text_file(self, slug: str, relative_path: str) -> str:
        """Read a UTF-8 text file from a strategy workspace."""
        file_path = self.resolve_strategy_path(slug, relative_path)
        return file_path.read_text(encoding="utf-8")

    def write_text_file(self, slug: str, relative_path: str, content: str) -> Path:
        """Write a UTF-8 text file inside a strategy workspace."""
        file_path = self.resolve_strategy_path(slug, relative_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return file_path

    def update_strategy_status(self, slug: str, status: StrategyStatus) -> Strategy:
        """Update the status of a strategy."""
        strategy = self.get_strategy(slug)
        if not strategy:
            raise FileNotFoundError(f"Strategy '{slug}' not found.")
        strategy.status = status
        strategy_dir = self._strategy_dir(slug)
        self._write_strategy_meta(strategy_dir, strategy)
        return strategy
