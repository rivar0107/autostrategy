"""Pydantic models returned by autostrategy services."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from autostrategy.core.strategy import Strategy, StrategyStatus


class StrategySummary(BaseModel):
    """Compact strategy information for lists and API responses."""

    name: str
    slug: str
    description: str = ""
    market: str
    status: StrategyStatus
    template: str | None = None
    tags: list[str] = Field(default_factory=list)

    @classmethod
    def from_strategy(cls, strategy: Strategy) -> "StrategySummary":
        """Build a summary from a domain strategy model."""
        return cls(
            name=strategy.name,
            slug=strategy.slug,
            description=strategy.description,
            market=strategy.market,
            status=strategy.status,
            template=strategy.template,
            tags=strategy.tags,
        )


class StrategyPaths(BaseModel):
    """Important file paths for a strategy workspace."""

    workspace: Path
    metadata: Path
    design: Path
    strategy_code: Path
    config: Path
    readme: Path
    backtest_result: Path
    paper_run_result: Path
    paper_run_events: Path
    paper_run_log: Path


class StrategyDetail(BaseModel):
    """Strategy detail plus its local paths."""

    strategy: StrategySummary
    paths: StrategyPaths


class DesignResult(BaseModel):
    """Result of a design generation operation."""

    strategy: StrategySummary
    design_path: Path


class CodegenResult(BaseModel):
    """Result of a code generation operation."""

    strategy: StrategySummary
    generated_files: list[str]


class BacktestResult(BaseModel):
    """Result of a backtest operation."""

    strategy: StrategySummary
    result_path: Path
    score: float
    result: dict[str, Any]


class PaperRunResult(BaseModel):
    """Result of a paper replay operation."""

    strategy: StrategySummary
    result_path: Path
    result: dict[str, Any]


BacktestJobStatus = Literal["queued", "running", "succeeded", "failed", "timed_out", "stopped"]


class BacktestJob(BaseModel):
    """State of a local backtest job."""

    job_id: str
    slug: str
    status: BacktestJobStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result_path: Path | None = None
    score: float | None = None
    error: str | None = None
    stop_requested: bool = False


class AppInfo(BaseModel):
    """Safe application info for local API/UI clients."""

    version: str
    workspace_root: Path
    templates: list[str]
    llm_provider: str
    llm_model: str
