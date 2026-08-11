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
    version: int = 1
    content_digest: str = ""
    current_version_id: str | None = None
    active_version_id: str | None = None

    @classmethod
    def from_strategy(cls, strategy: Strategy) -> StrategySummary:
        """Build a summary from a domain strategy model."""
        return cls(
            name=strategy.name,
            slug=strategy.slug,
            description=strategy.description,
            market=strategy.market,
            status=strategy.status,
            template=strategy.template,
            tags=strategy.tags,
            version=strategy.version,
            content_digest=strategy.content_digest,
            current_version_id=strategy.current_version_id,
            active_version_id=strategy.active_version_id,
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


class BacktestRunSummary(BaseModel):
    """Immutable metadata for one persisted backtest execution."""

    run_id: str
    strategy_slug: str
    strategy_version: int
    strategy_digest: str
    created_at: str
    score: float
    result_path: Path
    version_id: str | None = None
    manifest_id: str | None = None
    session_id: str | None = None
    phase: Literal["full", "train", "validation", "test"] = "full"
    candidate_id: str | None = None


class BacktestRunRecord(BacktestRunSummary):
    """Persisted backtest metadata and its validated result snapshot."""

    result: dict[str, Any]


class OptimizationCandidate(BaseModel):
    """One named set of configuration overrides to evaluate."""

    name: str
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class OptimizationCandidateResult(BaseModel):
    """Evaluation outcome for one isolated candidate."""

    name: str
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    status: Literal["evaluated", "failed"]
    score: float | None = None
    improvement: float | None = None
    eligible: bool = False
    error: str | None = None


class OptimizationReport(BaseModel):
    """Persisted, human-approved optimization ratchet report."""

    report_id: str
    strategy_slug: str
    strategy_version: int
    strategy_digest: str
    base_version_id: str | None = None
    created_at: str
    baseline_score: float
    minimum_improvement: float
    candidates: list[OptimizationCandidateResult]
    recommended_candidate: str | None = None
    accepted: bool = False
    accepted_candidate: str | None = None
    accepted_version_id: str | None = None
    accepted_at: str | None = None


class PaperRunResult(BaseModel):
    """Result of a paper replay operation."""

    strategy: StrategySummary
    result_path: Path
    result: dict[str, Any]


DesignJobStatus = Literal["queued", "running", "succeeded", "failed"]


class DesignJob(BaseModel):
    """State of a design generation job."""

    job_id: str
    name: str
    status: DesignJobStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    strategy: StrategySummary | None = None
    design_path: Path | None = None
    error: str | None = None
    error_code: str | None = None


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
