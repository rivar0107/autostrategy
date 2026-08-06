"""FastAPI dependencies."""

from __future__ import annotations

from pathlib import Path

from fastapi import Request

from autostrategy.config import load_settings
from autostrategy.services import (
    BacktestJobService,
    BacktestService,
    CodegenService,
    DesignService,
    PaperRunJobService,
    PaperRunService,
    StrategyService,
)


def get_workspace_root(request: Request) -> Path | None:
    """Return workspace root configured on the FastAPI app."""
    return getattr(request.app.state, "workspace_root", None)


def get_strategy_service(request: Request) -> StrategyService:
    """Build strategy service for the current request."""
    return StrategyService(workspace_root=get_workspace_root(request))


def get_design_service(request: Request) -> DesignService:
    """Build design service for the current request."""
    settings = load_settings()
    return DesignService(workspace_root=get_workspace_root(request), llm_config=settings.llm)


def get_codegen_service(request: Request) -> CodegenService:
    """Build codegen service for the current request."""
    settings = load_settings()
    return CodegenService(workspace_root=get_workspace_root(request), llm_config=settings.llm)


def get_backtest_service(request: Request) -> BacktestService:
    """Build backtest service for the current request."""
    return BacktestService(workspace_root=get_workspace_root(request))


def get_backtest_job_service(request: Request) -> BacktestJobService:
    """Return app-level backtest job service."""
    return request.app.state.backtest_job_service


def get_paper_run_service(request: Request) -> PaperRunService:
    """Build paper run service for the current request."""
    return PaperRunService(workspace_root=get_workspace_root(request))


def get_paper_run_job_service(request: Request) -> PaperRunJobService:
    """Return app-level paper run job service."""
    return request.app.state.paper_run_job_service
