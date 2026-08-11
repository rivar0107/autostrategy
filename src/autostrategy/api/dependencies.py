"""FastAPI dependencies."""

from __future__ import annotations

from pathlib import Path

from fastapi import Request

from autostrategy.config import load_settings
from autostrategy.services import (
    BacktestJobService,
    BacktestService,
    ClientSimulationService,
    CodegenService,
    DatasetManifestService,
    DesignJobService,
    DesignService,
    ExperimentService,
    OptimizationService,
    PaperRunJobService,
    PaperRunService,
    StrategyService,
    VersionService,
)
from autostrategy.services.client_simulation_service import (
    FtshareTenMinuteMarketContextProvider,
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


def get_design_job_service(request: Request) -> DesignJobService:
    """Return app-level design job service."""
    service = getattr(request.app.state, "design_job_service", None)
    if service is None:
        settings = load_settings()
        service = DesignJobService(
            workspace_root=get_workspace_root(request), llm_config=settings.llm
        )
        request.app.state.design_job_service = service
    return service


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


def get_optimization_service(request: Request) -> OptimizationService:
    """Build the safe configuration optimization service."""
    return OptimizationService(workspace_root=get_workspace_root(request))


def get_version_service(request: Request) -> VersionService:
    """Build immutable strategy version service."""
    return VersionService(workspace_root=get_workspace_root(request))


def get_dataset_manifest_service(request: Request) -> DatasetManifestService:
    """Build frozen dataset manifest service."""
    return DatasetManifestService(workspace_root=get_workspace_root(request))


def get_experiment_service(request: Request) -> ExperimentService:
    """Build persistent strategy experiment service."""
    return ExperimentService(workspace_root=get_workspace_root(request))


def get_paper_run_service(request: Request) -> PaperRunService:
    """Build paper run service for the current request."""
    return PaperRunService(workspace_root=get_workspace_root(request))


def get_paper_run_job_service(request: Request) -> PaperRunJobService:
    """Return app-level paper run job service."""
    return request.app.state.paper_run_job_service


def get_client_simulation_service(request: Request) -> ClientSimulationService:
    """Return the app-level FT client simulation service."""
    service = getattr(request.app.state, "client_simulation_service", None)
    if service is None:
        settings = load_settings()
        service = ClientSimulationService(
            workspace_root=get_workspace_root(request),
            config=settings.broker_connections.ft_client,
            market_context_provider=FtshareTenMinuteMarketContextProvider(),
            enable_background_reconciliation=True,
            market_poll_interval_seconds=30.0,
        )
        request.app.state.client_simulation_service = service
    return service
