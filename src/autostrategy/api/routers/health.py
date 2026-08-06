"""Health and app info routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from autostrategy import __version__
from autostrategy.api.dependencies import get_strategy_service
from autostrategy.api.schemas import HealthResponse, InfoResponse
from autostrategy.config import load_settings
from autostrategy.services.strategy_service import StrategyService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return API health status."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/info", response_model=InfoResponse)
def info(service: StrategyService = Depends(get_strategy_service)) -> InfoResponse:
    """Return safe application information."""
    settings = load_settings()
    return InfoResponse(
        version=__version__,
        workspace_root=service.workspace_root,
        templates=service.list_templates(),
        llm_provider=settings.llm.provider,
        llm_model=settings.llm.model,
    )
