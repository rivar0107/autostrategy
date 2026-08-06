"""Strategy and template routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from autostrategy.api.dependencies import get_strategy_service
from autostrategy.api.schemas import (
    StrategyCreateRequest,
    StrategyDetailResponse,
    StrategyPathsResponse,
    StrategyResponse,
)
from autostrategy.services.strategy_service import StrategyService

router = APIRouter(tags=["strategies"])


@router.get("/templates", response_model=list[str])
def list_templates(service: StrategyService = Depends(get_strategy_service)) -> list[str]:
    """List built-in strategy templates."""
    return service.list_templates()


@router.get("/strategies", response_model=list[StrategyResponse])
def list_strategies(service: StrategyService = Depends(get_strategy_service)) -> list[StrategyResponse]:
    """List strategies in the workspace."""
    return [StrategyResponse(**strategy.model_dump()) for strategy in service.list_strategies()]


@router.post("/strategies", response_model=StrategyResponse)
def create_strategy(
    request: StrategyCreateRequest,
    service: StrategyService = Depends(get_strategy_service),
) -> StrategyResponse:
    """Create a strategy workspace."""
    strategy = service.create_strategy(
        name=request.name,
        description=request.description,
        market=request.market,
        template=request.template,
        tags=request.tags,
    )
    return StrategyResponse(**strategy.model_dump())


@router.get("/strategies/{slug}", response_model=StrategyDetailResponse)
def get_strategy(
    slug: str,
    service: StrategyService = Depends(get_strategy_service),
) -> StrategyDetailResponse:
    """Get strategy detail and paths."""
    detail = service.get_strategy_detail(slug)
    return StrategyDetailResponse(
        strategy=StrategyResponse(**detail.strategy.model_dump()),
        paths=StrategyPathsResponse(**detail.paths.model_dump()),
    )


@router.delete("/strategies/{slug}", status_code=204)
def delete_strategy(slug: str, service: StrategyService = Depends(get_strategy_service)) -> None:
    """Delete a strategy workspace."""
    service.delete_strategy(slug)


@router.get("/strategies/{slug}/paths", response_model=StrategyPathsResponse)
def get_strategy_paths(
    slug: str,
    service: StrategyService = Depends(get_strategy_service),
) -> StrategyPathsResponse:
    """Get important strategy file paths."""
    return StrategyPathsResponse(**service.get_strategy_paths(slug).model_dump())
