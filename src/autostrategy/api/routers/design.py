"""Design routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from autostrategy.api.dependencies import get_design_service
from autostrategy.api.schemas import DesignCreateRequest, DesignResponse, StrategyResponse
from autostrategy.services.design_service import DesignService

router = APIRouter(tags=["design"])


@router.post("/designs", response_model=DesignResponse)
def create_design(
    request: DesignCreateRequest,
    service: DesignService = Depends(get_design_service),
) -> DesignResponse:
    """Create a strategy and generate its design document."""
    result = service.create_design(
        name=request.name,
        prompt=request.prompt,
        market=request.market,
        template=request.template,
    )
    return DesignResponse(
        strategy=StrategyResponse(**result.strategy.model_dump()),
        design_path=result.design_path,
    )
