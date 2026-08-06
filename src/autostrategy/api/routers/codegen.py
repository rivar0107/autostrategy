"""Code generation routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from autostrategy.api.dependencies import get_codegen_service
from autostrategy.api.schemas import CodegenRequest, CodegenResponse, StrategyResponse
from autostrategy.services.codegen_service import CodegenService

router = APIRouter(tags=["codegen"])


@router.post("/strategies/{slug}/codegen", response_model=CodegenResponse)
def generate_code(
    slug: str,
    request: CodegenRequest,
    service: CodegenService = Depends(get_codegen_service),
) -> CodegenResponse:
    """Generate executable strategy files from STRATEGY_DESIGN.md."""
    result = service.generate_code(slug=slug, force=request.force)
    return CodegenResponse(
        strategy=StrategyResponse(**result.strategy.model_dump()),
        generated_files=result.generated_files,
    )
