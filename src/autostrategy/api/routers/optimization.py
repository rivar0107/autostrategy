"""Configuration optimization ratchet routes."""

from fastapi import APIRouter, Depends

from autostrategy.api.dependencies import get_optimization_service
from autostrategy.api.schemas import (
    OptimizationAcceptRequest,
    OptimizationEvaluateRequest,
    OptimizationReportResponse,
)
from autostrategy.services.models import OptimizationCandidate
from autostrategy.services.optimization_service import OptimizationService

router = APIRouter(tags=["optimization"])


@router.post(
    "/strategies/{slug}/optimizations",
    response_model=OptimizationReportResponse,
)
def evaluate_optimization(
    slug: str,
    request: OptimizationEvaluateRequest,
    service: OptimizationService = Depends(get_optimization_service),
) -> OptimizationReportResponse:
    """Evaluate config candidates without mutating the live strategy."""
    report = service.evaluate(
        slug,
        [OptimizationCandidate(**candidate.model_dump()) for candidate in request.candidates],
        minimum_improvement=request.minimum_improvement,
    )
    return OptimizationReportResponse(**report.model_dump())


@router.get(
    "/strategies/{slug}/optimizations/latest",
    response_model=OptimizationReportResponse,
)
def get_latest_optimization(
    slug: str,
    service: OptimizationService = Depends(get_optimization_service),
) -> OptimizationReportResponse:
    """Read the latest optimization report."""
    return OptimizationReportResponse(**service.get_latest_report(slug).model_dump())


@router.post(
    "/strategies/{slug}/optimizations/{report_id}/accept",
    response_model=OptimizationReportResponse,
)
def accept_optimization(
    slug: str,
    report_id: str,
    request: OptimizationAcceptRequest,
    service: OptimizationService = Depends(get_optimization_service),
) -> OptimizationReportResponse:
    """Explicitly apply one eligible and fresh candidate."""
    report = service.accept(slug, report_id, request.candidate_name)
    return OptimizationReportResponse(**report.model_dump())
