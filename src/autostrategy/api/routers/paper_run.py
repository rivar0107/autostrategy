"""Paper run routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from autostrategy.api.dependencies import get_paper_run_job_service, get_paper_run_service
from autostrategy.api.schemas import BacktestJobResponse, PaperRunResponse, StrategyResponse
from autostrategy.services.exceptions import PaperRunServiceError
from autostrategy.services.models import BacktestJob
from autostrategy.services.paper_run_job_service import PaperRunJobService
from autostrategy.services.paper_run_service import PaperRunService

router = APIRouter(tags=["paper-run"])


def _job_response(job: BacktestJob) -> BacktestJobResponse:
    return BacktestJobResponse(**job.model_dump())


@router.post("/strategies/{slug}/paper-run", response_model=BacktestJobResponse, status_code=status.HTTP_202_ACCEPTED)
def start_paper_run(
    slug: str,
    service: PaperRunJobService = Depends(get_paper_run_job_service),
) -> BacktestJobResponse:
    """Start a replay-first paper run job."""
    return _job_response(service.submit_paper_run(slug))


@router.get("/strategies/{slug}/paper-run-jobs/{job_id}", response_model=BacktestJobResponse)
def get_paper_run_job(
    slug: str,
    job_id: str,
    service: PaperRunJobService = Depends(get_paper_run_job_service),
) -> BacktestJobResponse:
    """Read a paper run job state."""
    try:
        return _job_response(service.get_job(slug, job_id))
    except FileNotFoundError as exc:
        raise PaperRunServiceError(str(exc)) from exc


@router.post("/strategies/{slug}/paper-run-jobs/{job_id}/stop", response_model=BacktestJobResponse)
def stop_paper_run_job(
    slug: str,
    job_id: str,
    service: PaperRunJobService = Depends(get_paper_run_job_service),
) -> BacktestJobResponse:
    """Request a paper run job stop."""
    try:
        return _job_response(service.request_stop(slug, job_id))
    except FileNotFoundError as exc:
        raise PaperRunServiceError(str(exc)) from exc


@router.get("/strategies/{slug}/paper-run-result", response_model=PaperRunResponse)
def get_paper_run_result(
    slug: str,
    service: PaperRunService = Depends(get_paper_run_service),
) -> PaperRunResponse:
    """Read the latest paper run result for a strategy."""
    result = service.get_paper_result(slug)
    return PaperRunResponse(
        strategy=StrategyResponse(**result.strategy.model_dump()),
        result_path=result.result_path,
        result=result.result,
    )
