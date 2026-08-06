"""Backtest routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from autostrategy.api.dependencies import get_backtest_job_service, get_backtest_service
from autostrategy.api.schemas import BacktestJobResponse, BacktestResponse, StrategyResponse
from autostrategy.services.backtest_job_service import BacktestJobService
from autostrategy.services.backtest_service import BacktestService
from autostrategy.services.exceptions import BacktestServiceError
from autostrategy.services.models import BacktestJob

router = APIRouter(tags=["backtest"])


def _job_response(job: BacktestJob) -> BacktestJobResponse:
    return BacktestJobResponse(**job.model_dump())


@router.post("/strategies/{slug}/backtest", response_model=BacktestJobResponse, status_code=status.HTTP_202_ACCEPTED)
def run_backtest(
    slug: str,
    response: Response,
    service: BacktestJobService = Depends(get_backtest_job_service),
) -> BacktestJobResponse:
    """Start a backtest job for a strategy."""
    job = service.submit_backtest(slug)
    if job.status in {"succeeded", "failed", "timed_out"}:
        response.status_code = status.HTTP_200_OK
    return _job_response(job)


@router.get("/strategies/{slug}/backtest-jobs/{job_id}", response_model=BacktestJobResponse)
def get_backtest_job(
    slug: str,
    job_id: str,
    service: BacktestJobService = Depends(get_backtest_job_service),
) -> BacktestJobResponse:
    """Read a backtest job state."""
    try:
        return _job_response(service.get_job(slug, job_id))
    except FileNotFoundError as exc:
        raise BacktestServiceError(str(exc)) from exc


@router.get("/strategies/{slug}/backtest-result", response_model=BacktestResponse)
def get_backtest_result(
    slug: str,
    service: BacktestService = Depends(get_backtest_service),
) -> BacktestResponse:
    """Read the latest backtest result for a strategy."""
    result = service.get_backtest_result(slug)
    return BacktestResponse(
        strategy=StrategyResponse(**result.strategy.model_dump()),
        result_path=result.result_path,
        score=result.score,
        result=result.result,
    )
