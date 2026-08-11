"""Design routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from autostrategy.api.dependencies import get_design_job_service
from autostrategy.api.schemas import DesignCreateRequest, DesignJobStatusResponse, StrategyResponse
from autostrategy.services.design_job_service import DesignJobService
from autostrategy.services.exceptions import JobNotFoundError
from autostrategy.services.models import DesignJob

router = APIRouter(tags=["design"])


def _job_response(job: DesignJob) -> DesignJobStatusResponse:
    data = job.model_dump()
    if data.get("strategy") is not None:
        data["strategy"] = StrategyResponse(**data["strategy"])
    return DesignJobStatusResponse(**data)


@router.post(
    "/designs",
    response_model=DesignJobStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_design(
    request: DesignCreateRequest,
    response: Response,
    service: DesignJobService = Depends(get_design_job_service),
) -> DesignJobStatusResponse:
    """Start a design generation job."""
    job = service.submit_design(
        name=request.name,
        prompt=request.prompt,
        market=request.market,
        template=request.template,
    )
    if job.status in {"succeeded", "failed"}:
        response.status_code = status.HTTP_200_OK
    return _job_response(job)


@router.get("/design-jobs/{job_id}", response_model=DesignJobStatusResponse)
def get_design_job(
    job_id: str,
    service: DesignJobService = Depends(get_design_job_service),
) -> DesignJobStatusResponse:
    """Read a design generation job state."""
    try:
        return _job_response(service.get_job(job_id))
    except FileNotFoundError as exc:
        raise JobNotFoundError(str(exc)) from exc
