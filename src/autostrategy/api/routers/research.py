"""Immutable strategy research lifecycle routes."""

from fastapi import APIRouter, Depends

from autostrategy.api.dependencies import (
    get_dataset_manifest_service,
    get_experiment_service,
    get_version_service,
)
from autostrategy.api.schemas import (
    DatasetManifestCreateRequest,
    DatasetManifestResponse,
    ExperimentCreateRequest,
    ExperimentOOSRequest,
    ExperimentOptimizeRequest,
    ExperimentResponse,
    ResearchDecisionRequest,
    StrategyVersionResponse,
    VersionEventResponse,
)
from autostrategy.services.dataset_manifest_service import DatasetManifestService
from autostrategy.services.experiment_service import ExperimentService
from autostrategy.services.models import OptimizationCandidate
from autostrategy.services.version_service import VersionService

router = APIRouter(tags=["research"])


@router.get(
    "/strategies/{slug}/versions",
    response_model=list[StrategyVersionResponse],
)
def list_strategy_versions(
    slug: str,
    service: VersionService = Depends(get_version_service),
) -> list[StrategyVersionResponse]:
    """List immutable versions, lazily snapshotting a legacy workspace."""
    service.ensure_current_version(slug)
    return [
        StrategyVersionResponse(**version.model_dump())
        for version in service.list_versions(slug)
    ]


@router.get(
    "/strategies/{slug}/versions/{version_id}",
    response_model=StrategyVersionResponse,
)
def get_strategy_version(
    slug: str,
    version_id: str,
    service: VersionService = Depends(get_version_service),
) -> StrategyVersionResponse:
    return StrategyVersionResponse(**service.get_version(slug, version_id).model_dump())


@router.post(
    "/strategies/{slug}/versions/{version_id}/rollback",
    response_model=StrategyVersionResponse,
)
def rollback_strategy_version(
    slug: str,
    version_id: str,
    request: ResearchDecisionRequest,
    service: ExperimentService = Depends(get_experiment_service),
) -> StrategyVersionResponse:
    version = service.rollback(slug, version_id, reason=request.reason)
    return StrategyVersionResponse(**version.model_dump())


@router.get(
    "/strategies/{slug}/version-events",
    response_model=list[VersionEventResponse],
)
def list_version_events(
    slug: str,
    service: ExperimentService = Depends(get_experiment_service),
) -> list[VersionEventResponse]:
    service.version_service.ensure_current_version(slug)
    return [
        VersionEventResponse(**event.model_dump())
        for event in service.store.list_version_events(slug)
    ]


@router.post(
    "/strategies/{slug}/dataset-manifests",
    response_model=DatasetManifestResponse,
)
def capture_dataset_manifest(
    slug: str,
    request: DatasetManifestCreateRequest,
    service: DatasetManifestService = Depends(get_dataset_manifest_service),
) -> DatasetManifestResponse:
    manifest = service.capture(
        slug,
        request.version_id,
        train=request.train,
        validation=request.validation,
        test=request.test,
        benchmark=request.benchmark,
        data_source=request.data_source,
        frequency=request.frequency,
        adjustment=request.adjustment,
        commission=request.commission,
        slippage=request.slippage,
    )
    return DatasetManifestResponse(**manifest.model_dump())


@router.get(
    "/strategies/{slug}/dataset-manifests",
    response_model=list[DatasetManifestResponse],
)
def list_dataset_manifests(
    slug: str,
    service: DatasetManifestService = Depends(get_dataset_manifest_service),
) -> list[DatasetManifestResponse]:
    return [
        DatasetManifestResponse(**manifest.model_dump())
        for manifest in service.list_manifests(slug)
    ]


@router.get(
    "/strategies/{slug}/dataset-manifests/{manifest_id}",
    response_model=DatasetManifestResponse,
)
def get_dataset_manifest(
    slug: str,
    manifest_id: str,
    service: DatasetManifestService = Depends(get_dataset_manifest_service),
) -> DatasetManifestResponse:
    return DatasetManifestResponse(**service.get_manifest(slug, manifest_id).model_dump())


@router.post(
    "/strategies/{slug}/experiments",
    response_model=ExperimentResponse,
)
def create_experiment(
    slug: str,
    request: ExperimentCreateRequest,
    service: ExperimentService = Depends(get_experiment_service),
) -> ExperimentResponse:
    session = service.create_session(slug, request.base_version_id, request.manifest_id)
    return ExperimentResponse(**session.model_dump())


@router.get(
    "/strategies/{slug}/experiments",
    response_model=list[ExperimentResponse],
)
def list_experiments(
    slug: str,
    service: ExperimentService = Depends(get_experiment_service),
) -> list[ExperimentResponse]:
    return [ExperimentResponse(**session.model_dump()) for session in service.list_sessions(slug)]


@router.get(
    "/strategies/{slug}/experiments/{session_id}",
    response_model=ExperimentResponse,
)
def get_experiment(
    slug: str,
    session_id: str,
    service: ExperimentService = Depends(get_experiment_service),
) -> ExperimentResponse:
    return ExperimentResponse(**service.get_session(slug, session_id).model_dump())


@router.post(
    "/strategies/{slug}/experiments/{session_id}/baseline",
    response_model=ExperimentResponse,
)
def run_experiment_baseline(
    slug: str,
    session_id: str,
    service: ExperimentService = Depends(get_experiment_service),
) -> ExperimentResponse:
    return ExperimentResponse(**service.run_baseline(slug, session_id).model_dump())


@router.post(
    "/strategies/{slug}/experiments/{session_id}/diagnose",
    response_model=ExperimentResponse,
)
def diagnose_experiment(
    slug: str,
    session_id: str,
    service: ExperimentService = Depends(get_experiment_service),
) -> ExperimentResponse:
    return ExperimentResponse(**service.diagnose(slug, session_id).model_dump())


@router.post(
    "/strategies/{slug}/experiments/{session_id}/optimize",
    response_model=ExperimentResponse,
)
def optimize_experiment(
    slug: str,
    session_id: str,
    request: ExperimentOptimizeRequest,
    service: ExperimentService = Depends(get_experiment_service),
) -> ExperimentResponse:
    candidates = (
        [OptimizationCandidate(**candidate.model_dump()) for candidate in request.candidates]
        if request.candidates is not None
        else None
    )
    session = service.optimize(
        slug,
        session_id,
        candidates,
        minimum_improvement=request.minimum_improvement,
        minimum_trades=request.minimum_trades,
        maximum_drawdown=request.maximum_drawdown,
    )
    return ExperimentResponse(**session.model_dump())


@router.post(
    "/strategies/{slug}/experiments/{session_id}/validate-oos",
    response_model=ExperimentResponse,
)
def validate_experiment_oos(
    slug: str,
    session_id: str,
    request: ExperimentOOSRequest,
    service: ExperimentService = Depends(get_experiment_service),
) -> ExperimentResponse:
    session = service.validate_oos(
        slug,
        session_id,
        minimum_trades=request.minimum_trades,
        maximum_drawdown=request.maximum_drawdown,
        maximum_score_degradation=request.maximum_score_degradation,
    )
    return ExperimentResponse(**session.model_dump())


@router.post(
    "/strategies/{slug}/experiments/{session_id}/accept",
    response_model=ExperimentResponse,
)
def accept_experiment(
    slug: str,
    session_id: str,
    request: ResearchDecisionRequest,
    service: ExperimentService = Depends(get_experiment_service),
) -> ExperimentResponse:
    return ExperimentResponse(
        **service.accept(slug, session_id, reason=request.reason).model_dump()
    )


@router.post(
    "/strategies/{slug}/experiments/{session_id}/reject",
    response_model=ExperimentResponse,
)
def reject_experiment(
    slug: str,
    session_id: str,
    request: ResearchDecisionRequest,
    service: ExperimentService = Depends(get_experiment_service),
) -> ExperimentResponse:
    return ExperimentResponse(
        **service.reject(slug, session_id, reason=request.reason).model_dump()
    )
