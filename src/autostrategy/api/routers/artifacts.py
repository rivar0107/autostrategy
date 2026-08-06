"""Strategy artifact routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from autostrategy.api.dependencies import get_workspace_root
from autostrategy.api.schemas import ArtifactContentResponse, ArtifactListResponse
from autostrategy.services.artifact_service import ArtifactService

router = APIRouter(tags=["artifacts"])


def get_artifact_service(request: Request) -> ArtifactService:
    """Build artifact service for the current request."""
    return ArtifactService(workspace_root=get_workspace_root(request))


@router.get("/strategies/{slug}/artifacts", response_model=ArtifactListResponse)
def list_artifacts(
    slug: str,
    service: ArtifactService = Depends(get_artifact_service),
) -> ArtifactListResponse:
    """List previewable artifacts for a strategy."""
    return ArtifactListResponse(**service.list_artifacts(slug))


@router.get("/strategies/{slug}/artifacts/{artifact_key}", response_model=ArtifactContentResponse)
def get_artifact(
    slug: str,
    artifact_key: str,
    service: ArtifactService = Depends(get_artifact_service),
) -> ArtifactContentResponse:
    """Read a named strategy artifact for preview."""
    return ArtifactContentResponse(**service.get_artifact(slug, artifact_key))
