"""FT client simulation routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, status
from pydantic import BaseModel, Field

from autostrategy.api.dependencies import get_client_simulation_service
from autostrategy.brokers.models import FtAccount
from autostrategy.services.client_simulation_service import (
    ClientSimulationRequest,
    ClientSimulationService,
    ClientSimulationSession,
    FtConnectionInput,
    PreflightResult,
)

router = APIRouter(tags=["client-simulation"])


class IntentRejectRequest(BaseModel):
    reason: str = Field(default="user_rejected", min_length=1, max_length=500)


@router.post(
    "/broker-connections/ft-client/check",
    response_model=PreflightResult,
)
def check_ft_client_connection(
    connection: FtConnectionInput | None = Body(default=None),
    service: ClientSimulationService = Depends(get_client_simulation_service),
) -> PreflightResult:
    """Validate customer-entered local FT credentials without persisting them."""
    return service.check_connection(connection)


@router.get(
    "/broker-connections/ft-client/accounts",
    response_model=list[FtAccount],
)
def list_ft_client_accounts(
    service: ClientSimulationService = Depends(get_client_simulation_service),
) -> list[FtAccount]:
    """Return only accounts present in the in-memory simulation allow-list."""
    return service.list_accounts()


@router.post(
    "/strategies/{slug}/client-simulation/preflight",
    response_model=PreflightResult,
)
def preflight_client_simulation(
    slug: str,
    request: ClientSimulationRequest,
    service: ClientSimulationService = Depends(get_client_simulation_service),
) -> PreflightResult:
    return service.preflight(slug, request)


@router.post(
    "/strategies/{slug}/client-simulation/sessions",
    response_model=ClientSimulationSession,
    status_code=status.HTTP_201_CREATED,
)
def create_client_simulation_session(
    slug: str,
    request: ClientSimulationRequest,
    service: ClientSimulationService = Depends(get_client_simulation_service),
) -> ClientSimulationSession:
    return service.create_session(slug, request)


@router.get(
    "/strategies/{slug}/client-simulation/sessions",
    response_model=list[ClientSimulationSession],
)
def list_client_simulation_sessions(
    slug: str,
    service: ClientSimulationService = Depends(get_client_simulation_service),
) -> list[ClientSimulationSession]:
    return service.list_sessions(slug)


@router.get(
    "/strategies/{slug}/client-simulation/sessions/{session_id}",
    response_model=ClientSimulationSession,
)
def get_client_simulation_session(
    slug: str,
    session_id: str,
    service: ClientSimulationService = Depends(get_client_simulation_service),
) -> ClientSimulationSession:
    return service.get_session(slug, session_id)


@router.post(
    "/strategies/{slug}/client-simulation/sessions/{session_id}/pause",
    response_model=ClientSimulationSession,
)
def pause_client_simulation_session(
    slug: str,
    session_id: str,
    service: ClientSimulationService = Depends(get_client_simulation_service),
) -> ClientSimulationSession:
    return service.pause_session(slug, session_id)


@router.post(
    "/strategies/{slug}/client-simulation/sessions/{session_id}/resume",
    response_model=ClientSimulationSession,
)
def resume_client_simulation_session(
    slug: str,
    session_id: str,
    service: ClientSimulationService = Depends(get_client_simulation_service),
) -> ClientSimulationSession:
    return service.resume_session(slug, session_id)


@router.post(
    "/strategies/{slug}/client-simulation/sessions/{session_id}/stop",
    response_model=ClientSimulationSession,
)
def stop_client_simulation_session(
    slug: str,
    session_id: str,
    service: ClientSimulationService = Depends(get_client_simulation_service),
) -> ClientSimulationSession:
    return service.stop_session(slug, session_id)


@router.post(
    "/strategies/{slug}/client-simulation/sessions/{session_id}/intents/{intent_id}/approve",
    response_model=ClientSimulationSession,
)
def approve_client_simulation_intent(
    slug: str,
    session_id: str,
    intent_id: str,
    service: ClientSimulationService = Depends(get_client_simulation_service),
) -> ClientSimulationSession:
    return service.approve_intent(slug, session_id, intent_id)


@router.post(
    "/strategies/{slug}/client-simulation/sessions/{session_id}/intents/{intent_id}/reject",
    response_model=ClientSimulationSession,
)
def reject_client_simulation_intent(
    slug: str,
    session_id: str,
    intent_id: str,
    request: IntentRejectRequest,
    service: ClientSimulationService = Depends(get_client_simulation_service),
) -> ClientSimulationSession:
    return service.reject_intent(slug, session_id, intent_id, request.reason)


@router.get(
    "/strategies/{slug}/client-simulation/sessions/{session_id}/events",
    response_model=list[dict[str, Any]],
)
def get_client_simulation_events(
    slug: str,
    session_id: str,
    service: ClientSimulationService = Depends(get_client_simulation_service),
) -> list[dict[str, Any]]:
    return service.get_events(slug, session_id)


@router.get(
    "/strategies/{slug}/client-simulation/sessions/{session_id}/account",
    response_model=dict[str, Any],
)
def get_client_simulation_account(
    slug: str,
    session_id: str,
    service: ClientSimulationService = Depends(get_client_simulation_service),
) -> dict[str, Any]:
    return service.get_account_snapshot(slug, session_id)
