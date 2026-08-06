"""Configuration routes."""

from __future__ import annotations

from fastapi import APIRouter

from autostrategy.api.schemas import ConfigResponse, LLMConfigUpdateRequest
from autostrategy.config import LLMConfig, get_llm_api_key_status, load_settings, save_settings

router = APIRouter(tags=["config"])


def build_config_response() -> ConfigResponse:
    """Build safe configuration response with derived readiness."""
    settings = load_settings()
    status = get_llm_api_key_status(settings.llm)
    return ConfigResponse(
        version=settings.version,
        default_market=settings.default_market,
        llm_provider=settings.llm.provider,
        llm_model=settings.llm.model,
        llm_base_url=settings.llm.base_url,
        llm_api_key_env=settings.llm.api_key_env,
        llm_ready=status.ready,
        llm_missing_api_key=status.missing_api_key,
        llm_setup_hint=status.setup_hint,
        llm_checked_env_vars=status.checked_env_vars,
    )


@router.get("/config", response_model=ConfigResponse)
def get_config() -> ConfigResponse:
    """Return safe configuration summary without API key values."""
    return build_config_response()


@router.put("/config/llm", response_model=ConfigResponse)
def update_llm_config(request: LLMConfigUpdateRequest) -> ConfigResponse:
    """Update non-secret LLM configuration fields."""
    settings = load_settings()
    settings.llm = LLMConfig(
        provider=request.provider,
        model=request.model,
        base_url=request.base_url,
        api_key_env=request.api_key_env,
        temperature=settings.llm.temperature,
    )
    save_settings(settings)
    return build_config_response()
