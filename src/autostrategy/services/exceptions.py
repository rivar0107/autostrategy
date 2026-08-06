"""Service layer exceptions."""

from __future__ import annotations

from typing import Any

from autostrategy.config import LLMApiKeyStatus


class AutostrategyServiceError(Exception):
    """Base error raised by service layer operations."""

    code = "service_error"

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class LLMConfigurationRequiredError(AutostrategyServiceError):
    """Raised when an LLM operation needs a configured API key."""

    code = "llm_configuration_required"

    def __init__(self, status: LLMApiKeyStatus, provider: str | None = None) -> None:
        details: dict[str, Any] = {
            "llm_ready": status.ready,
            "llm_missing_api_key": status.missing_api_key,
            "api_key_env": status.api_key_env,
            "checked_env_vars": status.checked_env_vars,
            "setup_hint": status.setup_hint,
        }
        if provider is not None:
            details["provider"] = provider
        super().__init__(
            "LLM API key is not configured.",
            details=details,
        )


class StrategyNotFoundError(AutostrategyServiceError):
    """Raised when a strategy slug cannot be found."""

    code = "strategy_not_found"


class StrategyAlreadyExistsError(AutostrategyServiceError):
    """Raised when creating a strategy that already exists."""

    code = "strategy_already_exists"


class ValidationServiceError(AutostrategyServiceError):
    """Raised when user input or generated content is invalid."""

    code = "validation_error"


class BacktestServiceError(AutostrategyServiceError):
    """Raised when a backtest cannot complete successfully."""

    code = "backtest_error"


class PaperRunServiceError(AutostrategyServiceError):
    """Raised when a paper run cannot complete successfully."""

    code = "paper_run_error"


class ArtifactNotFoundError(AutostrategyServiceError):
    """Raised when a strategy artifact cannot be previewed."""

    code = "artifact_not_found"
