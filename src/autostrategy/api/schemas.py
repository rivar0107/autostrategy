"""API schemas for autostrategy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator


_ALLOWED_LLM_API_KEY_ENVS = {
    "AUTOSTRATEGY_LLM_API_KEY",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "KIMI_API_KEY",
    "QWEN_API_KEY",
    "ZAI_API_KEY",
    "MINIMAX_API_KEY",
    "GEMINI_API_KEY",
    "LOCAL_API_KEY",
}
_ALLOWED_LLM_BASE_URL_HOSTS = {
    "api.openai.com",
    "api.deepseek.com",
    "api.moonshot.cn",
    "dashscope.aliyuncs.com",
    "open.bigmodel.cn",
    "api.minimax.chat",
    "generativelanguage.googleapis.com",
    "localhost",
    "127.0.0.1",
}
_LLM_ENV_NAME_PATTERN = __import__("re").compile(r"^[A-Z][A-Z0-9_]*$")

from autostrategy.core.strategy import StrategyStatus


class ErrorBody(BaseModel):
    """Structured API error body."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Structured API error response."""

    error: ErrorBody


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


class InfoResponse(BaseModel):
    """Application info response."""

    version: str
    workspace_root: Path
    templates: list[str]
    llm_provider: str
    llm_model: str


class ConfigResponse(BaseModel):
    """Safe configuration summary."""

    version: str
    default_market: str
    llm_provider: str
    llm_model: str
    llm_base_url: str | None = None
    llm_api_key_env: str
    llm_ready: bool
    llm_missing_api_key: bool
    llm_setup_hint: str | None = None
    llm_checked_env_vars: list[str] = Field(default_factory=list)


class LLMConfigUpdateRequest(BaseModel):
    """Update safe LLM configuration fields."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str

    @field_validator("provider", "model", "api_key_env")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Reject blank required LLM config fields."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Field cannot be blank.")
        return stripped

    @field_validator("api_key_env")
    @classmethod
    def validate_api_key_env(cls, value: str) -> str:
        """Allow only known API key environment variable names."""
        if not _LLM_ENV_NAME_PATTERN.fullmatch(value):
            raise ValueError("API key environment variable must be uppercase letters, numbers, and underscores.")
        if value not in _ALLOWED_LLM_API_KEY_ENVS:
            raise ValueError("API key environment variable is not in the allowed provider list.")
        return value

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        """Allow only trusted provider or loopback LLM API hosts."""
        if value is None:
            return None
        stripped = value.strip().rstrip("/")
        if not stripped:
            return None
        parsed = urlparse(stripped)
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1"}:
            raise ValueError("Base URL must use HTTPS unless it targets localhost.")
        if parsed.hostname not in _ALLOWED_LLM_BASE_URL_HOSTS:
            raise ValueError("Base URL host is not in the allowed provider list.")
        return stripped


class StrategyResponse(BaseModel):
    """Strategy response model."""

    name: str
    slug: str
    description: str = ""
    market: str
    status: StrategyStatus
    template: str | None = None
    tags: list[str] = Field(default_factory=list)


class StrategyCreateRequest(BaseModel):
    """Create strategy request."""

    name: str
    description: str = ""
    market: str = "A股"
    template: str | None = None
    tags: list[str] = Field(default_factory=list)


class StrategyPathsResponse(BaseModel):
    """Strategy important paths response."""

    workspace: Path
    metadata: Path
    design: Path
    strategy_code: Path
    config: Path
    readme: Path
    backtest_result: Path
    paper_run_result: Path
    paper_run_events: Path
    paper_run_log: Path


class StrategyDetailResponse(BaseModel):
    """Strategy detail response."""

    strategy: StrategyResponse
    paths: StrategyPathsResponse


class DesignCreateRequest(BaseModel):
    """Create design request."""

    name: str
    prompt: str
    market: str = "A股"
    template: str | None = None


class DesignResponse(BaseModel):
    """Design operation response."""

    strategy: StrategyResponse
    design_path: Path


class CodegenRequest(BaseModel):
    """Code generation request."""

    force: bool = False


class CodegenResponse(BaseModel):
    """Code generation response."""

    strategy: StrategyResponse
    generated_files: list[str]


class BacktestResponse(BaseModel):
    """Backtest operation response."""

    strategy: StrategyResponse
    result_path: Path
    score: float
    result: dict[str, Any]


class PaperRunResponse(BaseModel):
    """Paper run result response."""

    strategy: StrategyResponse
    result_path: Path
    result: dict[str, Any]


BacktestJobStatus = Literal["queued", "running", "succeeded", "failed", "timed_out", "stopped"]


class BacktestJobResponse(BaseModel):
    """Backtest job state response."""

    job_id: str
    slug: str
    status: BacktestJobStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result_path: Path | None = None
    score: float | None = None
    error: str | None = None
    stop_requested: bool = False


class ArtifactMetaResponse(BaseModel):
    """Previewable strategy artifact metadata."""

    slug: str
    artifact_key: str
    relative_path: str
    path: Path
    exists: bool
    size: int
    modified_at: str | None = None
    content_type: str


class ArtifactListResponse(BaseModel):
    """List of previewable strategy artifacts."""

    slug: str
    artifacts: list[ArtifactMetaResponse]


class ArtifactContentResponse(ArtifactMetaResponse):
    """Previewable strategy artifact content."""

    content: str
    parsed_json: Any | None = None
