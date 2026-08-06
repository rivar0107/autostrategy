"""Configuration management for autostrategy.

Settings are stored in ~/.autostrategy/settings.yaml.
API keys should be stored via keyring or environment variables, not in this file.
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from autostrategy import __version__


DEFAULT_PROVIDER: Literal[
    "openai", "deepseek", "kimi", "qwen", "zai", "minimax", "gemini", "local"
] = "openai"


class LLMApiKeyStatus(BaseModel):
    """Safe LLM API key readiness status."""

    ready: bool
    missing_api_key: bool
    api_key_env: str
    checked_env_vars: list[str]
    setup_hint: str | None = None


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    model_config = ConfigDict(extra="forbid")

    provider: str = DEFAULT_PROVIDER
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key_env: str = "AUTOSTRATEGY_LLM_API_KEY"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @field_validator("provider", "model", "api_key_env")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Reject blank required LLM config fields."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Field cannot be blank.")
        return stripped


class Settings(BaseModel):
    """Top-level application settings."""

    version: str = __version__
    llm: LLMConfig = Field(default_factory=LLMConfig)
    default_market: str = "A股"
    data_cache_dir: str | None = None


def get_settings_dir() -> Path:
    """Return the user-level settings directory."""
    return Path.home() / ".autostrategy"


def get_default_settings_path() -> Path:
    """Return the default settings file path."""
    return get_settings_dir() / "settings.yaml"


def save_settings(settings: Settings, path: Path | None = None) -> None:
    """Save settings to YAML file."""
    target = path or get_default_settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(settings.model_dump(), f, allow_unicode=True, sort_keys=False)


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from YAML file, returning defaults if missing."""
    target = path or get_default_settings_path()
    if not target.exists():
        return Settings()
    with open(target, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Settings(**data)


def get_llm_api_key_env_vars(config: LLMConfig) -> list[str]:
    """Return API key environment variables in resolution order."""
    env_vars = [
        config.api_key_env,
        "AUTOSTRATEGY_LLM_API_KEY",
        f"{config.provider.upper()}_API_KEY",
        "OPENAI_API_KEY",
    ]
    return list(dict.fromkeys(env_vars))


def resolve_llm_api_key(config: LLMConfig) -> str | None:
    """Resolve the LLM API key value from environment variables."""
    for env_var in get_llm_api_key_env_vars(config):
        value = os.environ.get(env_var)
        if value:
            return value
    return None


def get_llm_api_key_status(config: LLMConfig) -> LLMApiKeyStatus:
    """Return safe LLM API key readiness metadata."""
    checked_env_vars = get_llm_api_key_env_vars(config)
    ready = resolve_llm_api_key(config) is not None
    setup_hint = None
    if not ready:
        setup_hint = (
            f"Set {config.api_key_env} in the local shell before starting autostrategy."
        )
    return LLMApiKeyStatus(
        ready=ready,
        missing_api_key=not ready,
        api_key_env=config.api_key_env,
        checked_env_vars=checked_env_vars,
        setup_hint=setup_hint,
    )


def init_settings() -> Settings:
    """Initialize settings directory and return current settings."""
    settings = load_settings()
    save_settings(settings)
    return settings
