"""Configuration management for autostrategy.

Settings are stored in ~/.autostrategy/settings.yaml.
API keys should be stored via keyring or environment variables, not in this file.
"""

from __future__ import annotations

import json
from pathlib import Path
import os
import tomllib
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


def get_codex_config_path() -> Path:
    """Return the Codex CLI config path."""
    return Path.home() / ".codex" / "config.toml"


def get_codex_auth_path() -> Path:
    """Return the Codex CLI auth (API key) path."""
    return Path.home() / ".codex" / "auth.json"


def load_codex_llm_defaults() -> dict | None:
    """Read LLM defaults from the local Codex CLI configuration.

    Returns ``{"model", "base_url", "api_key"}`` when the Codex desktop
    app / CLI is configured, otherwise None. Used as the default LLM
    configuration so the workbench reuses the user's Codex setup.
    """
    config_path = get_codex_config_path()
    auth_path = get_codex_auth_path()
    if not config_path.exists() or not auth_path.exists():
        return None
    try:
        with open(config_path, "rb") as f:
            codex_config = tomllib.load(f)
        with open(auth_path, "r", encoding="utf-8") as f:
            auth = json.load(f)
    except (OSError, ValueError, tomllib.TOMLDecodeError):
        return None
    api_key = auth.get("OPENAI_API_KEY")
    model = codex_config.get("model")
    if not api_key or not model:
        return None
    base_url = None
    provider_name = codex_config.get("model_provider")
    providers = codex_config.get("model_providers") or {}
    if provider_name and isinstance(providers.get(provider_name), dict):
        base_url = providers[provider_name].get("base_url")
    return {"model": model, "base_url": base_url, "api_key": api_key}


_CODEX_LLM_DEFAULTS: dict | None = None


def codex_llm_defaults() -> dict | None:
    """Cached accessor for Codex LLM defaults."""
    global _CODEX_LLM_DEFAULTS
    if _CODEX_LLM_DEFAULTS is None:
        _CODEX_LLM_DEFAULTS = load_codex_llm_defaults() or {}
    return _CODEX_LLM_DEFAULTS or None


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from YAML file, falling back to Codex LLM defaults.

    When the user has not explicitly configured the LLM (settings file
    missing or LLM section untouched), the local Codex CLI configuration
    is used as the default provider/model/base_url.
    """
    target = path or get_default_settings_path()
    if target.exists():
        with open(target, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        settings = Settings(**data)
    else:
        settings = Settings()
    codex = codex_llm_defaults()
    if codex and settings.llm.provider == DEFAULT_PROVIDER and not settings.llm.base_url:
        if settings.llm.model == "gpt-4o-mini":
            settings.llm.model = codex["model"]
        settings.llm.base_url = codex["base_url"]
    return settings


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
    """Resolve the LLM API key from env vars, then the Codex CLI auth."""
    for env_var in get_llm_api_key_env_vars(config):
        value = os.environ.get(env_var)
        if value:
            return value
    codex = codex_llm_defaults()
    if codex:
        return codex["api_key"]
    return None


def get_llm_api_key_status(config: LLMConfig) -> LLMApiKeyStatus:
    """Return safe LLM API key readiness metadata."""
    checked_env_vars = get_llm_api_key_env_vars(config)
    ready = resolve_llm_api_key(config) is not None
    setup_hint = None
    if not ready:
        setup_hint = (
            f"Set {config.api_key_env} in the local shell before starting autostrategy, "
            "or configure the Codex CLI (autostrategy reuses ~/.codex by default)."
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
