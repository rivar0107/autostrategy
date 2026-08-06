"""Tests for LLM client."""

import pytest

from autostrategy.config import LLMConfig, get_llm_api_key_status, resolve_llm_api_key
from autostrategy.llm.client import LLMClient
from autostrategy.services.exceptions import LLMConfigurationRequiredError


def test_client_initialization():
    config = LLMConfig(provider="openai", model="gpt-4o-mini")
    client = LLMClient(config)
    assert client.config == config


def test_resolve_api_key_from_env(monkeypatch):
    monkeypatch.setenv("AUTOSTRATEGY_LLM_API_KEY", "test-key")
    config = LLMConfig(api_key_env="AUTOSTRATEGY_LLM_API_KEY")
    client = LLMClient(config)
    assert client.api_key == "test-key"


def test_resolve_api_key_missing():
    config = LLMConfig(api_key_env="NON_EXISTENT_KEY")
    client = LLMClient(config)
    assert client.api_key is None


def test_api_key_resolution_precedence(monkeypatch):
    monkeypatch.setenv("CUSTOM_KEY", "custom")
    monkeypatch.setenv("AUTOSTRATEGY_LLM_API_KEY", "default")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "provider")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")

    config = LLMConfig(provider="deepseek", api_key_env="CUSTOM_KEY")

    assert resolve_llm_api_key(config) == "custom"


def test_api_key_resolution_falls_back_to_default_env(monkeypatch):
    monkeypatch.delenv("CUSTOM_KEY", raising=False)
    monkeypatch.setenv("AUTOSTRATEGY_LLM_API_KEY", "default")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "provider")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")

    config = LLMConfig(provider="deepseek", api_key_env="CUSTOM_KEY")

    assert resolve_llm_api_key(config) == "default"


def test_api_key_resolution_falls_back_to_provider_env(monkeypatch):
    monkeypatch.delenv("CUSTOM_KEY", raising=False)
    monkeypatch.delenv("AUTOSTRATEGY_LLM_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "provider")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")

    config = LLMConfig(provider="deepseek", api_key_env="CUSTOM_KEY")

    assert resolve_llm_api_key(config) == "provider"


def test_api_key_resolution_falls_back_to_openai_env(monkeypatch):
    monkeypatch.delenv("CUSTOM_KEY", raising=False)
    monkeypatch.delenv("AUTOSTRATEGY_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai")

    config = LLMConfig(provider="deepseek", api_key_env="CUSTOM_KEY")

    assert resolve_llm_api_key(config) == "openai"


def test_stale_api_key_env_reports_missing(monkeypatch):
    monkeypatch.delenv("STALE_KEY", raising=False)
    monkeypatch.delenv("AUTOSTRATEGY_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    status = get_llm_api_key_status(LLMConfig(provider="deepseek", api_key_env="STALE_KEY"))

    assert status.ready is False
    assert status.missing_api_key is True
    assert status.api_key_env == "STALE_KEY"
    assert "STALE_KEY" in status.checked_env_vars


@pytest.mark.parametrize("field", ["provider", "model", "api_key_env"])
def test_llm_config_rejects_blank_required_fields(field):
    payload = {"provider": "openai", "model": "gpt-4o-mini", "api_key_env": "API_KEY"}
    payload[field] = "   "

    with pytest.raises(ValueError):
        LLMConfig(**payload)


def test_chat_without_api_key_raises_structured_error(monkeypatch):
    monkeypatch.delenv("AUTOSTRATEGY_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = LLMConfig(provider="openai", api_key_env="NON_EXISTENT_KEY")
    client = LLMClient(config)

    with pytest.raises(LLMConfigurationRequiredError) as exc_info:
        client.chat([])

    assert exc_info.value.code == "llm_configuration_required"
    assert exc_info.value.details["api_key_env"] == "NON_EXISTENT_KEY"
    assert exc_info.value.details["llm_ready"] is False
