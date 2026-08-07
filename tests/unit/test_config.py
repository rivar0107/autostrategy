"""Tests for autostrategy.config."""

from pathlib import Path

import pytest

from autostrategy.config import Settings, get_settings_dir, load_settings, save_settings


def test_get_settings_dir():
    """Settings dir should be under user home."""
    settings_dir = get_settings_dir()
    assert isinstance(settings_dir, Path)
    assert settings_dir.name == ".autostrategy"
    assert settings_dir.parent == Path.home()


def test_settings_defaults():
    """Default settings should be valid."""
    settings = Settings()
    assert settings.version == "0.1.0"
    assert settings.llm.provider == "openai"
    assert settings.llm.model == "gpt-4o-mini"


def test_save_and_load_settings(tmp_path):
    """Round-trip settings save/load."""
    settings_path = tmp_path / "settings.yaml"
    settings = Settings()
    save_settings(settings, settings_path)
    loaded = load_settings(settings_path)
    assert loaded.version == settings.version
    assert loaded.llm.provider == settings.llm.provider


def test_load_missing_settings_returns_defaults(tmp_path):
    """Loading missing settings should return defaults."""
    settings_path = tmp_path / "not_exists.yaml"
    settings = load_settings(settings_path)
    assert settings.version == "0.1.0"


def test_codex_llm_defaults_applied(monkeypatch, tmp_path):
    """Codex CLI config should be the default LLM source when unconfigured."""
    import json

    import autostrategy.config as config_module

    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "config.toml").write_text(
        'model = "gpt-5.6-sol"\nmodel_provider = "newapi"\n'
        '[model_providers.newapi]\nbase_url = "http://127.0.0.1:15721/v1"\n',
        encoding="utf-8",
    )
    (codex_dir / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": "sk-codex-test"}), encoding="utf-8")

    monkeypatch.setattr(config_module, "_CODEX_LLM_DEFAULTS", None)
    monkeypatch.setattr(config_module, "get_codex_config_path", lambda: codex_dir / "config.toml")
    monkeypatch.setattr(config_module, "get_codex_auth_path", lambda: codex_dir / "auth.json")
    for var in ("AUTOSTRATEGY_LLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    settings = load_settings(tmp_path / "missing.yaml")
    assert settings.llm.model == "gpt-5.6-sol"
    assert settings.llm.base_url == "http://127.0.0.1:15721/v1"
    assert config_module.resolve_llm_api_key(settings.llm) == "sk-codex-test"


def test_explicit_llm_config_overrides_codex(monkeypatch, tmp_path):
    """User-configured LLM settings win over Codex defaults."""
    import json

    import autostrategy.config as config_module

    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    (codex_dir / "config.toml").write_text('model = "gpt-5.6-sol"\n', encoding="utf-8")
    (codex_dir / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": "sk-codex"}), encoding="utf-8")

    monkeypatch.setattr(config_module, "_CODEX_LLM_DEFAULTS", None)
    monkeypatch.setattr(config_module, "get_codex_config_path", lambda: codex_dir / "config.toml")
    monkeypatch.setattr(config_module, "get_codex_auth_path", lambda: codex_dir / "auth.json")

    settings_path = tmp_path / "settings.yaml"
    save_settings(Settings(llm=config_module.LLMConfig(model="deepseek-chat", base_url="https://api.deepseek.com/v1")), settings_path)
    loaded = load_settings(settings_path)
    assert loaded.llm.model == "deepseek-chat"
    assert loaded.llm.base_url == "https://api.deepseek.com/v1"


def test_no_codex_config_keeps_defaults(monkeypatch, tmp_path):
    """Without Codex CLI config, defaults stay as-is."""
    import autostrategy.config as config_module

    monkeypatch.setattr(config_module, "_CODEX_LLM_DEFAULTS", None)
    monkeypatch.setattr(config_module, "get_codex_config_path", lambda: tmp_path / "nope.toml")
    monkeypatch.setattr(config_module, "get_codex_auth_path", lambda: tmp_path / "nope.json")
    for var in ("AUTOSTRATEGY_LLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    settings = load_settings(tmp_path / "missing.yaml")
    assert settings.llm.model == "gpt-4o-mini"
    assert settings.llm.base_url is None
    assert config_module.resolve_llm_api_key(settings.llm) is None
