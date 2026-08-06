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
