"""Configuration API integration tests."""

from fastapi.testclient import TestClient

from autostrategy.api.app import create_app


def test_api_config_reports_missing_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("AUTOSTRATEGY_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(create_app(workspace_root=tmp_path / "workspace"))

    response = client.get("/api/v1/config")

    assert response.status_code == 200
    data = response.json()
    assert data["llm_provider"] == "openai"
    assert data["llm_ready"] is False
    assert data["llm_missing_api_key"] is True
    assert data["llm_api_key_env"] == "AUTOSTRATEGY_LLM_API_KEY"
    assert "AUTOSTRATEGY_LLM_API_KEY" in data["llm_checked_env_vars"]
    assert "api_key" not in data


def test_api_config_reports_ready_when_env_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AUTOSTRATEGY_LLM_API_KEY", "test-key")
    client = TestClient(create_app(workspace_root=tmp_path / "workspace"))

    response = client.get("/api/v1/config")

    assert response.status_code == 200
    data = response.json()
    assert data["llm_ready"] is True
    assert data["llm_missing_api_key"] is False
    assert data["llm_setup_hint"] is None


def test_api_update_llm_config_persists_only_safe_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("AUTOSTRATEGY_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(create_app(workspace_root=tmp_path / "workspace"))

    response = client.put(
        "/api/v1/config/llm",
        json={
            "provider": "deepseek",
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["llm_provider"] == "deepseek"
    assert data["llm_model"] == "deepseek-chat"
    assert data["llm_base_url"] == "https://api.deepseek.com"
    assert data["llm_api_key_env"] == "DEEPSEEK_API_KEY"
    assert data["llm_ready"] is False

    settings_text = (tmp_path / ".autostrategy" / "settings.yaml").read_text(encoding="utf-8")
    assert "deepseek-chat" in settings_text
    assert "api_key:" not in settings_text
    assert "test-secret" not in settings_text


def test_api_update_llm_config_rejects_api_key_field(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    client = TestClient(create_app(workspace_root=tmp_path / "workspace"))

    response = client.put(
        "/api/v1/config/llm",
        json={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "base_url": None,
            "api_key_env": "AUTOSTRATEGY_LLM_API_KEY",
            "api_key": "test-secret",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_api_update_llm_config_rejects_untrusted_base_url(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    client = TestClient(create_app(workspace_root=tmp_path / "workspace"))

    response = client.put(
        "/api/v1/config/llm",
        json={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "base_url": "https://attacker.example/v1",
            "api_key_env": "OPENAI_API_KEY",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_api_update_llm_config_rejects_untrusted_api_key_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    client = TestClient(create_app(workspace_root=tmp_path / "workspace"))

    response = client.put(
        "/api/v1/config/llm",
        json={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "AWS_SECRET_ACCESS_KEY",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
