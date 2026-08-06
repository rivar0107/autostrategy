"""LLM-backed API error integration tests."""

from fastapi.testclient import TestClient

from autostrategy.api.app import create_app


def test_api_create_design_returns_llm_configuration_error(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("AUTOSTRATEGY_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(create_app(workspace_root=tmp_path / "workspace"))

    response = client.post(
        "/api/v1/designs",
        json={"name": "demo", "prompt": "帮我做一个策略", "market": "A股"},
    )

    assert response.status_code == 428
    error = response.json()["error"]
    assert error["code"] == "llm_configuration_required"
    assert error["details"]["llm_ready"] is False
    assert error["details"]["api_key_env"] == "AUTOSTRATEGY_LLM_API_KEY"
    assert "api_key" not in error["details"]


def test_api_codegen_returns_llm_configuration_error(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("AUTOSTRATEGY_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(create_app(workspace_root=tmp_path / "workspace"))
    created = client.post("/api/v1/strategies", json={"name": "demo"})
    assert created.status_code == 200
    strategy_dir = tmp_path / "workspace" / "demo"
    (strategy_dir / "STRATEGY_DESIGN.md").write_text("# 策略设计\n\n有效策略内容", encoding="utf-8")

    response = client.post("/api/v1/strategies/demo/codegen", json={"force": False})

    assert response.status_code == 428
    error = response.json()["error"]
    assert error["code"] == "llm_configuration_required"
    assert error["details"]["llm_ready"] is False
    assert error["details"]["api_key_env"] == "AUTOSTRATEGY_LLM_API_KEY"
    assert "api_key" not in error["details"]
