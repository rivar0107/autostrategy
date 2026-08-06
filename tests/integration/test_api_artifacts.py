"""Strategy artifact API integration tests."""

import json

from fastapi.testclient import TestClient

from autostrategy.api.app import create_app


def test_api_artifacts_list_and_preview(tmp_path):
    client = TestClient(create_app(workspace_root=tmp_path))
    create = client.post("/api/v1/strategies", json={"name": "demo", "template": "dual-ma"})
    assert create.status_code == 200

    artifacts = client.get("/api/v1/strategies/demo/artifacts")
    assert artifacts.status_code == 200
    payload = artifacts.json()
    keys = {item["artifact_key"] for item in payload["artifacts"]}
    assert {"design", "config", "backtest_result"}.issubset(keys)
    assert any(item["artifact_key"] == "design" and item["exists"] for item in payload["artifacts"])

    design = client.get("/api/v1/strategies/demo/artifacts/design")
    assert design.status_code == 200
    assert "策略" in design.json()["content"]
    assert design.json()["content_type"] == "markdown"


def test_api_artifacts_backtest_result_json_preview(tmp_path):
    client = TestClient(create_app(workspace_root=tmp_path))
    create = client.post("/api/v1/strategies", json={"name": "demo"})
    assert create.status_code == 200
    result_path = tmp_path / "demo" / "backtest" / "results" / "backtest_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps({"score": 80, "backtest": {"total_trades": 3}}), encoding="utf-8")

    response = client.get("/api/v1/strategies/demo/artifacts/backtest_result")

    assert response.status_code == 200
    assert response.json()["parsed_json"]["score"] == 80


def test_api_artifacts_invalid_key_returns_structured_error(tmp_path):
    client = TestClient(create_app(workspace_root=tmp_path))
    create = client.post("/api/v1/strategies", json={"name": "demo"})
    assert create.status_code == 200

    response = client.get("/api/v1/strategies/demo/artifacts/unknown")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "artifact_not_found"


def test_api_artifacts_missing_file_returns_structured_error(tmp_path):
    client = TestClient(create_app(workspace_root=tmp_path))
    create = client.post("/api/v1/strategies", json={"name": "demo"})
    assert create.status_code == 200

    response = client.get("/api/v1/strategies/demo/artifacts/backtest_result")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "artifact_not_found"
