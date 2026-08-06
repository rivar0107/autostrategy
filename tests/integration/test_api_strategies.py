"""Strategy API integration tests."""

from fastapi.testclient import TestClient

from autostrategy.api.app import create_app


def test_api_strategy_crud(tmp_path):
    client = TestClient(create_app(workspace_root=tmp_path))

    empty = client.get("/api/v1/strategies")
    assert empty.status_code == 200
    assert empty.json() == []

    created = client.post(
        "/api/v1/strategies",
        json={"name": "demo", "market": "A股", "template": "dual-ma"},
    )
    assert created.status_code == 200
    assert created.json()["slug"] == "demo"

    listed = client.get("/api/v1/strategies")
    assert listed.status_code == 200
    assert len(listed.json()) == 1

    detail = client.get("/api/v1/strategies/demo")
    assert detail.status_code == 200
    assert detail.json()["strategy"]["slug"] == "demo"
    assert detail.json()["paths"]["design"].endswith("STRATEGY_DESIGN.md")

    deleted = client.delete("/api/v1/strategies/demo")
    assert deleted.status_code == 204

    missing = client.get("/api/v1/strategies/demo")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "strategy_not_found"


def test_api_templates(tmp_path):
    client = TestClient(create_app(workspace_root=tmp_path))

    response = client.get("/api/v1/templates")

    assert response.status_code == 200
    assert "dual-ma" in response.json()
