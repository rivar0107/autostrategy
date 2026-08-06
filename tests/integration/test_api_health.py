"""API health integration tests."""

from fastapi.testclient import TestClient

from autostrategy.api.app import create_app


def test_api_health(tmp_path):
    client = TestClient(create_app(workspace_root=tmp_path))

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_info(tmp_path):
    client = TestClient(create_app(workspace_root=tmp_path))

    response = client.get("/api/v1/info")

    assert response.status_code == 200
    assert response.json()["workspace_root"] == str(tmp_path)
    assert "dual-ma" in response.json()["templates"]
