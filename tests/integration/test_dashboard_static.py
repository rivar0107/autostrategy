"""Dashboard static asset integration tests."""

from fastapi.testclient import TestClient

from autostrategy.api.app import create_app


def test_dashboard_shell_is_served(tmp_path):
    client = TestClient(create_app(workspace_root=tmp_path))

    response = client.get("/")

    assert response.status_code == 200
    assert '<div id="root"></div>' in response.text
    assert "/static/assets/" in response.text


def test_dashboard_static_asset_is_served(tmp_path):
    client = TestClient(create_app(workspace_root=tmp_path))
    shell = client.get("/")
    assert shell.status_code == 200
    asset_path = shell.text.split('src="')[1].split('"')[0]

    asset = client.get(asset_path)

    assert asset.status_code == 200
    assert "javascript" in asset.headers["content-type"]
