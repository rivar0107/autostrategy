"""End-to-end API tests for the reproducible research lifecycle."""

import yaml
from fastapi.testclient import TestClient

from autostrategy.api.app import create_app


def _create_api_strategy(client: TestClient, root):
    created = client.post("/api/v1/strategies", json={"name": "research-api"})
    assert created.status_code == 200
    strategy_dir = root / "research-api"
    (strategy_dir / "config.yaml").write_text(
        "market: A股\nsymbols:\n  - 510300.SH\nalpha: 0.0\n"
        "commission: 0.0003\nslippage: 0.001\n",
        encoding="utf-8",
    )
    (strategy_dir / "data").mkdir(exist_ok=True)
    (strategy_dir / "data" / "fetch_data.py").write_text(
        "import pandas as pd\n\n"
        "def fetch(config):\n"
        "    dates = pd.date_range('2020-01-01', '2025-12-31', freq='D')\n"
        "    frame = pd.DataFrame({\n"
        "        'open': 1.0, 'high': 1.0, 'low': 1.0,\n"
        "        'close': 1.0, 'volume': 1000.0,\n"
        "    }, index=dates)\n"
        "    frame.index.name = 'date'\n"
        "    return frame\n",
        encoding="utf-8",
    )
    (strategy_dir / "strategy.py").write_text(
        "def run_backtest(config):\n"
        "    alpha = float(config.get('alpha', 0))\n"
        "    return {\n"
        "        'annual_return': 10.0 + alpha, 'max_drawdown': 8.0,\n"
        "        'sharpe': 1.2 + alpha / 20, 'win_rate': 52.0,\n"
        "        'profit_loss_ratio': 1.6, 'total_trades': 30,\n"
        "    }\n",
        encoding="utf-8",
    )


def test_api_complete_research_accept_and_rollback_flow(tmp_path):
    client = TestClient(create_app(workspace_root=tmp_path))
    _create_api_strategy(client, tmp_path)

    versions = client.get("/api/v1/strategies/research-api/versions")
    assert versions.status_code == 200
    assert len(versions.json()) == 1
    base_version_id = versions.json()[0]["version_id"]

    captured = client.post(
        "/api/v1/strategies/research-api/dataset-manifests",
        json={
            "version_id": base_version_id,
            "data_source": "fixture",
            "benchmark": "000300.SH",
            "train": {"start": "2020-01-01", "end": "2022-12-31"},
            "validation": {"start": "2023-01-01", "end": "2024-12-31"},
            "test": {"start": "2025-01-01", "end": "2025-12-31"},
        },
    )
    assert captured.status_code == 200
    manifest_id = captured.json()["manifest_id"]

    created = client.post(
        "/api/v1/strategies/research-api/experiments",
        json={"base_version_id": base_version_id, "manifest_id": manifest_id},
    )
    assert created.status_code == 200
    session_id = created.json()["session_id"]

    baseline = client.post(
        f"/api/v1/strategies/research-api/experiments/{session_id}/baseline"
    )
    assert baseline.status_code == 200
    assert baseline.json()["status"] == "baseline_completed"

    repeated = client.post(
        f"/api/v1/strategies/research-api/experiments/{session_id}/baseline"
    )
    assert repeated.status_code == 400

    diagnosed = client.post(
        f"/api/v1/strategies/research-api/experiments/{session_id}/diagnose"
    )
    assert diagnosed.status_code == 200
    assert diagnosed.json()["diagnostics"]

    optimized = client.post(
        f"/api/v1/strategies/research-api/experiments/{session_id}/optimize",
        json={
            "minimum_improvement": 1.0,
            "candidates": [
                {"name": "better", "config_overrides": {"alpha": 4.0}}
            ],
        },
    )
    assert optimized.status_code == 200
    assert optimized.json()["status"] == "optimized"
    selected_version_id = optimized.json()["selected_version_id"]

    oos = client.post(
        f"/api/v1/strategies/research-api/experiments/{session_id}/validate-oos",
        json={},
    )
    assert oos.status_code == 200
    assert oos.json()["status"] == "awaiting_decision"
    assert oos.json()["oos_passed"] is True

    repeated_oos = client.post(
        f"/api/v1/strategies/research-api/experiments/{session_id}/validate-oos",
        json={},
    )
    assert repeated_oos.status_code == 400

    stale_marker = tmp_path / "research-api" / "README.md"
    stale_marker.write_text("# external change after OOS\n", encoding="utf-8")
    stale_accept = client.post(
        f"/api/v1/strategies/research-api/experiments/{session_id}/accept",
        json={"reason": "不应接受过期基础版本"},
    )
    assert stale_accept.status_code == 400
    assert "changed after the experiment started" in stale_accept.json()["error"]["message"]
    stale_marker.unlink()

    accepted = client.post(
        f"/api/v1/strategies/research-api/experiments/{session_id}/accept",
        json={"reason": "API 样本外验证通过"},
    )
    assert accepted.status_code == 200
    assert accepted.json()["accepted_version_id"] == selected_version_id
    accepted_config = yaml.safe_load(
        (tmp_path / "research-api" / "config.yaml").read_text(encoding="utf-8")
    )
    assert accepted_config["alpha"] == 4.0

    rolled_back = client.post(
        f"/api/v1/strategies/research-api/versions/{base_version_id}/rollback",
        json={"reason": "API 回滚验证"},
    )
    assert rolled_back.status_code == 200
    assert rolled_back.json()["version_id"] == base_version_id
    restored_config = yaml.safe_load(
        (tmp_path / "research-api" / "config.yaml").read_text(encoding="utf-8")
    )
    assert restored_config["alpha"] == 0.0

    detail = client.get(
        f"/api/v1/strategies/research-api/experiments/{session_id}"
    )
    assert detail.status_code == 200
    assert detail.json()["status"] == "accepted"
