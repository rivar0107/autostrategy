"""Optimization API integration tests."""

import yaml
from fastapi.testclient import TestClient

from autostrategy.api.app import create_app


def _create_tunable_strategy(client, workspace_root):
    response = client.post("/api/v1/strategies", json={"name": "tunable"})
    assert response.status_code == 200
    strategy_dir = workspace_root / "tunable"
    (strategy_dir / "config.yaml").write_text(
        "market: A股\nannual_return: 10.0\n", encoding="utf-8"
    )
    (strategy_dir / "strategy.py").write_text(
        "def run_backtest(config):\n"
        "    return {\n"
        "        'annual_return': float(config['annual_return']),\n"
        "        'max_drawdown': 8.0, 'sharpe': 1.2, 'win_rate': 52.0,\n"
        "        'profit_loss_ratio': 1.6, 'total_trades': 30,\n"
        "    }\n",
        encoding="utf-8",
    )


def test_api_optimization_evaluate_read_and_accept(tmp_path):
    client = TestClient(create_app(workspace_root=tmp_path))
    _create_tunable_strategy(client, tmp_path)

    evaluated = client.post(
        "/api/v1/strategies/tunable/optimizations",
        json={
            "minimum_improvement": 1.0,
            "candidates": [
                {"name": "better", "config_overrides": {"annual_return": 16.0}}
            ],
        },
    )
    assert evaluated.status_code == 200
    report = evaluated.json()
    assert report["recommended_candidate"] == "better"
    assert report["base_version_id"]

    latest = client.get("/api/v1/strategies/tunable/optimizations/latest")
    assert latest.status_code == 200
    assert latest.json()["report_id"] == report["report_id"]

    accepted = client.post(
        f"/api/v1/strategies/tunable/optimizations/{report['report_id']}/accept",
        json={"candidate_name": "better"},
    )
    assert accepted.status_code == 200
    assert accepted.json()["accepted"] is True
    assert accepted.json()["accepted_version_id"]
    config = yaml.safe_load((tmp_path / "tunable" / "config.yaml").read_text(encoding="utf-8"))
    assert config["annual_return"] == 16.0
