"""Backtest API integration tests."""

import time

from fastapi.testclient import TestClient

from autostrategy.api.app import create_app


def _write_minimal_coded_strategy(strategy_dir):
    (strategy_dir / "strategy.py").write_text(
        "def run_backtest(config):\n"
        "    return {\n"
        "        'annual_return': 12.0,\n"
        "        'max_drawdown': 8.0,\n"
        "        'sharpe': 1.5,\n"
        "        'win_rate': 55.0,\n"
        "        'profit_loss_ratio': 1.8,\n"
        "        'total_trades': 10,\n"
        "        'initial_cash': 1000000,\n"
        "    }\n",
        encoding="utf-8",
    )
    (strategy_dir / "config.yaml").write_text(
        "market: A股\ninitial_cash: 1000000\nstart_date: '2024-01-01'\nend_date: '2024-12-31'\n",
        encoding="utf-8",
    )


def _wait_for_job(client: TestClient, slug: str, job_id: str, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/strategies/{slug}/backtest-jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"succeeded", "failed", "timed_out"}:
            return payload
        time.sleep(0.05)
    raise AssertionError("Backtest job did not reach terminal state")


def test_api_backtest_run_and_read(tmp_path):
    client = TestClient(create_app(workspace_root=tmp_path))
    create = client.post("/api/v1/strategies", json={"name": "demo"})
    assert create.status_code == 200
    _write_minimal_coded_strategy(tmp_path / "demo")

    run = client.post("/api/v1/strategies/demo/backtest")
    assert run.status_code == 202
    job = _wait_for_job(client, "demo", run.json()["job_id"])
    assert job["status"] == "succeeded"
    assert job["score"] > 0

    result = client.get("/api/v1/strategies/demo/backtest-result")
    assert result.status_code == 200
    assert result.json()["result"]["backtest"]["total_trades"] == 10


def test_api_backtest_missing_strategy_file_returns_failed_job(tmp_path):
    client = TestClient(create_app(workspace_root=tmp_path))
    create = client.post("/api/v1/strategies", json={"name": "demo"})
    assert create.status_code == 200

    response = client.post("/api/v1/strategies/demo/backtest")

    assert response.status_code == 202
    job = _wait_for_job(client, "demo", response.json()["job_id"])
    assert job["status"] == "failed"
    assert "strategy.py" in job["error"]
