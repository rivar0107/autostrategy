"""Paper run API integration tests."""

import time

from fastapi.testclient import TestClient

from autostrategy.api.app import create_app


def _write_paper_strategy(strategy_dir):
    (strategy_dir / "strategy.py").write_text(
        "def run_paper(config):\n"
        "    return {\n"
        "        'paper': {'initial_cash': 1000000, 'final_value': 1010000, 'max_drawdown': 1.2, 'trade_count': 1},\n"
        "        'events': [{'timestamp': '2024-01-02', 'symbol': '000001.SZ', 'action': 'buy', 'price': 10, 'size': 100, 'cash_after': 999000, 'position_after': 100, 'reason': 'signal', 'equity_after': 1000000}],\n"
        "    }\n",
        encoding="utf-8",
    )
    (strategy_dir / "config.yaml").write_text("market: A股\ninitial_cash: 1000000\n", encoding="utf-8")


def _wait_for_job(client: TestClient, slug: str, job_id: str, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/strategies/{slug}/paper-run-jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"succeeded", "failed", "timed_out", "stopped"}:
            return payload
        time.sleep(0.05)
    raise AssertionError("Paper run job did not reach terminal state")


def test_api_paper_run_start_poll_and_read_result(tmp_path):
    client = TestClient(create_app(workspace_root=tmp_path))
    create = client.post("/api/v1/strategies", json={"name": "demo"})
    assert create.status_code == 200
    _write_paper_strategy(tmp_path / "demo")

    run = client.post("/api/v1/strategies/demo/paper-run")
    assert run.status_code == 202
    job = _wait_for_job(client, "demo", run.json()["job_id"])
    assert job["status"] == "succeeded"

    result = client.get("/api/v1/strategies/demo/paper-run-result")
    assert result.status_code == 200
    assert result.json()["result"]["mode"] == "paper_run"
    assert result.json()["result"]["summary"]["paper_return"] == 1.0


def test_api_paper_run_missing_run_paper_returns_failed_job(tmp_path):
    client = TestClient(create_app(workspace_root=tmp_path))
    create = client.post("/api/v1/strategies", json={"name": "demo"})
    assert create.status_code == 200
    (tmp_path / "demo" / "strategy.py").write_text("def run_backtest(config):\n    return {}\n", encoding="utf-8")

    response = client.post("/api/v1/strategies/demo/paper-run")

    assert response.status_code == 202
    job = _wait_for_job(client, "demo", response.json()["job_id"])
    assert job["status"] == "failed"
    assert "run_paper" in job["error"]


def test_api_paper_run_stop_request_reaches_stopped_job(tmp_path):
    client = TestClient(create_app(workspace_root=tmp_path))
    create = client.post("/api/v1/strategies", json={"name": "demo"})
    assert create.status_code == 200
    (tmp_path / "demo" / "strategy.py").write_text(
        "import time\n"
        "def run_paper(config):\n"
        "    for index in range(5):\n"
        "        time.sleep(0.05)\n"
        "        yield {'timestamp': f'2024-01-0{index + 1}', 'action': 'hold', 'progress': (index + 1) / 5}\n",
        encoding="utf-8",
    )
    (tmp_path / "demo" / "config.yaml").write_text("market: A股\ninitial_cash: 1000000\n", encoding="utf-8")

    run = client.post("/api/v1/strategies/demo/paper-run")
    assert run.status_code == 202
    job_id = run.json()["job_id"]

    stopped = client.post(f"/api/v1/strategies/demo/paper-run-jobs/{job_id}/stop")
    assert stopped.status_code == 200
    assert stopped.json()["stop_requested"] is True

    job = _wait_for_job(client, "demo", job_id)
    assert job["status"] == "stopped"
    result = client.get("/api/v1/strategies/demo/paper-run-result")
    assert result.status_code == 200
    assert result.json()["result"]["run_status"] == "stopped"


def test_api_paper_run_exposes_running_partial_result(tmp_path):
    client = TestClient(create_app(workspace_root=tmp_path))
    create = client.post("/api/v1/strategies", json={"name": "demo"})
    assert create.status_code == 200
    (tmp_path / "demo" / "strategy.py").write_text(
        "import time\n"
        "def run_paper(config):\n"
        "    yield {'timestamp': '2024-01-02', 'action': 'buy', 'progress': 0.5}\n"
        "    time.sleep(0.2)\n"
        "    yield {'paper': {'initial_cash': 1000000, 'final_value': 1010000}, 'replay': {'progress': 1.0}}\n",
        encoding="utf-8",
    )
    (tmp_path / "demo" / "config.yaml").write_text("market: A股\ninitial_cash: 1000000\n", encoding="utf-8")

    run = client.post("/api/v1/strategies/demo/paper-run")
    assert run.status_code == 202
    job_id = run.json()["job_id"]

    deadline = time.monotonic() + 2
    partial = None
    while time.monotonic() < deadline:
        result = client.get("/api/v1/strategies/demo/paper-run-result")
        if result.status_code == 200 and result.json()["result"]["run_status"] == "running":
            partial = result.json()["result"]
            break
        time.sleep(0.02)

    assert partial is not None
    assert partial["replay"]["progress"] == 0.5
    assert partial["latest_decision"]["action"] == "buy"
    job = _wait_for_job(client, "demo", job_id)
    assert job["status"] == "succeeded"
