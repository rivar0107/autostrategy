"""Tests for backtest job service."""

import time

from autostrategy.services.backtest_job_service import BacktestJobService
from autostrategy.services.backtest_service import BacktestService
from autostrategy.services.strategy_service import StrategyService


def _write_strategy(strategy_dir, body: str) -> None:
    (strategy_dir / "strategy.py").write_text(body, encoding="utf-8")
    (strategy_dir / "config.yaml").write_text(
        "market: A股\ninitial_cash: 1000000\nstart_date: '2024-01-01'\nend_date: '2024-12-31'\n",
        encoding="utf-8",
    )


def _wait_for_terminal(service: BacktestJobService, slug: str, job_id: str, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = service.get_job(slug, job_id)
        if job.status in {"succeeded", "failed", "timed_out"}:
            return job
        time.sleep(0.05)
    raise AssertionError("Job did not reach terminal state")


def test_backtest_job_service_succeeds(tmp_path):
    StrategyService(workspace_root=tmp_path).create_strategy("demo")
    _write_strategy(
        tmp_path / "demo",
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
    )
    service = BacktestJobService(workspace_root=tmp_path, timeout_seconds=5)

    submitted = service.submit_backtest("demo")
    finished = _wait_for_terminal(service, "demo", submitted.job_id)

    assert finished.status == "succeeded"
    assert finished.score and finished.score > 0
    assert finished.result_path and finished.result_path.exists()
    assert BacktestService(workspace_root=tmp_path).get_backtest_result("demo").score > 0


def test_backtest_job_service_fails_without_strategy_file(tmp_path):
    StrategyService(workspace_root=tmp_path).create_strategy("demo")
    service = BacktestJobService(workspace_root=tmp_path, timeout_seconds=5)

    submitted = service.submit_backtest("demo")
    finished = _wait_for_terminal(service, "demo", submitted.job_id)

    assert finished.status == "failed"
    assert finished.error and "strategy.py" in finished.error


def test_backtest_job_service_times_out(tmp_path):
    StrategyService(workspace_root=tmp_path).create_strategy("demo")
    _write_strategy(
        tmp_path / "demo",
        "import time\n"
        "def run_backtest(config):\n"
        "    time.sleep(2)\n"
        "    return {'annual_return': 1.0}\n",
    )
    service = BacktestJobService(workspace_root=tmp_path, timeout_seconds=1)

    submitted = service.submit_backtest("demo")
    finished = _wait_for_terminal(service, "demo", submitted.job_id, timeout=4)

    assert finished.status == "timed_out"
    assert finished.error and "timed out" in finished.error


def test_backtest_job_service_reuses_active_job(tmp_path):
    StrategyService(workspace_root=tmp_path).create_strategy("demo")
    _write_strategy(
        tmp_path / "demo",
        "import time\n"
        "def run_backtest(config):\n"
        "    time.sleep(0.5)\n"
        "    return {\n"
        "        'annual_return': 12.0,\n"
        "        'max_drawdown': 8.0,\n"
        "        'sharpe': 1.5,\n"
        "        'win_rate': 55.0,\n"
        "        'profit_loss_ratio': 1.8,\n"
        "    }\n",
    )
    service = BacktestJobService(workspace_root=tmp_path, timeout_seconds=5)

    first = service.submit_backtest("demo")
    second = service.submit_backtest("demo")

    assert second.job_id == first.job_id
    assert second.status in {"queued", "running"}
    _wait_for_terminal(service, "demo", first.job_id)
