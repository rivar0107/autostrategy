"""Tests for paper run job service."""

import time

from autostrategy.services.paper_run_job_service import PaperRunJobService
from autostrategy.services.paper_run_service import PaperRunService
from autostrategy.services.strategy_service import StrategyService


def _write_strategy(strategy_dir, body: str) -> None:
    (strategy_dir / "strategy.py").write_text(body, encoding="utf-8")
    (strategy_dir / "config.yaml").write_text("market: A股\ninitial_cash: 1000000\n", encoding="utf-8")


def _wait_for_terminal(service: PaperRunJobService, slug: str, job_id: str, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = service.get_job(slug, job_id)
        if job.status in {"succeeded", "failed", "timed_out", "stopped"}:
            return job
        time.sleep(0.05)
    raise AssertionError("Job did not reach terminal state")


def test_paper_run_job_service_succeeds(tmp_path):
    StrategyService(workspace_root=tmp_path).create_strategy("demo")
    _write_strategy(
        tmp_path / "demo",
        "def run_paper(config):\n"
        "    return {'paper': {'initial_cash': 1000000, 'final_value': 1010000}, 'events': []}\n",
    )
    service = PaperRunJobService(workspace_root=tmp_path, timeout_seconds=5)

    submitted = service.submit_paper_run("demo")
    finished = _wait_for_terminal(service, "demo", submitted.job_id)

    assert finished.status == "succeeded"
    assert finished.result_path and finished.result_path.exists()
    assert PaperRunService(workspace_root=tmp_path).get_paper_result("demo").result["run_status"] == "completed"


def test_paper_run_job_service_fails_without_run_paper(tmp_path):
    StrategyService(workspace_root=tmp_path).create_strategy("demo")
    _write_strategy(tmp_path / "demo", "def run_backtest(config):\n    return {}\n")
    service = PaperRunJobService(workspace_root=tmp_path, timeout_seconds=5)

    submitted = service.submit_paper_run("demo")
    finished = _wait_for_terminal(service, "demo", submitted.job_id)

    assert finished.status == "failed"
    assert finished.error and "run_paper" in finished.error


def test_paper_run_job_service_times_out(tmp_path):
    StrategyService(workspace_root=tmp_path).create_strategy("demo")
    _write_strategy(
        tmp_path / "demo",
        "import time\n"
        "def run_paper(config):\n"
        "    time.sleep(2)\n"
        "    return {'paper': {'initial_cash': 1000000, 'final_value': 1000000}}\n",
    )
    service = PaperRunJobService(workspace_root=tmp_path, timeout_seconds=1)

    submitted = service.submit_paper_run("demo")
    finished = _wait_for_terminal(service, "demo", submitted.job_id, timeout=4)

    assert finished.status == "timed_out"
    assert finished.error and "timed out" in finished.error


def test_paper_run_job_service_reuses_active_job(tmp_path):
    StrategyService(workspace_root=tmp_path).create_strategy("demo")
    _write_strategy(
        tmp_path / "demo",
        "import time\n"
        "def run_paper(config):\n"
        "    time.sleep(0.5)\n"
        "    return {'paper': {'initial_cash': 1000000, 'final_value': 1005000}, 'events': []}\n",
    )
    service = PaperRunJobService(workspace_root=tmp_path, timeout_seconds=5)

    first = service.submit_paper_run("demo")
    second = service.submit_paper_run("demo")

    assert second.job_id == first.job_id
    assert second.status in {"queued", "running"}
    _wait_for_terminal(service, "demo", first.job_id)
