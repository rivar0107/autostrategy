"""Tests for paper run core workflow."""

import json

from autostrategy.core.backtest_engine import run_paper_replay_workflow


def _write_strategy(strategy_dir, body: str) -> None:
    (strategy_dir / "strategy.py").write_text(body, encoding="utf-8")
    (strategy_dir / "config.yaml").write_text("market: A股\ninitial_cash: 1000000\n", encoding="utf-8")


def test_paper_replay_workflow_writes_result_and_events(tmp_path):
    _write_strategy(
        tmp_path,
        "def run_paper(config):\n"
        "    return {\n"
        "        'paper': {'initial_cash': 1000000, 'final_value': 1010000, 'max_drawdown': 1.2, 'trade_count': 1},\n"
        "        'events': [{'timestamp': '2024-01-02', 'symbol': '000001.SZ', 'action': 'buy', 'price': 10, 'size': 100, 'cash_after': 999000, 'position_after': 100, 'reason': 'signal', 'equity_after': 1000000}],\n"
        "    }\n",
    )

    result = run_paper_replay_workflow(tmp_path)

    result_path = tmp_path / "paper_run" / "results" / "paper_run_result.json"
    events_path = tmp_path / "paper_run" / "results" / "paper_run_events.jsonl"
    assert result["mode"] == "paper_run"
    assert result["run_status"] == "completed"
    assert result["summary"]["paper_return"] == 1.0
    assert result_path.exists()
    assert events_path.exists()
    assert json.loads(events_path.read_text(encoding="utf-8").strip())["action"] == "buy"


def test_paper_replay_workflow_fails_without_run_paper(tmp_path):
    _write_strategy(tmp_path, "def run_backtest(config):\n    return {}\n")

    result = run_paper_replay_workflow(tmp_path)

    assert result["run_status"] == "failed"
    assert "run_paper" in result["error"]
    assert (tmp_path / "paper_run" / "results" / "paper_run_result.json").exists()


def test_paper_replay_workflow_supports_stop_requested(tmp_path):
    _write_strategy(
        tmp_path,
        "def run_paper(config):\n"
        "    return {'paper': {'initial_cash': 1000000, 'final_value': 1005000}, 'events': []}\n",
    )

    result = run_paper_replay_workflow(tmp_path, stop_requested=lambda: True)

    assert result["run_status"] == "stopped"


def test_paper_replay_workflow_refreshes_incremental_result(tmp_path):
    _write_strategy(
        tmp_path,
        "def run_paper(config):\n"
        "    yield {'timestamp': '2024-01-02', 'symbol': '000001.SZ', 'action': 'buy', 'progress': 0.5, 'reason': 'signal'}\n"
        "    yield {'paper': {'initial_cash': 1000000, 'final_value': 1010000}, 'replay': {'progress': 1.0}}\n",
    )

    result = run_paper_replay_workflow(tmp_path)

    result_path = tmp_path / "paper_run" / "results" / "paper_run_result.json"
    events_path = tmp_path / "paper_run" / "results" / "paper_run_events.jsonl"
    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["run_status"] == "completed"
    assert persisted["replay"]["progress"] == 1.0
    assert persisted["replay"]["current_at"] == "2024-01-02"
    assert persisted["latest_decision"]["action"] == "buy"
    assert len(events_path.read_text(encoding="utf-8").splitlines()) == 1


def test_paper_replay_workflow_stops_incremental_replay(tmp_path):
    _write_strategy(
        tmp_path,
        "def run_paper(config):\n"
        "    yield {'timestamp': '2024-01-02', 'action': 'buy', 'progress': 0.5}\n"
        "    yield {'timestamp': '2024-01-03', 'action': 'sell', 'progress': 1.0}\n",
    )
    calls = {"count": 0}

    def stop_after_first_event():
        calls["count"] += 1
        return calls["count"] > 1

    result = run_paper_replay_workflow(tmp_path, stop_requested=stop_after_first_event)

    events_path = tmp_path / "paper_run" / "results" / "paper_run_events.jsonl"
    assert result["run_status"] == "stopped"
    assert result["replay"]["bars_processed"] == 1
    assert len(events_path.read_text(encoding="utf-8").splitlines()) == 1
