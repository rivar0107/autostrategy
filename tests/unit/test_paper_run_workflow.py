"""Tests for paper run core workflow."""

import json

import pytest

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


def test_paper_run_replays_account_from_decision_events(tmp_path):
    """5C.2: workflow derives virtual account from buy/sell events."""
    _write_strategy(
        tmp_path,
        "def run_paper(config):\n"
        "    return {\n"
        "        'events': [\n"
        "            {'timestamp': '2024-01-02', 'symbol': '000001.SZ', 'action': 'buy', 'price': 10, 'size': 1000},\n"
        "            {'timestamp': '2024-01-03', 'symbol': '000001.SZ', 'action': 'sell', 'price': 12, 'size': 1000},\n"
        "        ],\n"
        "    }\n",
    )

    result = run_paper_replay_workflow(tmp_path)

    assert result["run_status"] == "completed"
    summary = result["summary"]
    # 1,000,000 -> buy 1000 @10 -> sell 1000 @12 => 1,002,000 (+0.2%)
    assert summary["final_value"] == 1_002_000
    assert summary["paper_return"] == 0.2
    assert summary["trade_count"] == 2
    assert summary["position_count"] == 0


def test_paper_run_account_snapshot_in_result_file(tmp_path):
    """5C.2: result JSON carries cash/positions/equity account snapshot."""
    _write_strategy(
        tmp_path,
        "def run_paper(config):\n"
        "    return {\n"
        "        'events': [\n"
        "            {'timestamp': '2024-01-02', 'symbol': '000001.SZ', 'action': 'buy', 'price': 10, 'size': 1000},\n"
        "        ],\n"
        "    }\n",
    )

    run_paper_replay_workflow(tmp_path)

    persisted = json.loads((tmp_path / "paper_run" / "results" / "paper_run_result.json").read_text(encoding="utf-8"))
    paper = persisted.get("account") or persisted.get("paper") or {}
    assert paper.get("cash") == 990_000
    assert paper.get("equity") == 1_000_000
    assert paper.get("position_count") == 1
    assert paper["positions"][0]["symbol"] == "000001.SZ"


def test_paper_run_incremental_account_updates(tmp_path):
    """5C.2: incremental replay also derives account state."""
    _write_strategy(
        tmp_path,
        "def run_paper(config):\n"
        "    yield {'timestamp': '2024-01-02', 'symbol': '000001.SZ', 'action': 'buy', 'price': 10, 'size': 1000}\n"
        "    yield {'timestamp': '2024-01-03', 'symbol': '000001.SZ', 'action': 'hold', 'price': 11}\n",
    )

    result = run_paper_replay_workflow(tmp_path)

    assert result["run_status"] == "completed"
    summary = result["summary"]
    # bought 1000 @10, marked @11 => equity 1,001,000
    assert summary["final_value"] == 1_001_000
    assert summary["position_count"] == 1


# --- Phase 5D: local feed driven paper runs ---


def _write_feed_fixture(strategy_dir) -> None:
    data_dir = strategy_dir / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "bars.csv").write_text(
        "date,symbol,open,high,low,close,volume\n"
        "2024-01-02,000300.SH,10,10.5,9.8,10.0,1000\n"
        "2024-01-03,000300.SH,10,11,9.9,11.0,1200\n"
        "2024-01-04,000300.SH,11,11.5,10.5,10.5,900\n",
        encoding="utf-8",
    )


def test_feed_driven_replay_without_run_paper(tmp_path):
    _write_feed_fixture(tmp_path)
    (tmp_path / "strategy.py").write_text("def run_backtest(config):\n    return {}\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        "market: A股\ninitial_cash: 100000\nsymbols: ['000300.SH']\nfeed:\n  path: data/bars.csv\n",
        encoding="utf-8",
    )

    result = run_paper_replay_workflow(tmp_path)

    assert result["run_status"] == "completed"
    assert result["paper"]["cash"] == 100000
    assert result["paper"]["equity"] == 100000
    assert result["replay"]["bars_processed"] == 3
    feed_meta = result["replay"]["feed"]
    assert feed_meta["bar_count"] == 3
    assert feed_meta["symbols"] == ["000300.SH"]
    assert feed_meta["start"] == "2024-01-02T00:00:00"
    assert feed_meta["end"] == "2024-01-04T00:00:00"
    events = (tmp_path / "paper_run" / "results" / "paper_run_events.jsonl").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    assert len(events) == 3
    assert all(json.loads(line)["action"] == "hold" for line in events)


def test_feed_injected_into_run_paper_config(tmp_path):
    _write_feed_fixture(tmp_path)
    (tmp_path / "strategy.py").write_text(
        "def run_paper(config):\n"
        "    bars = config['feed_bars']\n"
        "    return {\n"
        "        'paper': {'initial_cash': 100000, 'final_value': 100000},\n"
        "        'events': [\n"
        "            {'timestamp': b['at'], 'symbol': b['symbol'], 'action': 'hold',\n"
        "             'price': b['close'], 'size': 0, 'reason': 'from feed'}\n"
        "            for b in bars\n"
        "        ],\n"
        "    }\n",
        encoding="utf-8",
    )
    (tmp_path / "config.yaml").write_text(
        "market: A股\ninitial_cash: 100000\nfeed:\n  path: data/bars.csv\n  start: '2024-01-03'\n",
        encoding="utf-8",
    )

    result = run_paper_replay_workflow(tmp_path)

    assert result["run_status"] == "completed"
    # start filter applied: only 2 of the 3 bars reached the strategy
    assert result["replay"]["bars_processed"] == 2
    assert result["replay"]["feed"]["bar_count"] == 2


def test_feed_missing_file_fails_gracefully(tmp_path):
    (tmp_path / "strategy.py").write_text("def run_backtest(config):\n    return {}\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        "market: A股\nfeed:\n  path: data/missing.csv\n", encoding="utf-8"
    )

    with pytest.raises(FileNotFoundError):
        run_paper_replay_workflow(tmp_path)
