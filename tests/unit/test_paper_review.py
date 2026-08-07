"""Phase 5E review tests: metrics, key events, markdown + workflow integration."""

from __future__ import annotations

import json

from autostrategy.core.backtest_engine import run_paper_replay_workflow
from autostrategy.core.review import build_review


def _events():
    return [
        {"timestamp": "2024-01-02", "symbol": "A", "action": "buy", "price": 10, "size": 100,
         "status": "filled", "cash_after": 999000, "equity_after": 1000000, "reason": "signal"},
        {"timestamp": "2024-01-03", "symbol": "A", "action": "hold", "price": 9, "size": 0,
         "status": "held", "cash_after": 999000, "equity_after": 999000, "reason": "no signal"},
        {"timestamp": "2024-01-04", "symbol": "A", "action": "sell", "price": 12, "size": 100,
         "status": "filled", "cash_after": 1000200, "equity_after": 1000200, "reason": "take profit"},
        {"timestamp": "2024-01-05", "symbol": "B", "action": "buy", "price": 50, "size": 10000,
         "status": "rejected", "reject_reason": "insufficient_cash", "cash_after": 1000200,
         "equity_after": 1000200, "reason": "signal"},
    ]


def test_build_review_metrics():
    result = {"run_status": "completed",
              "paper": {"initial_cash": 1000000, "final_value": 1000200, "realized_pnl": 200}}
    review = build_review(result, _events())
    metrics = review["metrics"]
    assert metrics["total_return"] == 0.02
    assert metrics["trade_count"] == 2
    assert metrics["buy_count"] == 1
    assert metrics["sell_count"] == 1
    assert metrics["rejected_count"] == 1
    assert metrics["realized_pnl"] == 200
    assert metrics["turnover"] == 10 * 100 + 12 * 100
    # equity 1000000 -> 999000 -> 1000200: drawdown = 1000/1000000 = 0.1%
    assert metrics["max_drawdown"] == 0.1


def test_build_review_key_events_and_markdown():
    review = build_review({"run_status": "completed", "paper": {"initial_cash": 100}}, _events())
    types = [e["type"] for e in review["key_events"]]
    assert types == ["buy", "sell", "rejected"]
    md = review["markdown"]
    assert md.startswith("# Paper Run 复盘")
    assert "BUY A 100 @ 10" in md
    assert "SELL A 100 @ 12" in md
    assert "拒绝 buy B: insufficient_cash" in md


def test_build_review_empty_events():
    review = build_review({"run_status": "completed", "paper": {}}, [])
    assert review["metrics"]["trade_count"] == 0
    assert review["key_events"] == []
    assert "无关键交易事件" in review["markdown"]


def test_workflow_writes_review_artifacts(tmp_path):
    (tmp_path / "strategy.py").write_text(
        "def run_paper(config):\n"
        "    return {\n"
        "        'paper': {'initial_cash': 1000000, 'final_value': 1005000},\n"
        "        'events': [\n"
        "            {'timestamp': '2024-01-02', 'symbol': 'A', 'action': 'buy', 'price': 10,\n"
        "             'size': 100, 'cash_after': 999000, 'equity_after': 1000000, 'reason': 's'},\n"
        "        ],\n"
        "    }\n",
        encoding="utf-8",
    )
    (tmp_path / "config.yaml").write_text("market: A股\n", encoding="utf-8")

    result = run_paper_replay_workflow(tmp_path)

    assert result["run_status"] == "completed"
    assert result["review"]["metrics"]["trade_count"] == 1
    assert result["review"]["key_events"][0]["type"] == "buy"
    review_md = (tmp_path / "paper_run" / "results" / "paper_run_review.md")
    assert review_md.exists()
    assert "# Paper Run 复盘" in review_md.read_text(encoding="utf-8")
    on_disk = json.loads((tmp_path / "paper_run" / "results" / "paper_run_result.json").read_text(encoding="utf-8"))
    assert on_disk["review"]["metrics"]["final_value"] == 1005000


def test_workflow_no_review_for_running_status(tmp_path):
    (tmp_path / "strategy.py").write_text(
        "def run_paper(config):\n    return {'paper': {'initial_cash': 1, 'final_value': 1}, 'events': []}\n",
        encoding="utf-8",
    )
    (tmp_path / "config.yaml").write_text("market: A股\n", encoding="utf-8")

    result = run_paper_replay_workflow(tmp_path)

    # completed run with no events: no review block, no crash
    assert result["run_status"] == "completed"
    assert "review" not in result
