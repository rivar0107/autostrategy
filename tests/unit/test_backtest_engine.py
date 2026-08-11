"""Tests for productized backtest engine."""

import json

from autostrategy.core.backtest_engine import (
    check_pass_criteria,
    run_backtest_workflow,
    run_diagnostics,
    score_strategy,
)


def test_score_strategy_applies_complexity_penalty():
    backtest = {
        "annual_return": 16.0,
        "max_drawdown": 10.0,
        "sharpe": 2.0,
        "win_rate": 60.0,
        "profit_loss_ratio": 2.5,
    }
    simple = {
        "num_buy_conditions": 2,
        "num_sell_conditions": 2,
        "num_filters": 1,
        "num_risk_rules": 2,
    }
    complex_design = {
        "num_buy_conditions": 6,
        "num_sell_conditions": 6,
        "num_filters": 2,
        "num_risk_rules": 2,
    }

    simple_score = score_strategy(backtest, simple, market="A股")
    complex_score = score_strategy(backtest, complex_design, market="A股")

    assert simple_score > complex_score
    assert complex_score == simple_score - 9.0


def test_run_diagnostics_detects_future_leak():
    diagnostics = run_diagnostics({"future_leak_detected": True})
    future = [item for item in diagnostics if item["item"] == "未来函数"][0]
    assert future["status"] == "❌"


def test_check_pass_criteria():
    criteria = check_pass_criteria(
        {
            "annual_return": 5.0,
            "max_drawdown": 10.0,
            "sharpe": 1.2,
            "win_rate": 50.0,
            "profit_loss_ratio": 1.8,
        }
    )
    assert all(item["passed"] for item in criteria)


def test_run_backtest_workflow(tmp_path):
    strategy_dir = tmp_path / "demo"
    strategy_dir.mkdir()
    (strategy_dir / "config.yaml").write_text(
        "market: A股\ninitial_cash: 1000000\nstart_date: '2024-01-01'\nend_date: '2024-12-31'\n",
        encoding="utf-8",
    )
    (strategy_dir / "STRATEGY_DESIGN.md").write_text(
        "# Demo\n\n## 信号逻辑\n\n条件1 买入\n条件2 卖出\n\n## 风控规则\n\n止损\n",
        encoding="utf-8",
    )
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

    result = run_backtest_workflow(strategy_dir)

    assert "error" not in result
    assert result["backtest"]["annual_return"] == 12.0
    assert result["score"] > 0
    result_path = strategy_dir / "backtest" / "results" / "backtest_result.json"
    assert result_path.exists()
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    assert saved["backtest"]["total_trades"] == 10


def test_run_backtest_workflow_rejects_empty_zero_trade_result(tmp_path):
    strategy_dir = tmp_path / "empty-data"
    strategy_dir.mkdir()
    (strategy_dir / "config.yaml").write_text("market: A股\n", encoding="utf-8")
    (strategy_dir / "strategy.py").write_text(
        "def run_backtest(config):\n"
        "    return {\n"
        "        'annual_return': 0.0,\n"
        "        'max_drawdown': 0.0,\n"
        "        'sharpe': 0.0,\n"
        "        'win_rate': 0.0,\n"
        "        'profit_loss_ratio': 0.0,\n"
        "        'total_trades': 0,\n"
        "    }\n",
        encoding="utf-8",
    )

    result = run_backtest_workflow(strategy_dir)

    assert result["score"] == 0
    assert "error" in result
    assert "未产生交易" in result["error"]
    saved = json.loads(
        (strategy_dir / "backtest" / "results" / "backtest_result.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved == result


def test_run_backtest_workflow_rejects_non_finite_metrics(tmp_path):
    strategy_dir = tmp_path / "invalid-metrics"
    strategy_dir.mkdir()
    (strategy_dir / "config.yaml").write_text("market: A股\n", encoding="utf-8")
    (strategy_dir / "strategy.py").write_text(
        "def run_backtest(config):\n"
        "    return {\n"
        "        'annual_return': 12.0, 'max_drawdown': 8.0, 'sharpe': float('nan'),\n"
        "        'win_rate': 55.0, 'profit_loss_ratio': 1.8, 'total_trades': 1,\n"
        "    }\n",
        encoding="utf-8",
    )

    result = run_backtest_workflow(strategy_dir)

    assert result["score"] == 0
    assert "回测结果不符合契约" in result["error"]


def test_run_backtest_workflow_attaches_research_quality(tmp_path):
    strategy_dir = tmp_path / "limited-evidence"
    strategy_dir.mkdir()
    (strategy_dir / "config.yaml").write_text("market: A股\n", encoding="utf-8")
    (strategy_dir / "strategy.py").write_text(
        "def run_backtest(config):\n"
        "    return {\n"
        "        'annual_return': 12.0, 'max_drawdown': 8.0, 'sharpe': 1.5,\n"
        "        'win_rate': 55.0, 'profit_loss_ratio': 1.8, 'total_trades': 10,\n"
        "    }\n",
        encoding="utf-8",
    )

    result = run_backtest_workflow(strategy_dir)

    assert result["research_quality"]["trade_sample"] == "limited"
    assert result["research_quality"]["has_benchmark"] is False
    assert result["research_quality"]["has_out_of_sample"] is False


def test_strategy_can_import_generated_data_fetcher(tmp_path):
    strategy_dir = tmp_path / "generated-import"
    (strategy_dir / "data").mkdir(parents=True)
    (strategy_dir / "config.yaml").write_text("market: A股\n", encoding="utf-8")
    (strategy_dir / "data" / "fetch_data.py").write_text(
        "def fetch(config):\n    return {'loaded': True}\n", encoding="utf-8"
    )
    (strategy_dir / "strategy.py").write_text(
        "def run_backtest(config):\n"
        "    from data.fetch_data import fetch\n"
        "    assert fetch(config)['loaded']\n"
        "    return {\n"
        "        'annual_return': 12.0, 'max_drawdown': 8.0, 'sharpe': 1.5,\n"
        "        'win_rate': 55.0, 'profit_loss_ratio': 1.8, 'total_trades': 1,\n"
        "    }\n",
        encoding="utf-8",
    )

    result = run_backtest_workflow(strategy_dir)

    assert "error" not in result
    assert result["backtest"]["total_trades"] == 1
