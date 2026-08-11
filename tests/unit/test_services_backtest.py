"""Tests for backtest service."""

from autostrategy.services.backtest_service import BacktestService
from autostrategy.services.exceptions import BacktestServiceError
from autostrategy.services.strategy_service import StrategyService


def _write_minimal_strategy(strategy_dir):
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


def test_backtest_service_run_and_read_result(tmp_path):
    strategy_service = StrategyService(workspace_root=tmp_path)
    strategy_service.create_strategy("demo")
    _write_minimal_strategy(tmp_path / "demo")

    service = BacktestService(workspace_root=tmp_path)
    result = service.run_backtest("demo")
    saved = service.get_backtest_result("demo")

    assert result.score > 0
    assert saved.result["backtest"]["total_trades"] == 10
    assert result.result_path.exists()

    runs = service.list_backtest_runs("demo")
    assert len(runs) == 1
    assert runs[0].strategy_slug == "demo"
    assert runs[0].strategy_version == 1
    assert runs[0].strategy_digest
    assert runs[0].version_id
    persisted_strategy = service.strategy_service.workspace.get_strategy("demo")
    assert persisted_strategy is not None
    assert persisted_strategy.current_version_id == runs[0].version_id
    assert persisted_strategy.active_version_id == runs[0].version_id
    assert runs[0].result_path != result.result_path
    assert runs[0].result_path.exists()

    restarted = BacktestService(workspace_root=tmp_path)
    persisted = restarted.get_backtest_run("demo", runs[0].run_id)
    assert persisted.result["backtest"]["total_trades"] == 10


def test_backtest_history_is_immutable_across_runs(tmp_path):
    strategy_service = StrategyService(workspace_root=tmp_path)
    strategy_service.create_strategy("demo")
    _write_minimal_strategy(tmp_path / "demo")
    service = BacktestService(workspace_root=tmp_path)

    first = service.run_backtest("demo")
    first_run = service.list_backtest_runs("demo")[0]
    (tmp_path / "demo" / "strategy.py").write_text(
        (tmp_path / "demo" / "strategy.py")
        .read_text(encoding="utf-8")
        .replace("12.0", "13.0"),
        encoding="utf-8",
    )
    service.strategy_service.workspace.bump_strategy_version("demo")
    second = service.run_backtest("demo")
    runs = service.list_backtest_runs("demo")

    assert first.score != second.score
    assert len(runs) == 2
    assert first_run.result_path.read_text(encoding="utf-8") != runs[0].result_path.read_text(
        encoding="utf-8"
    )
    assert {run.strategy_version for run in runs} == {1, 2}
    assert len({run.version_id for run in runs}) == 2


def test_backtest_service_missing_strategy_file(tmp_path):
    strategy_service = StrategyService(workspace_root=tmp_path)
    strategy_service.create_strategy("demo")
    service = BacktestService(workspace_root=tmp_path)

    try:
        service.run_backtest("demo")
    except BacktestServiceError as exc:
        assert "strategy.py" in exc.message
    else:
        raise AssertionError("Expected BacktestServiceError")
