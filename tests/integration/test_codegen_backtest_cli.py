"""Integration tests for codegen and backtest CLI."""

from typer.testing import CliRunner

from autostrategy.cli.main import app

runner = CliRunner()


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


def test_backtest_run_success(tmp_path):
    create = runner.invoke(app, ["strategy", "create", "demo", "--workspace-root", str(tmp_path)])
    assert create.exit_code == 0
    strategy_dir = tmp_path / "demo"
    _write_minimal_coded_strategy(strategy_dir)

    result = runner.invoke(app, ["backtest", "run", "demo", "--workspace-root", str(tmp_path)])

    assert result.exit_code == 0
    assert "Backtest completed" in result.stdout
    assert (strategy_dir / "backtest" / "results" / "backtest_result.json").exists()


def test_backtest_run_missing_strategy_file(tmp_path):
    create = runner.invoke(app, ["strategy", "create", "demo", "--workspace-root", str(tmp_path)])
    assert create.exit_code == 0

    result = runner.invoke(app, ["backtest", "run", "demo", "--workspace-root", str(tmp_path)])

    assert result.exit_code == 1
    assert "strategy.py" in result.stdout


def test_codegen_create_missing_strategy(tmp_path):
    result = runner.invoke(app, ["codegen", "create", "missing", "--workspace-root", str(tmp_path)])

    assert result.exit_code == 1
    assert "not found" in result.stdout
