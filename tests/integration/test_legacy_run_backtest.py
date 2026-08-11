"""Integration tests for legacy run_backtest.py wrapper."""

import subprocess
import sys


def test_legacy_run_backtest_wrapper(tmp_path):
    strategy_dir = tmp_path / "demo"
    strategy_dir.mkdir()
    (strategy_dir / "config.yaml").write_text(
        "market: A股\ninitial_cash: 1000000\n", encoding="utf-8"
    )
    (strategy_dir / "STRATEGY_DESIGN.md").write_text("# Demo\n", encoding="utf-8")
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

    result = subprocess.run(
        [sys.executable, "scripts/run_backtest.py", str(strategy_dir)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "score_strategy" in result.stdout
    assert (strategy_dir / "backtest" / "results" / "backtest_result.json").exists()
