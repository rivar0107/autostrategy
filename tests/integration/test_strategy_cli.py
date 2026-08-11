"""Integration tests for strategy CLI."""

from typer.testing import CliRunner

from autostrategy.cli.main import app

runner = CliRunner()


def test_strategy_create(tmp_path):
    result = runner.invoke(
        app, ["strategy", "create", "dual-ma-test", "--workspace-root", str(tmp_path)]
    )
    assert result.exit_code == 0
    assert "created" in result.stdout.lower()


def test_strategy_list(tmp_path):
    runner.invoke(app, ["strategy", "create", "s1", "--workspace-root", str(tmp_path)])
    result = runner.invoke(app, ["strategy", "list", "--workspace-root", str(tmp_path)])
    assert result.exit_code == 0
    assert "s1" in result.stdout


def test_strategy_show(tmp_path):
    runner.invoke(app, ["strategy", "create", "s1", "--workspace-root", str(tmp_path)])
    result = runner.invoke(app, ["strategy", "show", "s1", "--workspace-root", str(tmp_path)])
    assert result.exit_code == 0
    assert "Name: s1" in result.stdout
    assert "Workspace:" in result.stdout
    assert "STRATEGY_DESIGN.md" in result.stdout


def test_strategy_paths(tmp_path):
    runner.invoke(app, ["strategy", "create", "s1", "--workspace-root", str(tmp_path)])
    result = runner.invoke(app, ["strategy", "paths", "s1", "--workspace-root", str(tmp_path)])
    assert result.exit_code == 0
    assert "Workspace:" in result.stdout
    assert "Design:" in result.stdout
    assert "Backtest result:" in result.stdout


def test_strategy_delete(tmp_path):
    runner.invoke(app, ["strategy", "create", "s1", "--workspace-root", str(tmp_path)])
    result = runner.invoke(
        app, ["strategy", "delete", "s1", "--workspace-root", str(tmp_path), "--yes"]
    )
    assert result.exit_code == 0
    assert "deleted" in result.stdout.lower()
