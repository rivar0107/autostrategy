"""Integration tests for CLI."""

from typer.testing import CliRunner

from autostrategy.cli.main import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "autostrategy" in result.stdout
    assert "0.1.0" in result.stdout


def test_config_init():
    result = runner.invoke(app, ["config", "init"])
    assert result.exit_code == 0
    assert "initialized" in result.stdout.lower() or "Settings" in result.stdout


def test_strategy_list_empty():
    result = runner.invoke(app, ["strategy", "list"])
    assert result.exit_code == 0
