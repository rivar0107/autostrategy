"""CLI serve command tests."""

from typer.testing import CliRunner

from autostrategy.cli.main import app

runner = CliRunner()


def test_serve_help():
    result = runner.invoke(app, ["serve", "--help"])

    assert result.exit_code == 0
    assert "Start the local REST API" in result.stdout
    assert "--host" in result.stdout
