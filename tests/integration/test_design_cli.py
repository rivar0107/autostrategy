"""Integration tests for design CLI."""

from typer.testing import CliRunner

from autostrategy.cli.main import app

runner = CliRunner()


def test_design_command_requires_prompt():
    result = runner.invoke(app, ["design", "create", "--name", "s1"])
    assert result.exit_code != 0
    combined_output = f"{result.stdout}\n{result.stderr}".lower()
    assert "prompt" in combined_output or "missing" in combined_output


def test_config_set(tmp_path, monkeypatch):
    settings_path = tmp_path / "settings.yaml"
    monkeypatch.setattr("autostrategy.config.get_default_settings_path", lambda: settings_path)
    result = runner.invoke(app, ["config", "set", "llm.provider", "deepseek"])
    assert result.exit_code == 0
    assert "Set llm.provider = deepseek" in result.stdout
