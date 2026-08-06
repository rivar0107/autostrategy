"""Tests for template registry."""

from autostrategy.core.template_registry import TemplateRegistry


def test_list_templates():
    templates = TemplateRegistry.list_templates()
    assert "dual-ma" in templates
    assert "grid" in templates
    assert "momentum" in templates


def test_apply_template(tmp_path):
    target = tmp_path / "my-strategy"
    target.mkdir()
    TemplateRegistry.apply_template("dual-ma", target)
    assert (target / "config.yaml").exists()
    assert (target / "STRATEGY_DESIGN.md").exists()
