"""Tests for code generation version boundaries."""

from autostrategy.core.strategy import StrategyStatus
from autostrategy.services.codegen_service import CodegenService
from autostrategy.services.strategy_service import StrategyService
from autostrategy.services.version_service import VersionService


def test_codegen_preserves_design_version_and_activates_generated_version(
    tmp_path, monkeypatch
):
    strategy_service = StrategyService(workspace_root=tmp_path)
    strategy = strategy_service.create_strategy("demo")
    strategy_service.workspace.write_text_file(
        strategy.slug,
        "STRATEGY_DESIGN.md",
        "# Demo\n\n## 策略概述\n\n一个完整设计。\n",
    )
    service = CodegenService(workspace_root=tmp_path)

    def fake_codegen_and_save(workspace, slug, force=False):
        del force
        workspace.write_text_file(
            slug,
            "strategy.py",
            "def run_backtest(config):\n    return {'annual_return': 1}\n",
        )
        workspace.update_strategy_status(slug, StrategyStatus.CODED)
        return workspace.bump_strategy_version(slug)

    monkeypatch.setattr(service.agent, "codegen_and_save", fake_codegen_and_save)

    result = service.generate_code(strategy.slug)

    versions = VersionService(workspace_root=tmp_path).list_versions(strategy.slug)
    assert len(versions) == 2
    design_version, generated_version = versions
    assert not (design_version.artifact_path / "strategy.py").exists()
    assert (generated_version.artifact_path / "strategy.py").exists()
    assert generated_version.parent_version_id == design_version.version_id
    assert result.strategy.current_version_id == generated_version.version_id
    assert result.strategy.active_version_id == generated_version.version_id
