"""Tests for immutable strategy artifact versions."""

import json

import pytest
import yaml

from autostrategy.core.research import StrategyVersionState
from autostrategy.core.workspace import VERSIONED_ARTIFACTS, Workspace
from autostrategy.services.exceptions import ValidationServiceError
from autostrategy.services.version_service import VersionService


def _create_complete_strategy(tmp_path):
    workspace = Workspace(root=tmp_path)
    strategy = workspace.create_strategy("versioned")
    contents = {
        "STRATEGY_DESIGN.md": "# Design v1\n",
        "strategy.py": "def run_backtest(config):\n    return {}\n",
        "config.yaml": "market: A股\nlookback: 20\nrisk:\n  stop_loss: 0.08\n",
        "data/fetch_data.py": "def fetch(config):\n    return None\n",
        "requirements.txt": "pandas\n",
        "README.md": "# Versioned\n",
    }
    for relative_path, content in contents.items():
        workspace.write_text_file(strategy.slug, relative_path, content)
    workspace.refresh_strategy_digest(strategy.slug)
    return workspace, strategy.slug, contents


def test_ensure_current_version_migrates_legacy_strategy_and_snapshots_artifacts(tmp_path):
    workspace, slug, contents = _create_complete_strategy(tmp_path)
    service = VersionService(workspace_root=tmp_path)

    version = service.ensure_current_version(slug)
    repeated = service.ensure_current_version(slug)
    strategy = workspace.get_strategy(slug)

    assert repeated.version_id == version.version_id
    assert version.version == 1
    assert version.state == StrategyVersionState.ACCEPTED
    assert strategy is not None
    assert strategy.current_version_id == version.version_id
    assert strategy.active_version_id == version.version_id
    for relative_path in VERSIONED_ARTIFACTS:
        assert (version.artifact_path / relative_path).read_text(encoding="utf-8") == contents[
            relative_path
        ]
    metadata = json.loads((version.artifact_path / "version.json").read_text(encoding="utf-8"))
    assert metadata["content_digest"] == version.content_digest


def test_version_snapshot_tampering_is_detected_before_activation(tmp_path):
    _, slug, _ = _create_complete_strategy(tmp_path)
    service = VersionService(workspace_root=tmp_path)
    version = service.ensure_current_version(slug)
    (version.artifact_path / "config.yaml").write_text("market: corrupted\n", encoding="utf-8")

    with pytest.raises(ValidationServiceError, match="digest"):
        service.activate_version(slug, version.version_id)


def test_candidate_version_is_isolated_until_explicit_acceptance(tmp_path):
    workspace, slug, _ = _create_complete_strategy(tmp_path)
    service = VersionService(workspace_root=tmp_path)
    baseline = service.ensure_current_version(slug)
    live_config_before = workspace.read_text_file(slug, "config.yaml")

    candidate = service.create_candidate_version(
        slug,
        baseline.version_id,
        {"lookback": 30, "risk": {"stop_loss": 0.05}},
        change_summary="Tighten stop loss and extend lookback",
    )

    candidate_config = yaml.safe_load(
        (candidate.artifact_path / "config.yaml").read_text(encoding="utf-8")
    )
    assert candidate.parent_version_id == baseline.version_id
    assert candidate.state == StrategyVersionState.CANDIDATE
    assert candidate.version == 2
    assert candidate_config["lookback"] == 30
    assert candidate_config["risk"]["stop_loss"] == 0.05
    assert workspace.read_text_file(slug, "config.yaml") == live_config_before

    accepted = service.accept_version(slug, candidate.version_id)
    strategy = workspace.get_strategy(slug)
    live_config = yaml.safe_load(workspace.read_text_file(slug, "config.yaml"))

    assert accepted.state == StrategyVersionState.ACCEPTED
    assert strategy is not None
    assert strategy.version == 2
    assert strategy.current_version_id == candidate.version_id
    assert strategy.active_version_id == candidate.version_id
    assert live_config == candidate_config


def test_rollback_restores_parent_bytes_without_deleting_newer_version(tmp_path):
    workspace, slug, contents = _create_complete_strategy(tmp_path)
    service = VersionService(workspace_root=tmp_path)
    baseline = service.ensure_current_version(slug)
    candidate = service.create_candidate_version(
        slug,
        baseline.version_id,
        {"lookback": 30},
        change_summary="Longer lookback",
    )
    service.accept_version(slug, candidate.version_id)

    restored = service.rollback(slug, baseline.version_id)
    strategy = workspace.get_strategy(slug)

    assert restored.version_id == baseline.version_id
    assert workspace.read_text_file(slug, "config.yaml") == contents["config.yaml"]
    assert service.get_version(slug, candidate.version_id) is not None
    assert strategy is not None
    assert strategy.current_version_id == baseline.version_id
    assert strategy.active_version_id == baseline.version_id


def test_activation_failure_restores_live_files_pointers_and_candidate_state(
    tmp_path, monkeypatch
):
    workspace, slug, _ = _create_complete_strategy(tmp_path)
    service = VersionService(workspace_root=tmp_path)
    baseline = service.ensure_current_version(slug)
    candidate = service.create_candidate_version(
        slug,
        baseline.version_id,
        {"lookback": 99},
        change_summary="Activation failure fixture",
    )
    strategy_dir = workspace.get_strategy_dir(slug)
    original_files = {
        path: (strategy_dir / path).read_bytes()
        for path in VERSIONED_ARTIFACTS
        if (strategy_dir / path).is_file()
    }
    original_metadata = (strategy_dir / "strategy.yaml").read_bytes()
    original_restore = service._restore_artifacts
    calls = 0

    def fail_first_restore(snapshot_root, destination):
        nonlocal calls
        calls += 1
        if calls == 1:
            (destination / "config.yaml").write_text("partially: restored\n", encoding="utf-8")
            raise RuntimeError("injected restore failure")
        original_restore(snapshot_root, destination)

    monkeypatch.setattr(service, "_restore_artifacts", fail_first_restore)

    with pytest.raises(RuntimeError, match="injected restore failure"):
        service.accept_version(slug, candidate.version_id)

    strategy = workspace.get_strategy(slug)
    assert strategy is not None
    assert strategy.current_version_id == baseline.version_id
    assert strategy.active_version_id == baseline.version_id
    assert (strategy_dir / "strategy.yaml").read_bytes() == original_metadata
    for path, content in original_files.items():
        assert (strategy_dir / path).read_bytes() == content
    assert service.get_version(slug, candidate.version_id).state == StrategyVersionState.CANDIDATE


def test_rejected_candidate_never_changes_live_workspace(tmp_path):
    workspace, slug, _ = _create_complete_strategy(tmp_path)
    service = VersionService(workspace_root=tmp_path)
    baseline = service.ensure_current_version(slug)
    original = workspace.read_text_file(slug, "config.yaml")
    candidate = service.create_candidate_version(
        slug,
        baseline.version_id,
        {"lookback": 2},
        change_summary="Rejected lookback",
    )

    rejected = service.reject_version(slug, candidate.version_id)

    assert rejected.state == StrategyVersionState.REJECTED
    assert workspace.read_text_file(slug, "config.yaml") == original
    assert service.get_version(slug, candidate.version_id) == rejected
