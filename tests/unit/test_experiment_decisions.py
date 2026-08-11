"""Tests for explicit experiment decisions and pointer-based rollback."""

import pytest
import yaml
from test_experiment_optimization import _diagnosed_session

from autostrategy.core.research import ExperimentStatus, StrategyVersionState
from autostrategy.services.exceptions import ValidationServiceError
from autostrategy.services.models import OptimizationCandidate


def _awaiting_decision(tmp_path, *, oos_passes: bool = True):
    slug, base, _, service, diagnosed = _diagnosed_session(tmp_path)
    optimized = service.optimize(
        slug,
        diagnosed.session_id,
        [OptimizationCandidate(name="better", config_overrides={"alpha": 4.0})],
        minimum_improvement=1.0,
    )
    degradation = 5.0 if oos_passes else -10.0
    awaiting = service.validate_oos(
        slug,
        optimized.session_id,
        maximum_score_degradation=degradation,
    )
    return slug, base, service, awaiting


def test_accept_promotes_selected_snapshot_and_records_explicit_decision(tmp_path):
    slug, base, service, session = _awaiting_decision(tmp_path)

    accepted = service.accept(slug, session.session_id, reason="样本外门槛全部通过")
    strategy = service.workspace.get_strategy(slug)
    config = yaml.safe_load(service.workspace.read_text_file(slug, "config.yaml"))
    version = service.version_service.get_version(slug, accepted.accepted_version_id or "")
    events = service.store.list_version_events(slug)

    assert accepted.status == ExperimentStatus.ACCEPTED
    assert accepted.decision == "accepted"
    assert accepted.decision_reason == "样本外门槛全部通过"
    assert version.state == StrategyVersionState.ACCEPTED
    assert strategy is not None
    assert strategy.active_version_id == version.version_id
    assert strategy.current_version_id == version.version_id
    assert strategy.active_version_id != base.version_id
    assert config["alpha"] == 4.0
    assert events[-1].action == "accept"
    assert events[-1].session_id == session.session_id


def test_accept_rejects_failed_oos_and_stale_base_version(tmp_path):
    slug, _, service, failed_oos = _awaiting_decision(tmp_path, oos_passes=False)
    assert failed_oos.oos_passed is False

    with pytest.raises(ValidationServiceError, match="did not pass"):
        service.accept(slug, failed_oos.session_id, reason="不应接受")

    slug2, base2, service2, session2 = _awaiting_decision(tmp_path / "stale")
    service2.workspace.write_text_file(slug2, "README.md", "# externally changed\n")
    assert service2.workspace.compute_strategy_digest(slug2) != base2.content_digest

    with pytest.raises(ValidationServiceError, match="changed after the experiment started"):
        service2.accept(slug2, session2.session_id, reason="基础版本已过期")

    slug3, _, service3, session3 = _awaiting_decision(tmp_path / "stale-pointer")
    metadata_path = service3.workspace.get_strategy_dir(slug3) / "strategy.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    metadata["current_version_id"] = "externally-repointed"
    metadata_path.write_text(
        yaml.safe_dump(metadata, allow_unicode=True), encoding="utf-8"
    )

    with pytest.raises(ValidationServiceError, match="changed after the experiment started"):
        service3.accept(slug3, session3.session_id, reason="版本指针已变化")


def test_reject_keeps_base_active_and_rejects_candidate(tmp_path):
    slug, base, service, session = _awaiting_decision(tmp_path)
    original = service.workspace.read_text_file(slug, "config.yaml")

    rejected = service.reject(slug, session.session_id, reason="不接受该参数变化")
    strategy = service.workspace.get_strategy(slug)
    candidate = service.version_service.get_version(slug, session.selected_version_id or "")

    assert rejected.status == ExperimentStatus.REJECTED
    assert rejected.decision == "rejected"
    assert strategy is not None
    assert strategy.active_version_id == base.version_id
    assert candidate.state == StrategyVersionState.REJECTED
    assert service.workspace.read_text_file(slug, "config.yaml") == original

    with pytest.raises(ValidationServiceError, match="awaiting_decision"):
        service.reject(slug, session.session_id, reason="重复拒绝")


def test_rollback_restores_accepted_ancestor_and_keeps_audit_history(tmp_path):
    slug, base, service, session = _awaiting_decision(tmp_path)
    accepted = service.accept(slug, session.session_id, reason="先接受用于测试")
    accepted_version_id = accepted.accepted_version_id

    restored = service.rollback(slug, base.version_id, reason="线上观察后回滚")
    strategy = service.workspace.get_strategy(slug)
    config = yaml.safe_load(service.workspace.read_text_file(slug, "config.yaml"))
    events = service.store.list_version_events(slug)

    assert restored.version_id == base.version_id
    assert strategy is not None
    assert strategy.active_version_id == base.version_id
    assert config["alpha"] == 0.0
    assert service.version_service.get_version(slug, accepted_version_id or "") is not None
    assert events[-1].action == "rollback"
    assert events[-1].from_version_id == accepted_version_id
    assert events[-1].to_version_id == base.version_id
