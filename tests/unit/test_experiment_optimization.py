"""Tests for leak-resistant experiment optimization and one-time OOS."""

import pytest
import yaml
from test_experiment_service import _create_research_inputs

from autostrategy.core.research import ExperimentStatus
from autostrategy.services.exceptions import ValidationServiceError
from autostrategy.services.experiment_service import ExperimentService
from autostrategy.services.models import OptimizationCandidate
from autostrategy.services.optimization_service import generate_config_candidates


def _diagnosed_session(tmp_path):
    slug, version, manifest = _create_research_inputs(tmp_path)
    service = ExperimentService(workspace_root=tmp_path)
    created = service.create_session(slug, version.version_id, manifest.manifest_id)
    baseline = service.run_baseline(slug, created.session_id)
    diagnosed = service.diagnose(slug, baseline.session_id)
    return slug, version, manifest, service, diagnosed


def test_candidate_generator_changes_one_allowlisted_parameter_and_excludes_costs():
    candidates = generate_config_candidates(
        {
            "lookback": 20,
            "risk": {"stop_loss": 0.08},
            "commission": 0.0003,
            "start_date": "2020-01-01",
            "data_limit": 500,
        },
        limit=5,
    )

    assert candidates
    assert len(candidates) <= 5
    for candidate in candidates:
        assert len(_leaf_paths(candidate.config_overrides)) == 1
        path = _leaf_paths(candidate.config_overrides)[0]
        assert "commission" not in path
        assert "start_date" not in path
        assert "data_limit" not in path


def test_optimization_uses_train_and_validation_without_touching_live_workspace(tmp_path):
    slug, base_version, manifest, service, session = _diagnosed_session(tmp_path)
    live_config_before = service.workspace.read_text_file(slug, "config.yaml")

    optimized = service.optimize(
        slug,
        session.session_id,
        [
            OptimizationCandidate(name="better", config_overrides={"alpha": 4.0}),
            OptimizationCandidate(name="worse", config_overrides={"alpha": -4.0}),
        ],
        minimum_improvement=1.0,
    )

    assert optimized.status == ExperimentStatus.OPTIMIZED
    assert optimized.selected_candidate_id is not None
    selected = next(
        candidate
        for candidate in optimized.candidates
        if candidate.candidate_id == optimized.selected_candidate_id
    )
    assert selected.name == "better"
    assert selected.eligible is True
    assert selected.version_id is not None
    phases = {
        run.phase
        for run in service.run_store.list_backtest_runs(slug)
        if run.session_id == session.session_id
    }
    assert phases == {"train", "validation"}
    assert service.workspace.read_text_file(slug, "config.yaml") == live_config_before
    live_strategy = service.workspace.get_strategy(slug)
    assert live_strategy is not None
    assert live_strategy.active_version_id == base_version.version_id
    assert manifest.test.start.isoformat() not in live_config_before


def test_explicit_candidate_must_change_exactly_one_configuration_leaf(tmp_path):
    slug, base_version, _, service, session = _diagnosed_session(tmp_path)

    unchanged = service.optimize(
        slug,
        session.session_id,
        [
            OptimizationCandidate(
                name="confounded-change",
                config_overrides={"alpha": 4.0, "risk": {"stop_loss": 3.0}},
            )
        ],
        minimum_improvement=1.0,
    )

    assert unchanged.status == ExperimentStatus.DIAGNOSED
    assert unchanged.selected_candidate_id is None
    assert unchanged.candidates[0].status == "failed"
    assert "exactly one" in (unchanged.candidates[0].error or "")
    assert service.version_service.list_versions(slug) == [base_version]


def test_oos_is_revealed_once_for_base_and_selected_candidate(tmp_path):
    slug, _, _, service, session = _diagnosed_session(tmp_path)
    optimized = service.optimize(
        slug,
        session.session_id,
        [OptimizationCandidate(name="better", config_overrides={"alpha": 4.0})],
        minimum_improvement=1.0,
    )

    validated = service.validate_oos(slug, optimized.session_id)

    assert validated.status == ExperimentStatus.AWAITING_DECISION
    assert validated.oos_revealed is True
    assert validated.oos_passed is True
    assert validated.oos_base_run_id
    assert validated.oos_candidate_run_id
    test_runs = [
        run
        for run in service.run_store.list_backtest_runs(slug)
        if run.session_id == session.session_id and run.phase == "test"
    ]
    assert len(test_runs) == 2

    with pytest.raises(ValidationServiceError, match="already been revealed"):
        service.validate_oos(slug, optimized.session_id)
    test_runs_after = [
        run
        for run in service.run_store.list_backtest_runs(slug)
        if run.session_id == session.session_id and run.phase == "test"
    ]
    assert len(test_runs_after) == 2


def test_failed_oos_is_consumed_and_cannot_be_revealed_again(tmp_path):
    slug, _, _, service, session = _diagnosed_session(tmp_path)
    optimized = service.optimize(
        slug,
        session.session_id,
        [OptimizationCandidate(name="better", config_overrides={"alpha": 4.0})],
        minimum_improvement=1.0,
    )
    selected = next(candidate for candidate in optimized.candidates if candidate.eligible)
    candidate = service.version_service.get_version(slug, selected.version_id or "")
    config_path = candidate.artifact_path / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["alpha"] = -20.0
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValidationServiceError, match="digest"):
        service.validate_oos(slug, optimized.session_id)

    failed = service.get_session(slug, optimized.session_id)
    assert failed.status == ExperimentStatus.FAILED
    assert failed.oos_revealed is True
    assert failed.oos_base_run_id
    assert "digest" in (failed.error or "")
    with pytest.raises(ValidationServiceError, match="already been revealed"):
        service.validate_oos(slug, optimized.session_id)


def _leaf_paths(value, prefix=""):
    paths = []
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            paths.extend(_leaf_paths(item, path))
        else:
            paths.append(path)
    return paths
