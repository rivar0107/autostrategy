"""Tests for persistent research lifecycle contracts."""

import sqlite3
from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from autostrategy.core.research import (
    DatasetManifest,
    DateRange,
    DiagnosticFinding,
    ExperimentSession,
    ExperimentStatus,
    StrategyVersion,
    StrategyVersionState,
    validate_experiment_transition,
)
from autostrategy.core.strategy import Strategy
from autostrategy.persistence.research_store import ResearchStore


def _version(**overrides) -> StrategyVersion:
    values = {
        "version_id": "version-1",
        "strategy_slug": "demo",
        "version": 1,
        "content_digest": "a" * 64,
        "artifact_path": "/tmp/demo/.autostrategy/versions/version-1",
        "change_summary": "Initial snapshot",
        "state": StrategyVersionState.ACCEPTED,
    }
    values.update(overrides)
    return StrategyVersion(**values)


def _manifest(**overrides) -> DatasetManifest:
    values = {
        "manifest_id": "manifest-1",
        "strategy_slug": "demo",
        "version_id": "version-1",
        "data_source": "fixture",
        "symbols": ["510300.SH"],
        "frequency": "daily",
        "adjustment": "forward",
        "benchmark": "000300.SH",
        "commission": 0.0003,
        "slippage": 0.001,
        "train": {"start": "2020-01-01", "end": "2022-12-31"},
        "validation": {"start": "2023-01-01", "end": "2024-12-31"},
        "test": {"start": "2025-01-01", "end": "2025-12-31"},
        "snapshot_path": "/tmp/demo/.autostrategy/datasets/manifest-1",
        "snapshot_files": {"510300.SH": "frame-0.csv"},
        "output_type": "mapping",
        "data_digest": "b" * 64,
        "locked": True,
    }
    values.update(overrides)
    return DatasetManifest(**values)


def _session(**overrides) -> ExperimentSession:
    values = {
        "session_id": "session-1",
        "strategy_slug": "demo",
        "base_version_id": "version-1",
        "manifest_id": "manifest-1",
        "status": ExperimentStatus.CREATED,
    }
    values.update(overrides)
    return ExperimentSession(**values)


def test_date_range_rejects_reverse_dates():
    with pytest.raises(ValidationError):
        DateRange(start=date(2025, 1, 2), end=date(2025, 1, 1))


def test_dataset_manifest_requires_ordered_non_overlapping_splits():
    with pytest.raises(ValidationError, match="train.*validation.*test"):
        _manifest(
            validation={"start": "2022-12-01", "end": "2024-12-31"},
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"benchmark": ""},
        {"commission": -0.01},
        {"slippage": -0.01},
        {"locked": False},
    ],
)
def test_dataset_manifest_rejects_incomplete_or_mutable_contract(overrides):
    with pytest.raises(ValidationError):
        _manifest(**overrides)


def test_diagnostic_finding_is_structured():
    finding = DiagnosticFinding(
        code="sample.insufficient",
        category="robustness",
        severity="warning",
        evidence={"total_trades": 7},
        hypothesis="交易样本不足导致统计结果不稳定",
        suggested_actions=["延长训练区间"],
        auto_fixable=False,
    )

    assert finding.evidence["total_trades"] == 7
    assert finding.severity == "warning"


def test_experiment_transition_rejects_skipped_and_terminal_transitions():
    validate_experiment_transition(ExperimentStatus.CREATED, ExperimentStatus.BASELINE_COMPLETED)

    with pytest.raises(ValueError, match="Invalid experiment transition"):
        validate_experiment_transition(ExperimentStatus.CREATED, ExperimentStatus.OPTIMIZED)

    with pytest.raises(ValueError, match="Invalid experiment transition"):
        validate_experiment_transition(ExperimentStatus.ACCEPTED, ExperimentStatus.DIAGNOSED)


def test_strategy_supports_legacy_and_research_version_pointers():
    legacy = Strategy(name="legacy")
    versioned = Strategy(
        name="versioned",
        current_version_id="version-2",
        active_version_id="version-1",
    )

    assert legacy.current_version_id is None
    assert legacy.active_version_id is None
    assert versioned.current_version_id == "version-2"
    assert versioned.active_version_id == "version-1"


def test_research_store_persists_versions_manifests_and_sessions(tmp_path):
    store = ResearchStore(tmp_path)
    version = _version()
    manifest = _manifest()
    session = _session()

    store.create_version(version)
    store.create_manifest(manifest)
    store.create_session(session)

    restarted = ResearchStore(tmp_path)
    assert restarted.get_version("demo", "version-1") == version
    assert restarted.list_versions("demo") == [version]
    assert restarted.get_manifest("demo", "manifest-1") == manifest
    assert restarted.get_session("demo", "session-1") == session
    assert restarted.list_sessions("demo") == [session]


def test_research_store_enforces_unique_strategy_version_number(tmp_path):
    store = ResearchStore(tmp_path)
    store.create_version(_version())

    with pytest.raises(sqlite3.IntegrityError):
        store.create_version(_version(version_id="version-other"))


def test_research_store_updates_session_and_version_state(tmp_path):
    store = ResearchStore(tmp_path)
    store.create_version(_version(state=StrategyVersionState.CANDIDATE))
    store.create_manifest(_manifest())
    store.create_session(_session())
    updated_at = datetime.now(UTC)

    session = _session(
        status=ExperimentStatus.BASELINE_COMPLETED,
        baseline_train_run_id="train-run",
        baseline_validation_run_id="validation-run",
        updated_at=updated_at,
    )
    store.update_session(session)
    accepted = store.update_version_state(
        "demo", "version-1", StrategyVersionState.ACCEPTED
    )

    assert store.get_session("demo", "session-1") == session
    assert accepted.state == StrategyVersionState.ACCEPTED
