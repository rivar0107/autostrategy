"""Domain contracts for reproducible strategy research experiments."""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

_DIGEST_PATTERN = r"^[a-f0-9]{64}$"


class StrategyVersionState(StrEnum):
    """Lifecycle of an immutable strategy artifact snapshot."""

    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class StrategyVersion(BaseModel):
    """Immutable strategy artifacts plus mutable review state."""

    version_id: str = Field(min_length=1)
    strategy_slug: str = Field(min_length=1)
    version: int = Field(ge=1)
    parent_version_id: str | None = None
    content_digest: str = Field(pattern=_DIGEST_PATTERN)
    artifact_path: Path
    change_summary: str = Field(min_length=1)
    state: StrategyVersionState = StrategyVersionState.CANDIDATE
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DateRange(BaseModel):
    """Inclusive date range used by one research phase."""

    start: date
    end: date

    @model_validator(mode="after")
    def validate_order(self) -> DateRange:
        if self.start > self.end:
            raise ValueError("date range start must be on or before end")
        return self


class DatasetManifest(BaseModel):
    """Locked data snapshot and train/validation/test boundary contract."""

    manifest_id: str = Field(min_length=1)
    strategy_slug: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    data_source: str = Field(min_length=1)
    symbols: list[str] = Field(min_length=1)
    frequency: str = Field(min_length=1)
    adjustment: str = Field(min_length=1)
    benchmark: str = Field(min_length=1)
    commission: float = Field(ge=0)
    slippage: float = Field(ge=0)
    train: DateRange
    validation: DateRange
    test: DateRange
    snapshot_path: Path
    snapshot_files: dict[str, str] = Field(min_length=1)
    output_type: Literal["dataframe", "mapping"]
    data_digest: str = Field(pattern=_DIGEST_PATTERN)
    locked: Literal[True] = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_split_order(self) -> DatasetManifest:
        if not (self.train.end < self.validation.start < self.test.start):
            raise ValueError(
                "train, validation, and test ranges must be ordered and non-overlapping"
            )
        if self.validation.end >= self.test.start:
            raise ValueError(
                "train, validation, and test ranges must be ordered and non-overlapping"
            )
        return self


class DiagnosticFinding(BaseModel):
    """Machine-readable evidence and remediation guidance."""

    code: str = Field(min_length=1)
    category: Literal[
        "data", "signal", "risk", "execution", "overfit", "leakage", "robustness"
    ]
    severity: Literal["info", "warning", "critical"]
    evidence: dict[str, Any] = Field(default_factory=dict)
    hypothesis: str = Field(min_length=1)
    suggested_actions: list[str] = Field(default_factory=list)
    auto_fixable: bool = False


class ExperimentCandidate(BaseModel):
    """One isolated optimization hypothesis and its research evidence."""

    candidate_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    status: Literal["proposed", "evaluated", "failed", "selected"] = "proposed"
    train_run_id: str | None = None
    validation_run_id: str | None = None
    train_score: float | None = None
    validation_score: float | None = None
    improvement: float | None = None
    eligible: bool = False
    version_id: str | None = None
    error: str | None = None


class ExperimentStatus(StrEnum):
    """Persistent state machine for one strategy research experiment."""

    CREATED = "created"
    BASELINE_COMPLETED = "baseline_completed"
    DIAGNOSED = "diagnosed"
    OPTIMIZED = "optimized"
    OOS_VALIDATED = "oos_validated"
    AWAITING_DECISION = "awaiting_decision"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"


_EXPERIMENT_TRANSITIONS: dict[ExperimentStatus, set[ExperimentStatus]] = {
    ExperimentStatus.CREATED: {
        ExperimentStatus.BASELINE_COMPLETED,
        ExperimentStatus.FAILED,
    },
    ExperimentStatus.BASELINE_COMPLETED: {
        ExperimentStatus.DIAGNOSED,
        ExperimentStatus.FAILED,
    },
    ExperimentStatus.DIAGNOSED: {
        ExperimentStatus.OPTIMIZED,
        ExperimentStatus.FAILED,
    },
    ExperimentStatus.OPTIMIZED: {
        ExperimentStatus.OOS_VALIDATED,
        ExperimentStatus.FAILED,
    },
    ExperimentStatus.OOS_VALIDATED: {
        ExperimentStatus.AWAITING_DECISION,
        ExperimentStatus.FAILED,
    },
    ExperimentStatus.AWAITING_DECISION: {
        ExperimentStatus.ACCEPTED,
        ExperimentStatus.REJECTED,
        ExperimentStatus.FAILED,
    },
    ExperimentStatus.ACCEPTED: set(),
    ExperimentStatus.REJECTED: set(),
    ExperimentStatus.FAILED: set(),
}


def validate_experiment_transition(
    current: ExperimentStatus, target: ExperimentStatus
) -> None:
    """Reject skipped, repeated, or terminal experiment transitions."""
    if target not in _EXPERIMENT_TRANSITIONS[current]:
        raise ValueError(f"Invalid experiment transition: {current.value} -> {target.value}")


class ExperimentSession(BaseModel):
    """Complete audit record for one research and optimization lifecycle."""

    session_id: str = Field(min_length=1)
    strategy_slug: str = Field(min_length=1)
    base_version_id: str = Field(min_length=1)
    manifest_id: str = Field(min_length=1)
    status: ExperimentStatus = ExperimentStatus.CREATED
    baseline_train_run_id: str | None = None
    baseline_validation_run_id: str | None = None
    diagnostics: list[DiagnosticFinding] = Field(default_factory=list)
    candidates: list[ExperimentCandidate] = Field(default_factory=list)
    selected_candidate_id: str | None = None
    selected_version_id: str | None = None
    oos_base_run_id: str | None = None
    oos_candidate_run_id: str | None = None
    oos_revealed: bool = False
    oos_passed: bool | None = None
    decision: Literal["accepted", "rejected"] | None = None
    decision_reason: str | None = None
    accepted_version_id: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VersionEvent(BaseModel):
    """Auditable human decision that changes or rejects a version pointer."""

    event_id: str = Field(min_length=1)
    strategy_slug: str = Field(min_length=1)
    action: Literal["accept", "reject", "rollback"]
    from_version_id: str = Field(min_length=1)
    to_version_id: str = Field(min_length=1)
    session_id: str | None = None
    reason: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
