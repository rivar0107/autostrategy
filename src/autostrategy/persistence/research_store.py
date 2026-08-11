"""SQLite persistence for strategy versions, datasets, and experiments."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from autostrategy.core.research import (
    DatasetManifest,
    ExperimentSession,
    StrategyVersion,
    StrategyVersionState,
    VersionEvent,
    validate_experiment_transition,
)


class ResearchStore:
    """Persist immutable research inputs and resumable experiment state."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.database_path = workspace_root / "research.sqlite3"
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS strategy_versions (
                    version_id TEXT PRIMARY KEY,
                    strategy_slug TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    parent_version_id TEXT,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    UNIQUE(strategy_slug, version)
                );
                CREATE INDEX IF NOT EXISTS idx_strategy_versions_slug
                    ON strategy_versions(strategy_slug, version);

                CREATE TABLE IF NOT EXISTS dataset_manifests (
                    manifest_id TEXT PRIMARY KEY,
                    strategy_slug TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    data_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_dataset_manifests_slug
                    ON dataset_manifests(strategy_slug, created_at DESC);

                CREATE TABLE IF NOT EXISTS experiment_sessions (
                    session_id TEXT PRIMARY KEY,
                    strategy_slug TEXT NOT NULL,
                    base_version_id TEXT NOT NULL,
                    manifest_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_experiment_sessions_slug
                    ON experiment_sessions(strategy_slug, created_at DESC);

                CREATE TABLE IF NOT EXISTS version_events (
                    event_id TEXT PRIMARY KEY,
                    strategy_slug TEXT NOT NULL,
                    action TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_version_events_slug
                    ON version_events(strategy_slug, created_at ASC);
                """
            )

    def create_version(self, version: StrategyVersion) -> StrategyVersion:
        """Insert one immutable version identity and payload."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO strategy_versions (
                    version_id, strategy_slug, version, parent_version_id,
                    state, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version.version_id,
                    version.strategy_slug,
                    version.version,
                    version.parent_version_id,
                    version.state.value,
                    version.created_at.isoformat(),
                    version.model_dump_json(),
                ),
            )
        return version

    def get_version(self, strategy_slug: str, version_id: str) -> StrategyVersion | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM strategy_versions
                WHERE strategy_slug = ? AND version_id = ?
                """,
                (strategy_slug, version_id),
            ).fetchone()
        return StrategyVersion.model_validate_json(row["payload_json"]) if row else None

    def list_versions(self, strategy_slug: str) -> list[StrategyVersion]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM strategy_versions
                WHERE strategy_slug = ? ORDER BY version ASC
                """,
                (strategy_slug,),
            ).fetchall()
        return [StrategyVersion.model_validate_json(row["payload_json"]) for row in rows]

    def update_version_state(
        self,
        strategy_slug: str,
        version_id: str,
        state: StrategyVersionState,
    ) -> StrategyVersion:
        version = self.get_version(strategy_slug, version_id)
        if version is None:
            raise KeyError(f"Strategy version '{version_id}' not found.")
        updated = version.model_copy(update={"state": state})
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE strategy_versions SET state = ?, payload_json = ?
                WHERE strategy_slug = ? AND version_id = ?
                """,
                (state.value, updated.model_dump_json(), strategy_slug, version_id),
            )
        return updated

    def create_manifest(self, manifest: DatasetManifest) -> DatasetManifest:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO dataset_manifests (
                    manifest_id, strategy_slug, version_id, data_digest,
                    created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.manifest_id,
                    manifest.strategy_slug,
                    manifest.version_id,
                    manifest.data_digest,
                    manifest.created_at.isoformat(),
                    manifest.model_dump_json(),
                ),
            )
        return manifest

    def get_manifest(self, strategy_slug: str, manifest_id: str) -> DatasetManifest | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM dataset_manifests
                WHERE strategy_slug = ? AND manifest_id = ?
                """,
                (strategy_slug, manifest_id),
            ).fetchone()
        return DatasetManifest.model_validate_json(row["payload_json"]) if row else None

    def list_manifests(self, strategy_slug: str) -> list[DatasetManifest]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM dataset_manifests
                WHERE strategy_slug = ? ORDER BY created_at DESC
                """,
                (strategy_slug,),
            ).fetchall()
        return [DatasetManifest.model_validate_json(row["payload_json"]) for row in rows]

    def create_session(self, session: ExperimentSession) -> ExperimentSession:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO experiment_sessions (
                    session_id, strategy_slug, base_version_id, manifest_id,
                    status, created_at, updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.strategy_slug,
                    session.base_version_id,
                    session.manifest_id,
                    session.status.value,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                    session.model_dump_json(),
                ),
            )
        return session

    def get_session(self, strategy_slug: str, session_id: str) -> ExperimentSession | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM experiment_sessions
                WHERE strategy_slug = ? AND session_id = ?
                """,
                (strategy_slug, session_id),
            ).fetchone()
        return ExperimentSession.model_validate_json(row["payload_json"]) if row else None

    def list_sessions(self, strategy_slug: str) -> list[ExperimentSession]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM experiment_sessions
                WHERE strategy_slug = ? ORDER BY created_at DESC
                """,
                (strategy_slug,),
            ).fetchall()
        return [ExperimentSession.model_validate_json(row["payload_json"]) for row in rows]

    def update_session(self, session: ExperimentSession) -> ExperimentSession:
        existing = self.get_session(session.strategy_slug, session.session_id)
        if existing is None:
            raise KeyError(f"Experiment session '{session.session_id}' not found.")
        if existing.status != session.status:
            validate_experiment_transition(existing.status, session.status)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE experiment_sessions
                SET status = ?, updated_at = ?, payload_json = ?
                WHERE strategy_slug = ? AND session_id = ?
                """,
                (
                    session.status.value,
                    session.updated_at.isoformat(),
                    session.model_dump_json(),
                    session.strategy_slug,
                    session.session_id,
                ),
            )
        return session

    def create_version_event(self, event: VersionEvent) -> VersionEvent:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO version_events (
                    event_id, strategy_slug, action, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.strategy_slug,
                    event.action,
                    event.created_at.isoformat(),
                    event.model_dump_json(),
                ),
            )
        return event

    def list_version_events(self, strategy_slug: str) -> list[VersionEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM version_events
                WHERE strategy_slug = ? ORDER BY created_at ASC, event_id ASC
                """,
                (strategy_slug,),
            ).fetchall()
        return [VersionEvent.model_validate_json(row["payload_json"]) for row in rows]
