"""SQLite-backed immutable history for strategy executions."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from autostrategy.services.models import BacktestRunRecord, BacktestRunSummary


class RunStore:
    """Persist and query backtest run snapshots across service restarts."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.database_path = workspace_root / "runs.sqlite3"
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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    run_id TEXT PRIMARY KEY,
                    strategy_slug TEXT NOT NULL,
                    strategy_version INTEGER NOT NULL,
                    strategy_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    score REAL NOT NULL,
                    result_path TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    version_id TEXT,
                    manifest_id TEXT,
                    session_id TEXT,
                    phase TEXT NOT NULL DEFAULT 'full',
                    candidate_id TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_backtest_runs_strategy_created
                ON backtest_runs(strategy_slug, created_at DESC)
                """
            )
            existing_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(backtest_runs)").fetchall()
            }
            migrations = {
                "version_id": "ALTER TABLE backtest_runs ADD COLUMN version_id TEXT",
                "manifest_id": "ALTER TABLE backtest_runs ADD COLUMN manifest_id TEXT",
                "session_id": "ALTER TABLE backtest_runs ADD COLUMN session_id TEXT",
                "phase": (
                    "ALTER TABLE backtest_runs ADD COLUMN phase TEXT NOT NULL DEFAULT 'full'"
                ),
                "candidate_id": "ALTER TABLE backtest_runs ADD COLUMN candidate_id TEXT",
            }
            for column, statement in migrations.items():
                if column not in existing_columns:
                    connection.execute(statement)

    def record_backtest(
        self,
        *,
        strategy_slug: str,
        strategy_version: int,
        strategy_digest: str,
        score: float,
        result: dict,
        version_id: str | None = None,
        manifest_id: str | None = None,
        session_id: str | None = None,
        phase: str = "full",
        candidate_id: str | None = None,
    ) -> BacktestRunRecord:
        """Write an immutable JSON snapshot and its SQLite index row."""
        run_id = uuid4().hex
        created_at = datetime.now(UTC).isoformat()
        result_json = json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True)
        result_path = (
            self.workspace_root
            / strategy_slug
            / "backtest"
            / "results"
            / "runs"
            / f"{run_id}.json"
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO backtest_runs (
                    run_id, strategy_slug, strategy_version, strategy_digest,
                    created_at, score, result_path, result_json, version_id,
                    manifest_id, session_id, phase, candidate_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    strategy_slug,
                    strategy_version,
                    strategy_digest,
                    created_at,
                    score,
                    str(result_path),
                    result_json,
                    version_id,
                    manifest_id,
                    session_id,
                    phase,
                    candidate_id,
                ),
            )
        return BacktestRunRecord(
            run_id=run_id,
            strategy_slug=strategy_slug,
            strategy_version=strategy_version,
            strategy_digest=strategy_digest,
            created_at=created_at,
            score=score,
            result_path=result_path,
            result=result,
            version_id=version_id,
            manifest_id=manifest_id,
            session_id=session_id,
            phase=phase,
            candidate_id=candidate_id,
        )

    def list_backtest_runs(self, strategy_slug: str) -> list[BacktestRunSummary]:
        """List newest-first run metadata for a strategy."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, strategy_slug, strategy_version, strategy_digest,
                       created_at, score, result_path, version_id, manifest_id,
                       session_id, phase, candidate_id
                FROM backtest_runs
                WHERE strategy_slug = ?
                ORDER BY created_at DESC, run_id DESC
                """,
                (strategy_slug,),
            ).fetchall()
        return [BacktestRunSummary(**dict(row)) for row in rows]

    def get_backtest_run(self, strategy_slug: str, run_id: str) -> BacktestRunRecord | None:
        """Return one run only when it belongs to the requested strategy."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT run_id, strategy_slug, strategy_version, strategy_digest,
                       created_at, score, result_path, result_json, version_id,
                       manifest_id, session_id, phase, candidate_id
                FROM backtest_runs
                WHERE strategy_slug = ? AND run_id = ?
                """,
                (strategy_slug, run_id),
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["result"] = json.loads(data.pop("result_json"))
        return BacktestRunRecord(**data)
