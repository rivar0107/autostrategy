"""Compatibility tests for legacy backtest run databases."""

import json
import sqlite3
from contextlib import closing

from autostrategy.persistence.run_store import RunStore


def test_legacy_run_database_adds_research_columns_without_losing_history(tmp_path):
    result_path = tmp_path / "demo" / "legacy-result.json"
    result_path.parent.mkdir(parents=True)
    result = {"score": 42.0, "backtest": {"total_trades": 7}}
    result_path.write_text(json.dumps(result), encoding="utf-8")
    database_path = tmp_path / "runs.sqlite3"
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute(
            """
            CREATE TABLE backtest_runs (
                run_id TEXT PRIMARY KEY,
                strategy_slug TEXT NOT NULL,
                strategy_version INTEGER NOT NULL,
                strategy_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                score REAL NOT NULL,
                result_path TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO backtest_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-run",
                "demo",
                3,
                "legacy-digest",
                "2026-08-10T00:00:00+00:00",
                42.0,
                str(result_path),
                json.dumps(result),
            ),
        )

    store = RunStore(tmp_path)

    summary = store.list_backtest_runs("demo")[0]
    record = store.get_backtest_run("demo", "legacy-run")
    assert summary.run_id == "legacy-run"
    assert summary.version_id is None
    assert summary.manifest_id is None
    assert summary.session_id is None
    assert summary.phase == "full"
    assert summary.candidate_id is None
    assert record is not None
    assert record.result == result
    with closing(sqlite3.connect(database_path)) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(backtest_runs)")
        }
    assert {"version_id", "manifest_id", "session_id", "phase", "candidate_id"} <= columns
