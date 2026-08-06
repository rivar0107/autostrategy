"""In-memory backtest job runner for local API usage."""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autostrategy.services.backtest_service import BacktestService
from autostrategy.services.models import BacktestJob, BacktestJobStatus
from autostrategy.services.strategy_service import StrategyService

_ACTIVE_STATUSES = {"queued", "running"}
_TERMINAL_STATUSES = {"succeeded", "failed", "timed_out"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _run_backtest_worker(
    workspace_root: str | None,
    slug: str,
    result_queue: mp.Queue,
) -> None:
    try:
        service = BacktestService(workspace_root=Path(workspace_root) if workspace_root else None)
        result = service.run_backtest(slug)
        result_queue.put(
            {
                "status": "succeeded",
                "result_path": str(result.result_path),
                "score": result.score,
            }
        )
    except Exception as exc:  # pragma: no cover - exercised through parent process state
        result_queue.put({"status": "failed", "error": str(exc)})


class BacktestJobService:
    """Manage local backtest jobs and isolate strategy execution in subprocesses."""

    def __init__(self, workspace_root: Path | None = None, timeout_seconds: int = 300) -> None:
        self.workspace_root = workspace_root
        self.timeout_seconds = timeout_seconds
        self.strategy_service = StrategyService(workspace_root=workspace_root)
        self._jobs: dict[str, BacktestJob] = {}
        self._latest_by_slug: dict[str, str] = {}
        self._lock = threading.Lock()

    def submit_backtest(self, slug: str) -> BacktestJob:
        """Create or return an active backtest job for a strategy."""
        self.strategy_service.get_strategy(slug)
        with self._lock:
            existing_id = self._latest_by_slug.get(slug)
            if existing_id:
                existing = self._jobs.get(existing_id)
                if existing and existing.status in _ACTIVE_STATUSES:
                    return existing

            job = BacktestJob(
                job_id=uuid.uuid4().hex,
                slug=slug,
                status="queued",
                created_at=_utc_now(),
            )
            self._jobs[job.job_id] = job
            self._latest_by_slug[slug] = job.job_id

        thread = threading.Thread(target=self._run_job, args=(job.job_id,), daemon=True)
        thread.start()
        return self.get_job(slug, job.job_id)

    def get_job(self, slug: str, job_id: str) -> BacktestJob:
        """Return a job by id and strategy slug."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.slug != slug:
                raise FileNotFoundError(f"Backtest job '{job_id}' not found.")
            return job.model_copy()

    def get_latest_job(self, slug: str) -> BacktestJob | None:
        """Return the latest in-memory job for a strategy."""
        with self._lock:
            job_id = self._latest_by_slug.get(slug)
            if not job_id:
                return None
            job = self._jobs.get(job_id)
            return job.model_copy() if job else None

    def _run_job(self, job_id: str) -> None:
        self._update_job(job_id, status="running", started_at=_utc_now())
        result_queue: mp.Queue = mp.Queue()
        workspace = str(self.workspace_root) if self.workspace_root else None
        process = mp.Process(target=_run_backtest_worker, args=(workspace, self._jobs[job_id].slug, result_queue))
        process.start()
        try:
            process.join(self.timeout_seconds)

            if process.is_alive():
                process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join()
                self._update_job(
                    job_id,
                    status="timed_out",
                    finished_at=_utc_now(),
                    error=f"Backtest timed out after {self.timeout_seconds} seconds.",
                )
                return

            result: dict[str, Any] | None = None
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                result = None

            if process.exitcode != 0 and result is None:
                self._update_job(
                    job_id,
                    status="failed",
                    finished_at=_utc_now(),
                    error=f"Backtest process exited with code {process.exitcode}.",
                )
                return

            if not result or result.get("status") != "succeeded":
                self._update_job(
                    job_id,
                    status="failed",
                    finished_at=_utc_now(),
                    error=str((result or {}).get("error") or "Backtest failed."),
                )
                return

            self._update_job(
                job_id,
                status="succeeded",
                finished_at=_utc_now(),
                result_path=Path(str(result["result_path"])),
                score=float(result["score"]),
            )
        finally:
            result_queue.close()
            result_queue.join_thread()

    def _update_job(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            data = job.model_dump()
            data.update(updates)
            if data["status"] in _TERMINAL_STATUSES and data.get("finished_at") is None:
                data["finished_at"] = _utc_now()
            self._jobs[job_id] = BacktestJob(**data)
