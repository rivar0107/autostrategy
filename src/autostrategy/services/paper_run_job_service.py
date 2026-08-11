"""In-memory paper run job runner for local API usage."""

from __future__ import annotations

import json
import multiprocessing as mp
import queue
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autostrategy.services.models import BacktestJob
from autostrategy.services.paper_run_service import PaperRunService
from autostrategy.services.strategy_service import StrategyService

_ACTIVE_STATUSES = {"queued", "running"}
_TERMINAL_STATUSES = {"succeeded", "failed", "timed_out", "stopped"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _run_paper_worker(
    workspace_root: str | None,
    slug: str,
    stop_event: mp.Event,
    result_queue: mp.Queue,
) -> None:
    try:
        service = PaperRunService(workspace_root=Path(workspace_root) if workspace_root else None)
        result = service.run_paper(slug, stop_requested=stop_event.is_set)
        run_status = result.result.get("run_status")
        result_queue.put(
            {
                "status": "stopped" if run_status == "stopped" else "succeeded",
                "result_path": str(result.result_path),
            }
        )
    except Exception as exc:  # pragma: no cover - exercised through parent process state
        result_queue.put({"status": "failed", "error": str(exc)})


class PaperRunJobService:
    """Manage local paper run jobs."""

    def __init__(self, workspace_root: Path | None = None, timeout_seconds: int = 300) -> None:
        self.workspace_root = workspace_root
        self.timeout_seconds = timeout_seconds
        self.strategy_service = StrategyService(workspace_root=workspace_root)
        self._jobs: dict[str, BacktestJob] = {}
        self._latest_by_slug: dict[str, str] = {}
        self._stop_events: dict[str, mp.Event] = {}
        self._lock = threading.Lock()

    def submit_paper_run(self, slug: str) -> BacktestJob:
        """Create or return an active paper run job for a strategy."""
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

    def request_stop(self, slug: str, job_id: str) -> BacktestJob:
        """Request a running paper job to stop."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.slug != slug:
                raise FileNotFoundError(f"Paper run job '{job_id}' not found.")
            job.stop_requested = True
            self._jobs[job_id] = job
            event = self._stop_events.get(job_id)
            if event:
                event.set()
            return job.model_copy()

    def get_job(self, slug: str, job_id: str) -> BacktestJob:
        """Return a job by id and strategy slug."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.slug != slug:
                raise FileNotFoundError(f"Paper run job '{job_id}' not found.")
            return job.model_copy()

    def _run_job(self, job_id: str) -> None:
        self._update_job(job_id, status="running", started_at=_utc_now())
        result_queue: mp.Queue = mp.Queue()
        stop_event = mp.Event()
        with self._lock:
            self._stop_events[job_id] = stop_event
            slug = self._jobs[job_id].slug
        workspace = str(self.workspace_root) if self.workspace_root else None
        process = mp.Process(
            target=_run_paper_worker, args=(workspace, slug, stop_event, result_queue)
        )
        process.start()
        try:
            process.join(self.timeout_seconds)

            if process.is_alive():
                paths = self.strategy_service.get_strategy_paths(slug)
                process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join()
                error = f"Paper run timed out after {self.timeout_seconds} seconds."
                self._write_timeout_result(paths.paper_run_result, error)
                self._update_job(
                    job_id,
                    status="timed_out",
                    finished_at=_utc_now(),
                    result_path=paths.paper_run_result,
                    error=error,
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
                    error=f"Paper run process exited with code {process.exitcode}.",
                )
                return

            if not result or result.get("status") not in {"succeeded", "stopped"}:
                self._update_job(
                    job_id,
                    status="failed",
                    finished_at=_utc_now(),
                    error=str((result or {}).get("error") or "Paper run failed."),
                )
                return

            self._update_job(
                job_id,
                status=str(result["status"]),
                finished_at=_utc_now(),
                result_path=Path(str(result["result_path"])),
            )
        finally:
            with self._lock:
                self._stop_events.pop(job_id, None)
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
            if data["status"] in _TERMINAL_STATUSES:
                self._stop_events.pop(job_id, None)

    def _write_timeout_result(self, result_path: Path, error: str) -> None:
        started_at = _utc_now()
        result = {
            "mode": "paper_run",
            "run_status": "timed_out",
            "started_at": started_at,
            "updated_at": started_at,
            "replay": {"current_at": None, "bars_processed": 0, "progress": 0.0},
            "summary": {
                "paper_return": 0.0,
                "paper_max_drawdown": 0.0,
                "trade_count": 0,
                "position_count": 0,
                "final_value": 0.0,
            },
            "latest_decision": None,
            "diagnostics": [{"item": "paper_run", "status": "❌", "detail": error}],
            "error": error,
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
