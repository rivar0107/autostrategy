"""In-memory design job runner for local API usage."""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autostrategy.config import LLMConfig
from autostrategy.core.strategy import StrategyStatus
from autostrategy.services.design_service import DesignService
from autostrategy.services.exceptions import (
    AutostrategyServiceError,
    LLMConfigurationRequiredError,
    ValidationServiceError,
)
from autostrategy.services.models import DesignJob
from autostrategy.services.strategy_service import StrategyService

_ACTIVE_STATUSES = {"queued", "running"}
_TERMINAL_STATUSES = {"succeeded", "failed"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class DesignJobService:
    """Manage local design generation jobs in background threads."""

    def __init__(
        self,
        workspace_root: Path | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        self.workspace_root = workspace_root
        self.llm_config = llm_config
        self._jobs: dict[str, DesignJob] = {}
        self._lock = threading.Lock()

    def submit_design(
        self,
        name: str,
        prompt: str,
        market: str = "A股",
        template: str | None = None,
    ) -> DesignJob:
        """Create and start a design generation job."""
        job = DesignJob(
            job_id=uuid.uuid4().hex,
            name=name,
            status="queued",
            created_at=_utc_now(),
        )
        with self._lock:
            self._jobs[job.job_id] = job

        thread = threading.Thread(
            target=self._run_job,
            args=(job.job_id, name, prompt, market, template),
            daemon=True,
        )
        thread.start()
        return self.get_job(job.job_id)

    def submit_redesign(self, slug: str, prompt: str) -> DesignJob:
        """Regenerate the design document for an existing draft strategy."""
        strategy_service = StrategyService(workspace_root=self.workspace_root)
        strategy = strategy_service.get_strategy(slug)
        if strategy.status != StrategyStatus.DRAFT:
            raise ValidationServiceError(
                f"Strategy '{slug}' is '{strategy.status}', "
                "only draft strategies can be redesigned."
            )
        job = DesignJob(
            job_id=uuid.uuid4().hex,
            name=strategy.name,
            status="queued",
            created_at=_utc_now(),
        )
        with self._lock:
            self._jobs[job.job_id] = job

        thread = threading.Thread(
            target=self._run_redesign,
            args=(job.job_id, slug, prompt, strategy.market, strategy.template),
            daemon=True,
        )
        thread.start()
        return self.get_job(job.job_id)

    def get_job(self, job_id: str) -> DesignJob:
        """Return a job by id."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise FileNotFoundError(f"Design job '{job_id}' not found.")
            return job.model_copy()

    def _run_job(
        self,
        job_id: str,
        name: str,
        prompt: str,
        market: str,
        template: str | None,
    ) -> None:
        self._update_job(job_id, status="running", started_at=_utc_now())
        try:
            service = DesignService(workspace_root=self.workspace_root, llm_config=self.llm_config)
            result = service.create_design(
                name=name,
                prompt=prompt,
                market=market,
                template=template,
            )
            self._update_job(
                job_id,
                status="succeeded",
                finished_at=_utc_now(),
                strategy=result.strategy,
                design_path=result.design_path,
            )
        except LLMConfigurationRequiredError as exc:
            self._update_job(
                job_id,
                status="failed",
                finished_at=_utc_now(),
                error=str(exc),
                error_code="llm_not_configured",
            )
        except ValidationServiceError as exc:
            self._update_job(
                job_id,
                status="failed",
                finished_at=_utc_now(),
                error=str(exc),
                error_code="validation_error",
            )
        except AutostrategyServiceError as exc:
            self._update_job(
                job_id,
                status="failed",
                finished_at=_utc_now(),
                error=str(exc),
                error_code="service_error",
            )
        except Exception as exc:  # pragma: no cover - unexpected failure
            self._update_job(
                job_id,
                status="failed",
                finished_at=_utc_now(),
                error=f"Design generation failed: {exc}",
                error_code="unexpected_error",
            )

    def _run_redesign(
        self,
        job_id: str,
        slug: str,
        prompt: str,
        market: str,
        template: str | None,
    ) -> None:
        self._update_job(job_id, status="running", started_at=_utc_now())
        try:
            service = DesignService(workspace_root=self.workspace_root, llm_config=self.llm_config)
            result = service.redesign(slug=slug, prompt=prompt, market=market, template=template)
            self._update_job(
                job_id,
                status="succeeded",
                finished_at=_utc_now(),
                strategy=result.strategy,
                design_path=result.design_path,
            )
        except LLMConfigurationRequiredError as exc:
            self._update_job(
                job_id,
                status="failed",
                finished_at=_utc_now(),
                error=str(exc),
                error_code="llm_not_configured",
            )
        except ValidationServiceError as exc:
            self._update_job(
                job_id,
                status="failed",
                finished_at=_utc_now(),
                error=str(exc),
                error_code="validation_error",
            )
        except AutostrategyServiceError as exc:
            self._update_job(
                job_id,
                status="failed",
                finished_at=_utc_now(),
                error=str(exc),
                error_code="service_error",
            )
        except Exception as exc:  # pragma: no cover - unexpected failure
            self._update_job(
                job_id,
                status="failed",
                finished_at=_utc_now(),
                error=f"Design generation failed: {exc}",
                error_code="unexpected_error",
            )

    def _update_job(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            data = job.model_dump()
            data.update(updates)
            if data["status"] in _TERMINAL_STATUSES and data.get("finished_at") is None:
                data["finished_at"] = _utc_now()
            self._jobs[job_id] = DesignJob(**data)
