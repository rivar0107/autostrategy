"""Strategy artifact preview service."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from autostrategy.services.exceptions import ArtifactNotFoundError, StrategyNotFoundError
from autostrategy.services.strategy_service import StrategyService

ARTIFACTS: Final[dict[str, tuple[str, str]]] = {
    "design": ("STRATEGY_DESIGN.md", "markdown"),
    "strategy_code": ("strategy.py", "python"),
    "config": ("config.yaml", "yaml"),
    "readme": ("README.md", "markdown"),
    "requirements": ("requirements.txt", "text"),
    "fetch_data": ("data/fetch_data.py", "python"),
    "backtest_result": ("backtest/results/backtest_result.json", "json"),
    "paper_run_result": ("paper_run/results/paper_run_result.json", "json"),
    "paper_run_events": ("paper_run/results/paper_run_events.jsonl", "text"),
    "paper_run_log": ("paper_run/logs/paper_run.log", "text"),
}


class ArtifactService:
    """Application service for named strategy artifacts."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.strategy_service = StrategyService(workspace_root=workspace_root)

    def list_artifacts(self, slug: str) -> dict:
        """List previewable artifacts for a strategy."""
        self._ensure_strategy(slug)
        artifacts = []
        for key in ARTIFACTS:
            artifacts.append(self._artifact_meta(slug, key))
        return {"slug": slug, "artifacts": artifacts}

    def get_artifact(self, slug: str, artifact_key: str) -> dict:
        """Read a named text artifact for preview."""
        self._ensure_strategy(slug)
        if artifact_key not in ARTIFACTS:
            raise ArtifactNotFoundError(f"Artifact '{artifact_key}' is not supported.")

        meta = self._artifact_meta(slug, artifact_key)
        if not meta["exists"]:
            raise ArtifactNotFoundError(
                f"Artifact '{artifact_key}' for strategy '{slug}' not found."
            )

        path = self.strategy_service.workspace.resolve_strategy_path(slug, meta["relative_path"])
        content = path.read_text(encoding="utf-8")
        if meta["content_type"] == "json":
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = None
            meta["parsed_json"] = parsed
        meta["content"] = content
        return meta

    def _ensure_strategy(self, slug: str) -> None:
        try:
            self.strategy_service.get_strategy(slug)
        except StrategyNotFoundError:
            raise

    def _artifact_meta(self, slug: str, artifact_key: str) -> dict:
        relative_path, content_type = ARTIFACTS[artifact_key]
        path = self.strategy_service.workspace.resolve_strategy_path(slug, relative_path)
        exists = path.exists() and path.is_file()
        stat = path.stat() if exists else None
        modified_at = None
        if stat:
            modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
        return {
            "slug": slug,
            "artifact_key": artifact_key,
            "relative_path": relative_path,
            "path": path,
            "exists": exists,
            "size": stat.st_size if stat else 0,
            "modified_at": modified_at,
            "content_type": content_type,
        }
