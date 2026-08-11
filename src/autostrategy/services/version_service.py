"""Immutable strategy artifact snapshots and pointer-based activation."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import Any
from uuid import uuid4

import yaml

from autostrategy.core.research import StrategyVersion, StrategyVersionState
from autostrategy.core.workspace import (
    VERSIONED_ARTIFACTS,
    Workspace,
    compute_artifact_digest,
)
from autostrategy.persistence.research_store import ResearchStore
from autostrategy.services.exceptions import StrategyNotFoundError, ValidationServiceError


class VersionService:
    """Create, verify, select, and restore immutable strategy versions."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace = Workspace(root=workspace_root)
        self.store = ResearchStore(self.workspace.root)

    def ensure_current_version(self, slug: str) -> StrategyVersion:
        """Lazily migrate a live or legacy workspace into its first snapshot."""
        strategy = self.workspace.get_strategy(slug)
        if strategy is None:
            raise StrategyNotFoundError(f"Strategy '{slug}' not found.")
        if strategy.current_version_id:
            version = self.store.get_version(slug, strategy.current_version_id)
            if version is None:
                raise ValidationServiceError(
                    f"Strategy version pointer '{strategy.current_version_id}' is missing."
                )
            self._verify_snapshot(version)
            return version

        existing = self.store.list_versions(slug)
        if existing:
            version = existing[-1]
            self._verify_snapshot(version)
        else:
            version = self._create_snapshot(
                slug=slug,
                source_root=self.workspace.get_strategy_dir(slug),
                version_number=strategy.version,
                parent_version_id=None,
                change_summary="Initial workspace snapshot",
                state=StrategyVersionState.ACCEPTED,
            )
        self.workspace.set_strategy_version_pointers(
            slug,
            version=version.version,
            content_digest=version.content_digest,
            current_version_id=version.version_id,
            active_version_id=version.version_id,
        )
        return version

    def ensure_live_version(self, slug: str) -> StrategyVersion:
        """Bind the current live artifacts to an immutable accepted version.

        A strategy can still be edited outside the version service (for example by
        a user editing ``strategy.py`` directly).  Never attribute such bytes to
        the previously active snapshot: capture them as a new accepted child
        before an ordinary backtest is recorded.
        """
        current = self.ensure_current_version(slug)
        live_digest = self.workspace.compute_strategy_digest(slug)
        if live_digest == current.content_digest:
            return current
        return self.create_version_from_live(
            slug,
            parent_version_id=current.version_id,
            change_summary="Workspace change captured before backtest",
            state=StrategyVersionState.ACCEPTED,
            activate=True,
        )

    def create_candidate_version(
        self,
        slug: str,
        base_version_id: str,
        config_overrides: dict[str, Any],
        *,
        change_summary: str,
    ) -> StrategyVersion:
        """Create a config-only child snapshot without touching live artifacts."""
        base = self.get_version(slug, base_version_id)
        self._verify_snapshot(base)
        with TemporaryDirectory(prefix="autostrategy-version-") as temp_dir:
            candidate_root = Path(temp_dir)
            self._copy_artifacts(base.artifact_path, candidate_root)
            config_path = candidate_root / "config.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if not isinstance(config, dict):
                raise ValidationServiceError("config.yaml must contain a mapping.")
            merged = _deep_merge(config, config_overrides)
            config_path.write_text(
                yaml.safe_dump(merged, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
            versions = self.store.list_versions(slug)
            next_version = max((version.version for version in versions), default=0) + 1
            return self._create_snapshot(
                slug=slug,
                source_root=candidate_root,
                version_number=next_version,
                parent_version_id=base.version_id,
                change_summary=change_summary,
                state=StrategyVersionState.CANDIDATE,
            )

    def create_version_from_live(
        self,
        slug: str,
        *,
        parent_version_id: str | None,
        change_summary: str,
        state: StrategyVersionState = StrategyVersionState.ACCEPTED,
        activate: bool = True,
    ) -> StrategyVersion:
        """Snapshot the current live artifacts after a successful controlled change."""
        strategy = self.workspace.get_strategy(slug)
        if strategy is None:
            raise StrategyNotFoundError(f"Strategy '{slug}' not found.")
        versions = self.store.list_versions(slug)
        next_version = max((version.version for version in versions), default=0) + 1
        version_number = max(strategy.version, next_version)
        version = self._create_snapshot(
            slug=slug,
            source_root=self.workspace.get_strategy_dir(slug),
            version_number=version_number,
            parent_version_id=parent_version_id,
            change_summary=change_summary,
            state=state,
        )
        if activate:
            self._activate_snapshot(version)
        return version

    def get_version(self, slug: str, version_id: str) -> StrategyVersion:
        version = self.store.get_version(slug, version_id)
        if version is None:
            raise ValidationServiceError(f"Strategy version '{version_id}' not found.")
        return version

    def list_versions(self, slug: str) -> list[StrategyVersion]:
        if self.workspace.get_strategy(slug) is None:
            raise StrategyNotFoundError(f"Strategy '{slug}' not found.")
        return self.store.list_versions(slug)

    def accept_version(self, slug: str, version_id: str) -> StrategyVersion:
        version = self.get_version(slug, version_id)
        if version.state != StrategyVersionState.CANDIDATE:
            raise ValidationServiceError("Only a candidate version can be accepted.")
        self._verify_snapshot(version)
        accepted = self.store.update_version_state(
            slug, version_id, StrategyVersionState.ACCEPTED
        )
        try:
            self._activate_snapshot(accepted)
        except Exception:
            self.store.update_version_state(slug, version_id, StrategyVersionState.CANDIDATE)
            raise
        return accepted

    def activate_version(self, slug: str, version_id: str) -> StrategyVersion:
        version = self.get_version(slug, version_id)
        if version.state != StrategyVersionState.ACCEPTED:
            raise ValidationServiceError("Only an accepted strategy version can be activated.")
        self._verify_snapshot(version)
        self._activate_snapshot(version)
        return version

    def rollback(self, slug: str, version_id: str) -> StrategyVersion:
        """Restore any accepted version while retaining all newer snapshots."""
        return self.activate_version(slug, version_id)

    def reject_version(self, slug: str, version_id: str) -> StrategyVersion:
        version = self.get_version(slug, version_id)
        if version.state != StrategyVersionState.CANDIDATE:
            raise ValidationServiceError("Only a candidate version can be rejected.")
        return self.store.update_version_state(slug, version_id, StrategyVersionState.REJECTED)

    def materialize_version(self, slug: str, version_id: str, destination: Path) -> Path:
        """Copy a verified version into an isolated executable workspace."""
        version = self.get_version(slug, version_id)
        self._verify_snapshot(version)
        destination.mkdir(parents=True, exist_ok=False)
        self._copy_artifacts(version.artifact_path, destination)
        return destination

    def _create_snapshot(
        self,
        *,
        slug: str,
        source_root: Path,
        version_number: int,
        parent_version_id: str | None,
        change_summary: str,
        state: StrategyVersionState,
    ) -> StrategyVersion:
        version_id = uuid4().hex
        versions_root = self.workspace.resolve_strategy_path(slug, ".autostrategy/versions")
        versions_root.mkdir(parents=True, exist_ok=True)
        target = versions_root / version_id
        temp = Path(mkdtemp(prefix=".snapshot-", dir=versions_root))
        try:
            self._copy_artifacts(source_root, temp)
            content_digest = compute_artifact_digest(temp)
            version = StrategyVersion(
                version_id=version_id,
                strategy_slug=slug,
                version=version_number,
                parent_version_id=parent_version_id,
                content_digest=content_digest,
                artifact_path=target,
                change_summary=change_summary,
                state=state,
            )
            (temp / "version.json").write_text(
                json.dumps(version.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temp.replace(target)
            try:
                self.store.create_version(version)
            except Exception:
                shutil.rmtree(target)
                raise
            return version
        finally:
            if temp.exists():
                shutil.rmtree(temp)

    @staticmethod
    def _copy_artifacts(source_root: Path, destination: Path) -> None:
        for relative_path in VERSIONED_ARTIFACTS:
            source = source_root / relative_path
            if not source.is_file():
                continue
            target = destination / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    def _verify_snapshot(self, version: StrategyVersion) -> None:
        actual_digest = compute_artifact_digest(version.artifact_path)
        if actual_digest != version.content_digest:
            raise ValidationServiceError(
                f"Strategy version digest mismatch for '{version.version_id}'."
            )

    def _activate_snapshot(self, version: StrategyVersion) -> None:
        strategy_dir = self.workspace.get_strategy_dir(version.strategy_slug)
        metadata_path = strategy_dir / "strategy.yaml"
        original_metadata = metadata_path.read_bytes()
        with TemporaryDirectory(prefix="autostrategy-restore-") as backup_dir_text:
            backup_dir = Path(backup_dir_text)
            self._copy_artifacts(strategy_dir, backup_dir)
            try:
                self._restore_artifacts(version.artifact_path, strategy_dir)
                self.workspace.set_strategy_version_pointers(
                    version.strategy_slug,
                    version=version.version,
                    content_digest=version.content_digest,
                    current_version_id=version.version_id,
                    active_version_id=version.version_id,
                )
            except Exception:
                self._restore_artifacts(backup_dir, strategy_dir)
                metadata_path.write_bytes(original_metadata)
                raise

    @staticmethod
    def _restore_artifacts(snapshot_root: Path, strategy_dir: Path) -> None:
        for relative_path in VERSIONED_ARTIFACTS:
            snapshot = snapshot_root / relative_path
            live = strategy_dir / relative_path
            if snapshot.is_file():
                live.parent.mkdir(parents=True, exist_ok=True)
                staged = live.with_name(f".{live.name}.restore-{uuid4().hex}")
                staged.write_bytes(snapshot.read_bytes())
                staged.replace(live)
            elif live.exists():
                live.unlink()


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged
