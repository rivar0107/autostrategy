"""Safe, configuration-only optimization ratchet."""

from __future__ import annotations

import json
import re
import shutil
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import yaml

from autostrategy.core.backtest_engine import run_backtest_workflow
from autostrategy.core.strategy import StrategyStatus
from autostrategy.services.exceptions import ValidationServiceError
from autostrategy.services.models import (
    OptimizationCandidate,
    OptimizationCandidateResult,
    OptimizationReport,
)
from autostrategy.services.strategy_service import StrategyService
from autostrategy.services.version_service import VersionService

_REPORT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


class OptimizationService:
    """Evaluate isolated config candidates and apply only explicit approvals."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.strategy_service = StrategyService(workspace_root=workspace_root)
        self.version_service = VersionService(workspace_root=workspace_root)

    def evaluate(
        self,
        slug: str,
        candidates: list[OptimizationCandidate],
        minimum_improvement: float = 1.0,
    ) -> OptimizationReport:
        """Evaluate candidates without modifying the live strategy workspace."""
        if minimum_improvement < 0:
            raise ValidationServiceError("minimum_improvement must be non-negative.")
        if not candidates:
            raise ValidationServiceError("At least one optimization candidate is required.")
        names = [candidate.name.strip() for candidate in candidates]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValidationServiceError(
                "Optimization candidate names must be non-empty and unique."
            )

        base_version = self.version_service.ensure_live_version(slug)
        strategy = self.strategy_service.workspace.get_strategy(slug)
        if strategy is None:
            self.strategy_service.get_strategy(slug)
            raise AssertionError("unreachable")
        strategy_dir = self.strategy_service.workspace.get_strategy_dir(slug)
        baseline_config = self._load_config(strategy_dir / "config.yaml")
        baseline_result = self._evaluate_config(strategy_dir, baseline_config)
        baseline_score = float(baseline_result["score"])

        results: list[OptimizationCandidateResult] = []
        for candidate, name in zip(candidates, names, strict=True):
            candidate_config = _deep_merge(baseline_config, candidate.config_overrides)
            try:
                result = self._evaluate_config(strategy_dir, candidate_config)
            except Exception as exc:
                results.append(
                    OptimizationCandidateResult(
                        name=name,
                        config_overrides=candidate.config_overrides,
                        status="failed",
                        error=str(exc),
                    )
                )
                continue
            score = float(result["score"])
            improvement = round(score - baseline_score, 4)
            results.append(
                OptimizationCandidateResult(
                    name=name,
                    config_overrides=candidate.config_overrides,
                    status="evaluated",
                    score=score,
                    improvement=improvement,
                    eligible=improvement > minimum_improvement,
                )
            )

        eligible = [candidate for candidate in results if candidate.eligible]
        recommended = (
            max(eligible, key=lambda candidate: candidate.score or 0) if eligible else None
        )
        report = OptimizationReport(
            report_id=uuid4().hex,
            strategy_slug=slug,
            strategy_version=base_version.version,
            strategy_digest=base_version.content_digest,
            base_version_id=base_version.version_id,
            created_at=datetime.now(UTC).isoformat(),
            baseline_score=baseline_score,
            minimum_improvement=minimum_improvement,
            candidates=results,
            recommended_candidate=recommended.name if recommended else None,
        )
        self._save_report(report)
        return report

    def get_latest_report(self, slug: str) -> OptimizationReport:
        """Read the most recently evaluated optimization report."""
        self.strategy_service.get_strategy(slug)
        path = self._results_dir(slug) / "latest_report.json"
        if not path.exists():
            raise ValidationServiceError(f"Optimization report for strategy '{slug}' not found.")
        return OptimizationReport.model_validate_json(path.read_text(encoding="utf-8"))

    def accept(self, slug: str, report_id: str, candidate_name: str) -> OptimizationReport:
        """Explicitly apply one eligible candidate after freshness checks."""
        report = self._load_report(slug, report_id)
        if report.accepted:
            raise ValidationServiceError("Optimization report has already been accepted.")
        strategy = self.strategy_service.workspace.get_strategy(slug)
        if strategy is None:
            self.strategy_service.get_strategy(slug)
            raise AssertionError("unreachable")
        current_digest = self.strategy_service.workspace.compute_strategy_digest(slug)
        if strategy.version != report.strategy_version or current_digest != report.strategy_digest:
            raise ValidationServiceError(
                "Strategy changed after evaluation; run optimization again before accepting."
            )
        if report.base_version_id and (
            strategy.current_version_id != report.base_version_id
            or strategy.active_version_id != report.base_version_id
        ):
            raise ValidationServiceError(
                "Active strategy version changed after evaluation; run optimization again."
            )
        candidate = next(
            (item for item in report.candidates if item.name == candidate_name), None
        )
        if candidate is None or not candidate.eligible or candidate.status != "evaluated":
            raise ValidationServiceError("Only an eligible evaluated candidate can be accepted.")

        base_version = (
            self.version_service.get_version(slug, report.base_version_id)
            if report.base_version_id
            else self.version_service.ensure_live_version(slug)
        )
        candidate_version = self.version_service.create_candidate_version(
            slug,
            base_version.version_id,
            candidate.config_overrides,
            change_summary=f"Accepted legacy optimization candidate: {candidate.name}",
        )
        accepted_version = self.version_service.accept_version(
            slug, candidate_version.version_id
        )
        self.strategy_service.workspace.update_strategy_status(slug, StrategyStatus.OPTIMIZED)

        report.accepted = True
        report.accepted_candidate = candidate.name
        report.base_version_id = base_version.version_id
        report.accepted_version_id = accepted_version.version_id
        report.accepted_at = datetime.now(UTC).isoformat()
        self._save_report(report)
        return report

    def _evaluate_config(self, strategy_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
        with TemporaryDirectory(prefix="autostrategy-opt-") as temp_dir:
            candidate_dir = Path(temp_dir) / strategy_dir.name
            shutil.copytree(
                strategy_dir,
                candidate_dir,
                ignore=shutil.ignore_patterns(
                    "backtest", "paper_run", "optimization", "__pycache__", "*.pyc"
                ),
            )
            (candidate_dir / "config.yaml").write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
            result = run_backtest_workflow(candidate_dir)
        if "error" in result:
            raise ValidationServiceError(str(result["error"]))
        return result

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(data, dict):
            raise ValidationServiceError("config.yaml must contain a mapping.")
        return data

    def _results_dir(self, slug: str) -> Path:
        return self.strategy_service.workspace.resolve_strategy_path(
            slug, "optimization/results"
        )

    def _save_report(self, report: OptimizationReport) -> None:
        results_dir = self._results_dir(report.strategy_slug)
        results_dir.mkdir(parents=True, exist_ok=True)
        content = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
        (results_dir / f"{report.report_id}.json").write_text(content, encoding="utf-8")
        (results_dir / "latest_report.json").write_text(content, encoding="utf-8")

    def _load_report(self, slug: str, report_id: str) -> OptimizationReport:
        self.strategy_service.get_strategy(slug)
        if not _REPORT_ID_PATTERN.fullmatch(report_id):
            raise ValidationServiceError("Invalid optimization report id.")
        path = self._results_dir(slug) / f"{report_id}.json"
        if not path.exists():
            raise ValidationServiceError(f"Optimization report '{report_id}' not found.")
        return OptimizationReport.model_validate_json(path.read_text(encoding="utf-8"))


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings into an isolated deep copy."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


_TUNABLE_PARAMETER_TOKENS = {
    "lookback",
    "window",
    "period",
    "threshold",
    "stop_loss",
    "take_profit",
    "position",
    "weight",
    "rebalance",
    "holding",
    "fast",
    "slow",
}
_PROTECTED_PARAMETER_TOKENS = {
    "cash",
    "start",
    "end",
    "date",
    "commission",
    "slippage",
    "data_limit",
    "limit",
}


def generate_config_candidates(
    config: dict[str, Any], *, limit: int = 5
) -> list[OptimizationCandidate]:
    """Generate bounded one-parameter candidates from recognized numeric settings."""
    candidates: list[OptimizationCandidate] = []
    for path, value in _numeric_leaf_items(config):
        lowered = path.lower()
        if any(token in lowered for token in _PROTECTED_PARAMETER_TOKENS):
            continue
        if not any(token in lowered for token in _TUNABLE_PARAMETER_TOKENS):
            continue
        for direction, multiplier in (("lower", 0.9), ("raise", 1.1)):
            adjusted: int | float
            if isinstance(value, int):
                delta = max(1, round(abs(value) * 0.1))
                adjusted = value - delta if direction == "lower" else value + delta
            else:
                adjusted = round(value * multiplier, 10)
            if adjusted == value:
                continue
            candidates.append(
                OptimizationCandidate(
                    name=f"auto-{path.replace('.', '-')}-{direction}",
                    config_overrides=_nested_override(path, adjusted),
                )
            )
            if len(candidates) >= max(0, limit):
                return candidates
    return candidates


def _numeric_leaf_items(value: dict[str, Any], prefix: str = "") -> list[tuple[str, int | float]]:
    items: list[tuple[str, int | float]] = []
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, dict):
            items.extend(_numeric_leaf_items(child, path))
        elif isinstance(child, int | float) and not isinstance(child, bool):
            items.append((path, child))
    return items


def _nested_override(path: str, value: int | float) -> dict[str, Any]:
    nested: Any = value
    for part in reversed(path.split(".")):
        nested = {part: nested}
    return nested
