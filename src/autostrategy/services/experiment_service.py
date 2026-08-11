"""Persistent orchestration for reproducible strategy research experiments."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import yaml

from autostrategy.core.backtest_engine import run_backtest_workflow
from autostrategy.core.research import (
    DiagnosticFinding,
    ExperimentCandidate,
    ExperimentSession,
    ExperimentStatus,
    StrategyVersionState,
    VersionEvent,
)
from autostrategy.persistence.research_store import ResearchStore
from autostrategy.persistence.run_store import RunStore
from autostrategy.services.dataset_manifest_service import DatasetManifestService
from autostrategy.services.exceptions import ValidationServiceError
from autostrategy.services.models import BacktestRunRecord, OptimizationCandidate
from autostrategy.services.optimization_service import generate_config_candidates
from autostrategy.services.version_service import VersionService


class ExperimentService:
    """Advance one persisted experiment through explicit research steps."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.version_service = VersionService(workspace_root=workspace_root)
        self.dataset_service = DatasetManifestService(workspace_root=workspace_root)
        self.workspace = self.version_service.workspace
        self.store = ResearchStore(self.workspace.root)
        self.run_store = RunStore(self.workspace.root)

    def create_session(
        self,
        slug: str,
        base_version_id: str,
        manifest_id: str,
    ) -> ExperimentSession:
        """Bind one immutable strategy version to one locked dataset."""
        version = self.version_service.get_version(slug, base_version_id)
        manifest = self.dataset_service.get_manifest(slug, manifest_id)
        if manifest.version_id != version.version_id:
            raise ValidationServiceError(
                "Experiment base version must match the dataset capture version."
            )
        session = ExperimentSession(
            session_id=uuid4().hex,
            strategy_slug=slug,
            base_version_id=version.version_id,
            manifest_id=manifest.manifest_id,
        )
        return self.store.create_session(session)

    def get_session(self, slug: str, session_id: str) -> ExperimentSession:
        session = self.store.get_session(slug, session_id)
        if session is None:
            raise ValidationServiceError(f"Experiment session '{session_id}' not found.")
        return session

    def list_sessions(self, slug: str) -> list[ExperimentSession]:
        self.version_service.ensure_current_version(slug)
        return self.store.list_sessions(slug)

    def run_baseline(self, slug: str, session_id: str) -> ExperimentSession:
        """Run the immutable base version on train and validation only."""
        session = self._require_status(slug, session_id, ExperimentStatus.CREATED)
        try:
            train = self._run_version_split(
                session,
                version_id=session.base_version_id,
                split="train",
                phase="train",
            )
            validation = self._run_version_split(
                session,
                version_id=session.base_version_id,
                split="validation",
                phase="validation",
            )
        except Exception as exc:
            self._fail_session(session, str(exc))
            raise
        updated = session.model_copy(
            update={
                "status": ExperimentStatus.BASELINE_COMPLETED,
                "baseline_train_run_id": train.run_id,
                "baseline_validation_run_id": validation.run_id,
                "updated_at": datetime.now(UTC),
            }
        )
        return self.store.update_session(updated)

    def diagnose(self, slug: str, session_id: str) -> ExperimentSession:
        """Convert validation evidence into structured, actionable findings."""
        session = self._require_status(
            slug, session_id, ExperimentStatus.BASELINE_COMPLETED
        )
        run = self.run_store.get_backtest_run(slug, session.baseline_validation_run_id or "")
        if run is None:
            raise ValidationServiceError("Baseline validation run is missing.")
        findings = _build_findings(run.result)
        updated = session.model_copy(
            update={
                "status": ExperimentStatus.DIAGNOSED,
                "diagnostics": findings,
                "updated_at": datetime.now(UTC),
            }
        )
        return self.store.update_session(updated)

    def optimize(
        self,
        slug: str,
        session_id: str,
        candidates: list[OptimizationCandidate] | None = None,
        *,
        minimum_improvement: float = 1.0,
        minimum_trades: int = 30,
        maximum_drawdown: float = 20.0,
    ) -> ExperimentSession:
        """Evaluate candidates on train/validation without revealing test data."""
        session = self._require_status(slug, session_id, ExperimentStatus.DIAGNOSED)
        baseline_run = self.run_store.get_backtest_run(
            slug, session.baseline_validation_run_id or ""
        )
        if baseline_run is None:
            raise ValidationServiceError("Baseline validation run is missing.")
        base_version = self.version_service.get_version(slug, session.base_version_id)
        config = _load_version_config(base_version.artifact_path / "config.yaml")
        proposals = candidates if candidates is not None else generate_config_candidates(config)
        if not proposals:
            raise ValidationServiceError("No safe tunable configuration parameters were found.")

        evaluated: list[ExperimentCandidate] = []
        for proposal in proposals[:5]:
            candidate_id = uuid4().hex
            hypothesis = f"验证单一配置变更：{proposal.name}"
            changed_paths = _config_leaf_paths(proposal.config_overrides)
            if len(changed_paths) != 1:
                evaluated.append(
                    ExperimentCandidate(
                        candidate_id=candidate_id,
                        name=proposal.name,
                        hypothesis=hypothesis,
                        config_overrides=proposal.config_overrides,
                        status="failed",
                        error="Optimization candidate must change exactly one configuration leaf.",
                    )
                )
                continue
            version = None
            try:
                version = self.version_service.create_candidate_version(
                    slug,
                    session.base_version_id,
                    proposal.config_overrides,
                    change_summary=hypothesis,
                )
                train_run = self._run_version_split(
                    session,
                    version_id=version.version_id,
                    split="train",
                    phase="train",
                    candidate_id=candidate_id,
                )
                validation_run = self._run_version_split(
                    session,
                    version_id=version.version_id,
                    split="validation",
                    phase="validation",
                    candidate_id=candidate_id,
                )
                metrics = validation_run.result["backtest"]
                improvement = round(validation_run.score - baseline_run.score, 4)
                eligible = (
                    improvement > minimum_improvement
                    and int(metrics["total_trades"]) >= minimum_trades
                    and float(metrics["max_drawdown"]) <= maximum_drawdown
                )
                evaluated.append(
                    ExperimentCandidate(
                        candidate_id=candidate_id,
                        name=proposal.name,
                        hypothesis=hypothesis,
                        config_overrides=proposal.config_overrides,
                        status="evaluated",
                        train_run_id=train_run.run_id,
                        validation_run_id=validation_run.run_id,
                        train_score=train_run.score,
                        validation_score=validation_run.score,
                        improvement=improvement,
                        eligible=eligible,
                        version_id=version.version_id,
                    )
                )
            except Exception as exc:
                if version is not None:
                    self.version_service.reject_version(slug, version.version_id)
                evaluated.append(
                    ExperimentCandidate(
                        candidate_id=candidate_id,
                        name=proposal.name,
                        hypothesis=hypothesis,
                        config_overrides=proposal.config_overrides,
                        status="failed",
                        version_id=version.version_id if version else None,
                        error=str(exc),
                    )
                )

        eligible_candidates = [candidate for candidate in evaluated if candidate.eligible]
        if not eligible_candidates:
            updated = session.model_copy(
                update={"candidates": evaluated, "updated_at": datetime.now(UTC)}
            )
            return self.store.update_session(updated)
        selected = max(
            eligible_candidates,
            key=lambda candidate: candidate.validation_score or float("-inf"),
        )
        finalized: list[ExperimentCandidate] = []
        for candidate in evaluated:
            if candidate.candidate_id == selected.candidate_id:
                finalized.append(candidate.model_copy(update={"status": "selected"}))
            else:
                if candidate.version_id:
                    version = self.version_service.get_version(slug, candidate.version_id)
                    if version.state.value == "candidate":
                        self.version_service.reject_version(slug, candidate.version_id)
                finalized.append(candidate)
        updated = session.model_copy(
            update={
                "status": ExperimentStatus.OPTIMIZED,
                "candidates": finalized,
                "selected_candidate_id": selected.candidate_id,
                "selected_version_id": selected.version_id,
                "updated_at": datetime.now(UTC),
            }
        )
        return self.store.update_session(updated)

    def validate_oos(
        self,
        slug: str,
        session_id: str,
        *,
        minimum_trades: int = 30,
        maximum_drawdown: float = 20.0,
        maximum_score_degradation: float = 5.0,
    ) -> ExperimentSession:
        """Reveal the locked test split exactly once for base and selected versions."""
        existing = self.get_session(slug, session_id)
        if existing.oos_revealed:
            raise ValidationServiceError("Out-of-sample data has already been revealed.")
        session = self._require_status(slug, session_id, ExperimentStatus.OPTIMIZED)
        if not session.selected_candidate_id or not session.selected_version_id:
            raise ValidationServiceError("Experiment has no selected optimization candidate.")
        started = session.model_copy(
            update={"oos_revealed": True, "updated_at": datetime.now(UTC)}
        )
        self.store.update_session(started)
        try:
            base_run = self._run_version_split(
                started,
                version_id=started.base_version_id,
                split="test",
                phase="test",
                candidate_id="baseline-oos",
            )
            started = started.model_copy(
                update={
                    "oos_base_run_id": base_run.run_id,
                    "updated_at": datetime.now(UTC),
                }
            )
            self.store.update_session(started)
            candidate_run = self._run_version_split(
                started,
                version_id=started.selected_version_id,
                split="test",
                phase="test",
                candidate_id=started.selected_candidate_id,
            )
        except Exception as exc:
            self._fail_session(started, str(exc))
            raise
        metrics = candidate_run.result["backtest"]
        passed = (
            int(metrics["total_trades"]) >= minimum_trades
            and float(metrics["max_drawdown"]) <= maximum_drawdown
            and candidate_run.score >= base_run.score - maximum_score_degradation
        )
        revealed = started.model_copy(
            update={
                "status": ExperimentStatus.OOS_VALIDATED,
                "oos_base_run_id": base_run.run_id,
                "oos_candidate_run_id": candidate_run.run_id,
                "oos_passed": passed,
                "updated_at": datetime.now(UTC),
            }
        )
        self.store.update_session(revealed)
        awaiting = revealed.model_copy(
            update={
                "status": ExperimentStatus.AWAITING_DECISION,
                "updated_at": datetime.now(UTC),
            }
        )
        return self.store.update_session(awaiting)

    def accept(self, slug: str, session_id: str, *, reason: str) -> ExperimentSession:
        """Promote a successful selected candidate after explicit human approval."""
        session = self._require_status(
            slug, session_id, ExperimentStatus.AWAITING_DECISION
        )
        reason = reason.strip()
        if not reason:
            raise ValidationServiceError("Acceptance reason is required.")
        if session.oos_passed is not True:
            raise ValidationServiceError("Out-of-sample validation did not pass.")
        if not session.selected_version_id:
            raise ValidationServiceError("Experiment has no selected candidate version.")
        strategy = self.workspace.get_strategy(slug)
        base = self.version_service.get_version(slug, session.base_version_id)
        if (
            strategy is None
            or strategy.active_version_id != base.version_id
            or strategy.current_version_id != base.version_id
            or self.workspace.compute_strategy_digest(slug) != base.content_digest
        ):
            raise ValidationServiceError(
                "Strategy changed after the experiment started; create a new experiment."
            )

        accepted_version = self.version_service.accept_version(
            slug, session.selected_version_id
        )
        updated = session.model_copy(
            update={
                "status": ExperimentStatus.ACCEPTED,
                "decision": "accepted",
                "decision_reason": reason,
                "accepted_version_id": accepted_version.version_id,
                "updated_at": datetime.now(UTC),
            }
        )
        self.store.update_session(updated)
        self.store.create_version_event(
            VersionEvent(
                event_id=uuid4().hex,
                strategy_slug=slug,
                action="accept",
                from_version_id=base.version_id,
                to_version_id=accepted_version.version_id,
                session_id=session.session_id,
                reason=reason,
            )
        )
        return updated

    def reject(self, slug: str, session_id: str, *, reason: str) -> ExperimentSession:
        """Reject the selected candidate without changing the active workspace."""
        session = self._require_status(
            slug, session_id, ExperimentStatus.AWAITING_DECISION
        )
        reason = reason.strip()
        if not reason:
            raise ValidationServiceError("Rejection reason is required.")
        if not session.selected_version_id:
            raise ValidationServiceError("Experiment has no selected candidate version.")
        candidate = self.version_service.get_version(slug, session.selected_version_id)
        if candidate.state == StrategyVersionState.CANDIDATE:
            self.version_service.reject_version(slug, candidate.version_id)
        updated = session.model_copy(
            update={
                "status": ExperimentStatus.REJECTED,
                "decision": "rejected",
                "decision_reason": reason,
                "updated_at": datetime.now(UTC),
            }
        )
        self.store.update_session(updated)
        self.store.create_version_event(
            VersionEvent(
                event_id=uuid4().hex,
                strategy_slug=slug,
                action="reject",
                from_version_id=candidate.version_id,
                to_version_id=session.base_version_id,
                session_id=session.session_id,
                reason=reason,
            )
        )
        return updated

    def rollback(self, slug: str, version_id: str, *, reason: str):
        """Restore an accepted ancestor and persist a rollback audit event."""
        reason = reason.strip()
        if not reason:
            raise ValidationServiceError("Rollback reason is required.")
        strategy = self.workspace.get_strategy(slug)
        if strategy is None or not strategy.active_version_id:
            raise ValidationServiceError("Strategy has no active immutable version.")
        current = self.version_service.get_version(slug, strategy.active_version_id)
        target = self.version_service.get_version(slug, version_id)
        if target.state != StrategyVersionState.ACCEPTED:
            raise ValidationServiceError("Rollback target must be an accepted version.")
        ancestor_ids: set[str] = set()
        cursor = current
        while cursor.parent_version_id:
            ancestor_ids.add(cursor.parent_version_id)
            cursor = self.version_service.get_version(slug, cursor.parent_version_id)
        if target.version_id not in ancestor_ids:
            raise ValidationServiceError("Rollback target must be an accepted ancestor version.")
        restored = self.version_service.rollback(slug, target.version_id)
        self.store.create_version_event(
            VersionEvent(
                event_id=uuid4().hex,
                strategy_slug=slug,
                action="rollback",
                from_version_id=current.version_id,
                to_version_id=target.version_id,
                reason=reason,
            )
        )
        return restored

    def _run_version_split(
        self,
        session: ExperimentSession,
        *,
        version_id: str,
        split: str,
        phase: str,
        candidate_id: str | None = None,
    ) -> BacktestRunRecord:
        version = self.version_service.get_version(session.strategy_slug, version_id)
        with TemporaryDirectory(prefix="autostrategy-experiment-") as temp_dir:
            strategy_dir = Path(temp_dir) / session.strategy_slug
            self.dataset_service.materialize_split(
                session.strategy_slug,
                session.manifest_id,
                version_id,
                split,
                strategy_dir,
            )
            result = run_backtest_workflow(strategy_dir)
        if "error" in result:
            raise ValidationServiceError(str(result["error"]))
        return self.run_store.record_backtest(
            strategy_slug=session.strategy_slug,
            strategy_version=version.version,
            strategy_digest=version.content_digest,
            score=float(result["score"]),
            result=result,
            version_id=version.version_id,
            manifest_id=session.manifest_id,
            session_id=session.session_id,
            phase=phase,
            candidate_id=candidate_id,
        )

    def _require_status(
        self,
        slug: str,
        session_id: str,
        expected: ExperimentStatus,
    ) -> ExperimentSession:
        session = self.get_session(slug, session_id)
        if session.status != expected:
            raise ValidationServiceError(
                f"Experiment step requires status {expected.value}; current status is "
                f"{session.status.value}."
            )
        return session

    def _fail_session(self, session: ExperimentSession, error: str) -> ExperimentSession:
        failed = session.model_copy(
            update={
                "status": ExperimentStatus.FAILED,
                "error": error,
                "updated_at": datetime.now(UTC),
            }
        )
        return self.store.update_session(failed)


def _build_findings(result: dict) -> list[DiagnosticFinding]:
    findings: list[DiagnosticFinding] = []
    backtest = result.get("backtest") if isinstance(result.get("backtest"), dict) else {}
    quality = (
        result.get("research_quality")
        if isinstance(result.get("research_quality"), dict)
        else {}
    )
    evidence_checks = {
        "has_equity_curve": (
            "evidence.equity_curve_missing",
            "缺少净值曲线，收益与回撤路径无法复核",
            ["让策略返回逐日 equity_curve"],
        ),
        "has_trade_records": (
            "evidence.trade_records_missing",
            "缺少逐笔成交记录，成本和信号无法审计",
            ["让策略返回标准 trades 列表"],
        ),
        "has_benchmark": (
            "evidence.benchmark_missing",
            "回测结果没有真实基准序列，无法判断超额收益",
            ["在相同日期区间计算基准收益"],
        ),
    }
    for flag, (code, hypothesis, actions) in evidence_checks.items():
        if not quality.get(flag, False):
            findings.append(
                DiagnosticFinding(
                    code=code,
                    category="robustness",
                    severity="warning",
                    evidence={flag: False},
                    hypothesis=hypothesis,
                    suggested_actions=actions,
                    auto_fixable=False,
                )
            )
    total_trades = int(backtest.get("total_trades", 0) or 0)
    if total_trades < 30:
        findings.append(
            DiagnosticFinding(
                code="sample.insufficient",
                category="robustness",
                severity="warning",
                evidence={"total_trades": total_trades, "minimum": 30},
                hypothesis="交易样本不足导致评分和风险指标不稳定",
                suggested_actions=["延长训练区间", "减少过严的信号过滤"],
                auto_fixable=False,
            )
        )
    max_drawdown = float(backtest.get("max_drawdown", 0) or 0)
    if max_drawdown > 20:
        findings.append(
            DiagnosticFinding(
                code="risk.drawdown_exceeded",
                category="risk",
                severity="critical",
                evidence={"max_drawdown": max_drawdown, "maximum": 20},
                hypothesis="当前风控或仓位配置无法约束回撤",
                suggested_actions=["降低单标的仓位", "收紧止损阈值"],
                auto_fixable=True,
            )
        )
    for diagnostic in result.get("diagnostics", []):
        if not isinstance(diagnostic, dict) or diagnostic.get("status") not in {"⚠️", "❌"}:
            continue
        critical = diagnostic.get("status") == "❌"
        findings.append(
            DiagnosticFinding(
                code=f"engine.{diagnostic.get('item', 'unknown')}",
                category="leakage" if diagnostic.get("item") == "未来函数" else "robustness",
                severity="critical" if critical else "warning",
                evidence=diagnostic,
                hypothesis=str(diagnostic.get("detail") or "回测引擎发现异常"),
                suggested_actions=["检查策略实现和数据口径"],
                auto_fixable=False,
            )
        )
    return findings


def _load_version_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(config, dict):
        raise ValidationServiceError("config.yaml must contain a mapping.")
    return config


def _config_leaf_paths(value: dict, prefix: str = "") -> list[str]:
    paths: list[str] = []
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            paths.extend(_config_leaf_paths(item, path))
        else:
            paths.append(path)
    return paths
