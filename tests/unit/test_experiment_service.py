"""Tests for resumable baseline and diagnostic experiment steps."""

from datetime import date

import pytest

from autostrategy.core.research import DateRange, ExperimentStatus
from autostrategy.core.workspace import Workspace
from autostrategy.services.dataset_manifest_service import DatasetManifestService
from autostrategy.services.exceptions import ValidationServiceError
from autostrategy.services.experiment_service import ExperimentService
from autostrategy.services.version_service import VersionService


def _create_research_inputs(tmp_path):
    workspace = Workspace(root=tmp_path)
    strategy = workspace.create_strategy("experiment-demo")
    workspace.write_text_file(
        strategy.slug,
        "config.yaml",
        "market: A股\n"
        "symbols:\n  - 510300.SH\n"
        "alpha: 0.0\ncommission: 0.0003\nslippage: 0.001\n",
    )
    workspace.write_text_file(
        strategy.slug,
        "data/fetch_data.py",
        "import pandas as pd\n\n"
        "def fetch(config):\n"
        "    dates = pd.date_range('2020-01-01', '2025-12-31', freq='D')\n"
        "    frame = pd.DataFrame({\n"
        "        'open': 1.0, 'high': 1.0, 'low': 1.0,\n"
        "        'close': 1.0, 'volume': 1000.0,\n"
        "    }, index=dates)\n"
        "    frame.index.name = 'date'\n"
        "    return frame\n",
    )
    workspace.write_text_file(
        strategy.slug,
        "strategy.py",
        "def run_backtest(config):\n"
        "    alpha = float(config.get('alpha', 0))\n"
        "    return {\n"
        "        'annual_return': 10.0 + alpha, 'max_drawdown': 8.0,\n"
        "        'sharpe': 1.2 + alpha / 20, 'win_rate': 52.0,\n"
        "        'profit_loss_ratio': 1.6, 'total_trades': 30,\n"
        "    }\n",
    )
    workspace.refresh_strategy_digest(strategy.slug)
    version = VersionService(workspace_root=tmp_path).ensure_current_version(strategy.slug)
    manifest = DatasetManifestService(workspace_root=tmp_path).capture(
        strategy.slug,
        version.version_id,
        benchmark="000300.SH",
        data_source="fixture",
        train=DateRange(start=date(2020, 1, 1), end=date(2022, 12, 31)),
        validation=DateRange(start=date(2023, 1, 1), end=date(2024, 12, 31)),
        test=DateRange(start=date(2025, 1, 1), end=date(2025, 12, 31)),
    )
    return strategy.slug, version, manifest


def test_experiment_session_persists_across_service_restart(tmp_path):
    slug, version, manifest = _create_research_inputs(tmp_path)
    service = ExperimentService(workspace_root=tmp_path)

    session = service.create_session(slug, version.version_id, manifest.manifest_id)
    restarted = ExperimentService(workspace_root=tmp_path)

    assert session.status == ExperimentStatus.CREATED
    assert restarted.get_session(slug, session.session_id) == session
    assert restarted.list_sessions(slug) == [session]


def test_baseline_runs_train_and_validation_with_full_run_provenance(tmp_path):
    slug, version, manifest = _create_research_inputs(tmp_path)
    service = ExperimentService(workspace_root=tmp_path)
    session = service.create_session(slug, version.version_id, manifest.manifest_id)

    completed = service.run_baseline(slug, session.session_id)
    train_run = service.run_store.get_backtest_run(slug, completed.baseline_train_run_id)
    validation_run = service.run_store.get_backtest_run(
        slug, completed.baseline_validation_run_id
    )

    assert completed.status == ExperimentStatus.BASELINE_COMPLETED
    assert train_run is not None
    assert train_run.session_id == session.session_id
    assert train_run.manifest_id == manifest.manifest_id
    assert train_run.version_id == version.version_id
    assert train_run.phase == "train"
    assert validation_run is not None
    assert validation_run.phase == "validation"


def test_diagnose_converts_quality_gaps_to_structured_findings(tmp_path):
    slug, version, manifest = _create_research_inputs(tmp_path)
    service = ExperimentService(workspace_root=tmp_path)
    session = service.create_session(slug, version.version_id, manifest.manifest_id)
    baseline = service.run_baseline(slug, session.session_id)

    diagnosed = service.diagnose(slug, baseline.session_id)

    assert diagnosed.status == ExperimentStatus.DIAGNOSED
    assert diagnosed.diagnostics
    assert all(finding.code and finding.hypothesis for finding in diagnosed.diagnostics)
    codes = {finding.code for finding in diagnosed.diagnostics}
    assert "evidence.equity_curve_missing" in codes
    assert "evidence.benchmark_missing" in codes


def test_experiment_rejects_skipped_and_repeated_steps(tmp_path):
    slug, version, manifest = _create_research_inputs(tmp_path)
    service = ExperimentService(workspace_root=tmp_path)
    session = service.create_session(slug, version.version_id, manifest.manifest_id)

    with pytest.raises(ValidationServiceError, match="requires status baseline_completed"):
        service.diagnose(slug, session.session_id)

    completed = service.run_baseline(slug, session.session_id)
    with pytest.raises(ValidationServiceError, match="requires status created"):
        service.run_baseline(slug, completed.session_id)
