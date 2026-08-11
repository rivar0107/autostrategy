"""Tests for the safe configuration optimization ratchet."""

import yaml

from autostrategy.services.models import OptimizationCandidate
from autostrategy.services.optimization_service import OptimizationService
from autostrategy.services.strategy_service import StrategyService
from autostrategy.services.version_service import VersionService


def _create_tunable_strategy(tmp_path):
    service = StrategyService(workspace_root=tmp_path)
    strategy = service.create_strategy("tunable")
    strategy_dir = tmp_path / strategy.slug
    (strategy_dir / "config.yaml").write_text(
        "market: A股\nannual_return: 10.0\nrisk:\n  stop_loss: 5\n", encoding="utf-8"
    )
    (strategy_dir / "strategy.py").write_text(
        "def run_backtest(config):\n"
        "    return {\n"
        "        'annual_return': float(config['annual_return']),\n"
        "        'max_drawdown': float(config.get('risk', {}).get('stop_loss', 5)),\n"
        "        'sharpe': 1.2, 'win_rate': 52.0,\n"
        "        'profit_loss_ratio': 1.6, 'total_trades': 30,\n"
        "    }\n",
        encoding="utf-8",
    )
    service.workspace.refresh_strategy_digest(strategy.slug)
    return strategy_dir


def test_optimization_evaluates_candidates_without_mutating_source(tmp_path):
    strategy_dir = _create_tunable_strategy(tmp_path)
    original_config = (strategy_dir / "config.yaml").read_text(encoding="utf-8")
    service = OptimizationService(workspace_root=tmp_path)

    report = service.evaluate(
        "tunable",
        [
            OptimizationCandidate(name="worse", config_overrides={"annual_return": 8.0}),
            OptimizationCandidate(
                name="better",
                config_overrides={"annual_return": 16.0, "risk": {"stop_loss": 4}},
            ),
        ],
        minimum_improvement=1.0,
    )

    assert report.baseline_score > 0
    assert report.recommended_candidate == "better"
    assert [candidate.eligible for candidate in report.candidates] == [False, True]
    assert (strategy_dir / "config.yaml").read_text(encoding="utf-8") == original_config
    assert report.accepted is False
    assert report.base_version_id
    assert service.get_latest_report("tunable").report_id == report.report_id


def test_optimization_accept_requires_eligible_fresh_candidate(tmp_path):
    strategy_dir = _create_tunable_strategy(tmp_path)
    service = OptimizationService(workspace_root=tmp_path)
    report = service.evaluate(
        "tunable",
        [
            OptimizationCandidate(name="worse", config_overrides={"annual_return": 8.0}),
            OptimizationCandidate(name="better", config_overrides={"annual_return": 16.0}),
        ],
        minimum_improvement=1.0,
    )

    accepted = service.accept("tunable", report.report_id, "better")
    config = yaml.safe_load((strategy_dir / "config.yaml").read_text(encoding="utf-8"))
    strategy = service.strategy_service.workspace.get_strategy("tunable")

    assert accepted.accepted is True
    assert accepted.accepted_candidate == "better"
    assert config["annual_return"] == 16.0
    assert config["risk"]["stop_loss"] == 5
    assert strategy is not None
    assert strategy.version == 2
    assert strategy.status.value == "optimized"
    versions = VersionService(workspace_root=tmp_path).list_versions("tunable")
    assert len(versions) == 2
    base_version, accepted_version = versions
    assert report.base_version_id == base_version.version_id
    assert accepted.accepted_version_id == accepted_version.version_id
    assert accepted_version.parent_version_id == base_version.version_id
    assert strategy.current_version_id == accepted_version.version_id
    assert strategy.active_version_id == accepted_version.version_id


def test_optimization_invalid_candidate_does_not_corrupt_config(tmp_path):
    strategy_dir = _create_tunable_strategy(tmp_path)
    original_config = (strategy_dir / "config.yaml").read_text(encoding="utf-8")
    service = OptimizationService(workspace_root=tmp_path)

    report = service.evaluate(
        "tunable",
        [OptimizationCandidate(name="invalid", config_overrides={"annual_return": "bad"})],
    )

    assert report.candidates[0].status == "failed"
    assert report.recommended_candidate is None
    assert (strategy_dir / "config.yaml").read_text(encoding="utf-8") == original_config
