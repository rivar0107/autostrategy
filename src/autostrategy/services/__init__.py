"""Application service layer with cycle-safe lazy public exports."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BacktestJobService": "autostrategy.services.backtest_job_service",
    "BacktestService": "autostrategy.services.backtest_service",
    "CodegenService": "autostrategy.services.codegen_service",
    "ClientSimulationService": "autostrategy.services.client_simulation_service",
    "DatasetManifestService": "autostrategy.services.dataset_manifest_service",
    "DesignJobService": "autostrategy.services.design_job_service",
    "DesignService": "autostrategy.services.design_service",
    "ExperimentService": "autostrategy.services.experiment_service",
    "OptimizationService": "autostrategy.services.optimization_service",
    "PaperRunJobService": "autostrategy.services.paper_run_job_service",
    "PaperRunService": "autostrategy.services.paper_run_service",
    "StrategyService": "autostrategy.services.strategy_service",
    "VersionService": "autostrategy.services.version_service",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Load a public service only when requested to avoid package import cycles."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
