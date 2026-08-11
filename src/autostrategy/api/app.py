"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from autostrategy import __version__
from autostrategy.api.errors import register_error_handlers
from autostrategy.api.routers import (
    artifacts,
    backtest,
    client_simulation,
    codegen,
    config,
    design,
    health,
    optimization,
    paper_run,
    research,
    strategies,
)
from autostrategy.config import load_settings
from autostrategy.services.backtest_job_service import BacktestJobService
from autostrategy.services.client_simulation_service import (
    ClientSimulationService,
    FtshareTenMinuteMarketContextProvider,
)
from autostrategy.services.paper_run_job_service import PaperRunJobService


def create_app(workspace_root: Path | None = None) -> FastAPI:
    """Create the local autostrategy API and dashboard app."""
    app = FastAPI(
        title="autostrategy",
        description="本地开源量化策略 Agent 平台",
        version=__version__,
    )
    app.state.workspace_root = workspace_root
    app.state.backtest_job_service = BacktestJobService(workspace_root=workspace_root)
    app.state.paper_run_job_service = PaperRunJobService(workspace_root=workspace_root)
    app.state.client_simulation_service = ClientSimulationService(
        workspace_root=workspace_root,
        config=load_settings().broker_connections.ft_client,
        market_context_provider=FtshareTenMinuteMarketContextProvider(),
        enable_background_reconciliation=True,
        market_poll_interval_seconds=30.0,
    )
    app.router.on_shutdown.append(app.state.client_simulation_service.shutdown)

    register_error_handlers(app)
    api_prefix = "/api/v1"
    app.include_router(health.router, prefix=api_prefix)
    app.include_router(config.router, prefix=api_prefix)
    app.include_router(strategies.router, prefix=api_prefix)
    app.include_router(design.router, prefix=api_prefix)
    app.include_router(codegen.router, prefix=api_prefix)
    app.include_router(backtest.router, prefix=api_prefix)
    app.include_router(optimization.router, prefix=api_prefix)
    app.include_router(research.router, prefix=api_prefix)
    app.include_router(paper_run.router, prefix=api_prefix)
    app.include_router(client_simulation.router, prefix=api_prefix)
    app.include_router(artifacts.router, prefix=api_prefix)

    web_dir = Path(__file__).parent.parent / "web"
    static_dir = web_dir / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        """Serve the local dashboard shell."""
        return FileResponse(static_dir / "index.html")

    return app


app = create_app()
