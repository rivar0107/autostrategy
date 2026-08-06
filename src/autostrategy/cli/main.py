"""CLI entry point for autostrategy."""

from pathlib import Path

import typer

from autostrategy import __version__
from autostrategy.agents.codegen_agent import CodegenAgent
from autostrategy.agents.design_agent import DesignAgent
from autostrategy.config import init_settings, load_settings, save_settings
from autostrategy.core.backtest_engine import run_backtest_workflow
from autostrategy.core.strategy import StrategyStatus
from autostrategy.core.template_registry import TemplateRegistry
from autostrategy.core.workspace import Workspace

app = typer.Typer(
    name="autostrategy",
    help="面向个人投资者的本地开源量化策略平台",
    no_args_is_help=True,
    invoke_without_command=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"autostrategy {__version__}")
        raise typer.Exit()


@app.callback()
def callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """autostrategy CLI."""


config_app = typer.Typer(help="Manage autostrategy configuration.")
app.add_typer(config_app, name="config")


@config_app.command("init")
def config_init() -> None:
    """Initialize user settings directory."""
    settings = init_settings()
    typer.echo("Settings initialized at ~/.autostrategy/settings.yaml")
    typer.echo(f"Default LLM provider: {settings.llm.provider}")


@config_app.command("show")
def config_show() -> None:
    """Show current settings."""
    settings = load_settings()
    typer.echo(f"Version: {settings.version}")
    typer.echo(f"LLM provider: {settings.llm.provider}")
    typer.echo(f"LLM model: {settings.llm.model}")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Setting key, e.g. llm.provider"),
    value: str = typer.Argument(..., help="Setting value"),
) -> None:
    """Set a configuration value."""
    settings = load_settings()
    parts = key.split(".")
    current = settings
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)
    save_settings(settings)
    typer.echo(f"Set {key} = {value}")


strategy_app = typer.Typer(help="Manage trading strategies.")
app.add_typer(strategy_app, name="strategy")


def _workspace_root_option(workspace_root: str | None) -> Path | None:
    return Path(workspace_root) if workspace_root else None


def _echo_strategy_paths(workspace: Workspace, slug: str) -> None:
    strategy_dir = workspace.get_strategy_dir(slug)
    typer.echo(f"Workspace: {strategy_dir}")
    typer.echo(f"Metadata: {strategy_dir / 'strategy.yaml'}")
    typer.echo(f"Design: {strategy_dir / 'STRATEGY_DESIGN.md'}")
    typer.echo(f"Strategy code: {strategy_dir / 'strategy.py'}")
    typer.echo(f"Config: {strategy_dir / 'config.yaml'}")
    typer.echo(f"README: {strategy_dir / 'README.md'}")
    typer.echo(f"Backtest result: {strategy_dir / 'backtest' / 'results' / 'backtest_result.json'}")


@strategy_app.command("list")
def strategy_list(
    workspace_root: str | None = typer.Option(
        None, "--workspace-root", help="Workspace root directory."
    ),
) -> None:
    """List all strategies in workspace."""
    workspace = Workspace(root=_workspace_root_option(workspace_root))
    strategies = workspace.list_strategies()
    if not strategies:
        typer.echo("No strategies yet. Use `autostrategy strategy create` to add one.")
        return
    typer.echo(f"{'SLUG':20} {'MARKET':8} {'STATUS':12} NAME")
    for strategy in strategies:
        typer.echo(
            f"{strategy.slug:20} {strategy.market:8} {strategy.status.value:12} {strategy.name}"
        )


@strategy_app.command("create")
def strategy_create(
    name: str = typer.Argument(..., help="Strategy name."),
    template: str | None = typer.Option(None, "--template", help="Template name to use."),
    market: str = typer.Option("A股", "--market", help="Target market."),
    workspace_root: str | None = typer.Option(
        None, "--workspace-root", help="Workspace root directory."
    ),
) -> None:
    """Create a new strategy workspace."""
    if template and template not in TemplateRegistry.list_templates():
        typer.echo(
            f"Template '{template}' not found. "
            f"Available: {', '.join(TemplateRegistry.list_templates())}"
        )
        raise typer.Exit(1)

    workspace = Workspace(root=_workspace_root_option(workspace_root))
    try:
        strategy = workspace.create_strategy(name, market=market, template=template)
        typer.echo(f"Strategy '{strategy.name}' created at {strategy.workspace_dir}")
    except FileExistsError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1)


@strategy_app.command("show")
def strategy_show(
    slug: str = typer.Argument(..., help="Strategy slug."),
    workspace_root: str | None = typer.Option(
        None, "--workspace-root", help="Workspace root directory."
    ),
) -> None:
    """Show strategy details."""
    workspace = Workspace(root=_workspace_root_option(workspace_root))
    strategy = workspace.get_strategy(slug)
    if not strategy:
        typer.echo(f"Strategy '{slug}' not found.")
        raise typer.Exit(1)
    typer.echo(f"Name: {strategy.name}")
    typer.echo(f"Slug: {strategy.slug}")
    typer.echo(f"Market: {strategy.market}")
    typer.echo(f"Status: {strategy.status.value}")
    typer.echo(f"Template: {strategy.template or 'none'}")
    _echo_strategy_paths(workspace, slug)


@strategy_app.command("paths")
def strategy_paths(
    slug: str = typer.Argument(..., help="Strategy slug."),
    workspace_root: str | None = typer.Option(
        None, "--workspace-root", help="Workspace root directory."
    ),
) -> None:
    """Show local file paths for a strategy workspace."""
    workspace = Workspace(root=_workspace_root_option(workspace_root))
    if not workspace.get_strategy(slug):
        typer.echo(f"Strategy '{slug}' not found.")
        raise typer.Exit(1)
    _echo_strategy_paths(workspace, slug)


@strategy_app.command("delete")
def strategy_delete(
    slug: str = typer.Argument(..., help="Strategy slug."),
    workspace_root: str | None = typer.Option(
        None, "--workspace-root", help="Workspace root directory."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Delete a strategy workspace."""
    workspace = Workspace(root=_workspace_root_option(workspace_root))
    strategy = workspace.get_strategy(slug)
    if not strategy:
        typer.echo(f"Strategy '{slug}' not found.")
        raise typer.Exit(1)
    if not yes:
        confirm = typer.confirm(f"Delete strategy '{strategy.name}'?")
        if not confirm:
            typer.echo("Cancelled.")
            raise typer.Exit(0)
    workspace.delete_strategy(slug)
    typer.echo(f"Strategy '{slug}' deleted.")


design_app = typer.Typer(help="Design a trading strategy with AI.")
app.add_typer(design_app, name="design")


@design_app.command("create")
def design_create(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Strategy idea in natural language."),
    name: str = typer.Option(..., "--name", "-n", help="Strategy name."),
    market: str = typer.Option("A股", "--market", "-m", help="Target market."),
    template: str | None = typer.Option(None, "--template", "-t", help="Template name."),
    workspace_root: str | None = typer.Option(
        None, "--workspace-root", help="Workspace root directory."
    ),
) -> None:
    """Generate a strategy design from a natural language prompt."""
    settings = load_settings()
    agent = DesignAgent(llm_config=settings.llm)
    workspace = Workspace(root=_workspace_root_option(workspace_root))
    try:
        strategy = agent.design_and_save(
            workspace=workspace,
            name=name,
            prompt=prompt,
            market=market,
            template=template,
        )
        typer.echo(f"Strategy '{strategy.name}' designed at {strategy.workspace_dir}")
        typer.echo(f"Design saved to {strategy.workspace_dir / 'STRATEGY_DESIGN.md'}")
    except ValueError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1)


codegen_app = typer.Typer(help="Generate executable strategy files with AI.")
app.add_typer(codegen_app, name="codegen")


@codegen_app.command("create")
def codegen_create(
    slug: str = typer.Argument(..., help="Strategy slug."),
    workspace_root: str | None = typer.Option(
        None, "--workspace-root", help="Workspace root directory."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing strategy.py."),
) -> None:
    """Generate strategy.py, config.yaml, README.md and support files."""
    settings = load_settings()
    workspace = Workspace(root=_workspace_root_option(workspace_root))
    agent = CodegenAgent(llm_config=settings.llm)
    try:
        strategy = agent.codegen_and_save(workspace=workspace, slug=slug, force=force)
    except (FileNotFoundError, ValueError, FileExistsError) as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1)
    typer.echo(f"Strategy '{strategy.slug}' coded at {strategy.workspace_dir}")
    typer.echo("Generated files: strategy.py, config.yaml, README.md, requirements.txt, data/fetch_data.py")


backtest_app = typer.Typer(help="Run strategy backtests.")
app.add_typer(backtest_app, name="backtest")


@backtest_app.command("run")
def backtest_run(
    slug: str = typer.Argument(..., help="Strategy slug."),
    workspace_root: str | None = typer.Option(
        None, "--workspace-root", help="Workspace root directory."
    ),
) -> None:
    """Run a standard backtest for a strategy."""
    workspace = Workspace(root=_workspace_root_option(workspace_root))
    strategy = workspace.get_strategy(slug)
    if strategy is None:
        typer.echo(f"Error: Strategy '{slug}' not found.")
        raise typer.Exit(1)

    result = run_backtest_workflow(workspace.get_strategy_dir(slug))
    if "error" in result:
        typer.echo(f"Error: {result['error']}")
        raise typer.Exit(1)

    workspace.update_strategy_status(slug, StrategyStatus.BACKTESTED)
    typer.echo(f"Backtest completed for '{slug}'.")
    typer.echo(f"Score: {result['score']}/100")
    typer.echo("Result saved to backtest/results/backtest_result.json")


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind."),
    port: int = typer.Option(8000, "--port", help="Port to bind."),
    reload: bool = typer.Option(False, "--reload", help="Enable uvicorn reload."),
    workspace_root: str | None = typer.Option(
        None, "--workspace-root", help="Workspace root directory."
    ),
) -> None:
    """Start the local REST API and dashboard."""
    try:
        import uvicorn
    except ImportError as exc:
        typer.echo("Error: uvicorn is required. Install with: pip install -e '.[api,web]'")
        raise typer.Exit(1) from exc

    if workspace_root:
        from autostrategy.api.app import create_app

        uvicorn.run(
            create_app(workspace_root=_workspace_root_option(workspace_root)),
            host=host,
            port=port,
            reload=False,
        )
        return

    uvicorn.run("autostrategy.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
