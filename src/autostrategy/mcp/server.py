"""Minimal MCP server registration for autostrategy."""

from __future__ import annotations

from autostrategy.mcp import tools


def create_server():
    """Create an MCP server with conservative local strategy tools."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("The 'mcp' package is required. Install with: pip install -e '.[mcp]'") from exc

    server = FastMCP("autostrategy")
    server.tool()(tools.list_strategies)
    server.tool()(tools.get_strategy)
    server.tool()(tools.get_strategy_paths)
    server.tool()(tools.list_templates)
    server.tool()(tools.get_backtest_result)
    server.tool()(tools.create_strategy)
    server.tool()(tools.run_backtest)
    return server


def main() -> None:
    """Run the stdio MCP server."""
    create_server().run()


if __name__ == "__main__":
    main()
