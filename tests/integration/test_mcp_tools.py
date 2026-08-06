"""MCP tool adapter tests."""

from autostrategy.mcp import tools


def test_mcp_tools_list_and_get_strategy(tmp_path):
    created = tools.create_strategy("demo", template="dual-ma", workspace_root=str(tmp_path))
    listed = tools.list_strategies(workspace_root=str(tmp_path))
    detail = tools.get_strategy("demo", workspace_root=str(tmp_path))

    assert created["ok"]
    assert listed["ok"]
    assert listed["data"][0]["slug"] == "demo"
    assert detail["ok"]
    assert detail["data"]["strategy"]["slug"] == "demo"


def test_mcp_tools_missing_strategy_returns_structured_error(tmp_path):
    result = tools.get_strategy("missing", workspace_root=str(tmp_path))

    assert not result["ok"]
    assert result["error"]["code"] == "strategy_not_found"


def test_mcp_tools_list_templates(tmp_path):
    result = tools.list_templates(workspace_root=str(tmp_path))

    assert result["ok"]
    assert "dual-ma" in result["data"]
