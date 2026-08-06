"""Integration tests for DesignAgent."""

from autostrategy.agents.design_agent import DesignAgent
from autostrategy.config import LLMConfig
from autostrategy.core.workspace import Workspace


class FakeLLMClient:
    """Fake LLM client for testing without network."""

    def __init__(self, config):
        self.config = config

    def chat(self, messages, **kwargs):
        return """
# 双均线策略

## 策略概述

基于 5 日和 20 日均线交叉。

## 买入条件

- 5 日均线上穿 20 日均线

## 卖出条件

- 5 日均线下穿 20 日均线

## 止损

- 亏损 5% 止损

## 仓位管理

- 满仓买入

## 风控规则

- 单票最大仓位 100%

## 已知局限

- 震荡市失效
"""


def test_design_agent_generates_design(tmp_path, monkeypatch):
    agent = DesignAgent(llm_config=LLMConfig())
    monkeypatch.setattr(agent, "llm_client", FakeLLMClient(agent.llm_config))

    workspace = Workspace(root=tmp_path)
    strategy = agent.design_and_save(
        workspace=workspace,
        name="dual-ma",
        prompt="帮我做一个双均线策略",
        market="A股",
    )
    assert strategy.status == "designed"
    design_path = tmp_path / "dual-ma" / "STRATEGY_DESIGN.md"
    assert design_path.exists()
    content = design_path.read_text(encoding="utf-8")
    assert "买入条件" in content
    assert "卖出条件" in content
