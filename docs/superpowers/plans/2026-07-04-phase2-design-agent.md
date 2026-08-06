# Phase 2: AI 策略设计 Agent 产品化实现计划

**Goal:** 将现有 Skill 的设计 Agent 能力封装为产品级 `DesignAgent`，支持 CLI 通过自然语言生成策略设计文档。

**Architecture:** 用 `LLMClient` 统一封装多 provider 调用，`DesignAgent` 负责 prompt 组装与后处理，`QualityCheck` 校验输出结构，CLI 提供 `autostrategy design` 命令。

**Tech Stack:** Python 3.11+, Pydantic, Typer, OpenAI SDK (兼容多 provider), pytest

---

## Task 1: 实现 LLM Client

**Files:**
- Create: `src/autostrategy/llm/client.py`
- Create: `tests/unit/test_llm_client.py`

- [ ] **Step 1: 写失败测试**

```python
"""Tests for LLM client."""

from autostrategy.config import LLMConfig
from autostrategy.llm.client import LLMClient


def test_client_initialization():
    config = LLMConfig(provider="openai", model="gpt-4o-mini")
    client = LLMClient(config)
    assert client.config == config


def test_resolve_api_key_from_env(monkeypatch):
    monkeypatch.setenv("AUTOSTRATEGY_LLM_API_KEY", "test-key")
    config = LLMConfig(api_key_env="AUTOSTRATEGY_LLM_API_KEY")
    client = LLMClient(config)
    assert client.api_key == "test-key"


def test_resolve_api_key_missing():
    config = LLMConfig(api_key_env="NON_EXISTENT_KEY")
    client = LLMClient(config)
    assert client.api_key is None
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/unit/test_llm_client.py -v
```

Expected: 3 FAIL

- [ ] **Step 3: 实现 LLMClient**

```python
"""Unified LLM client for autostrategy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from autostrategy.config import LLMConfig


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str
    content: str


class LLMClient:
    """Client for calling LLM providers."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.api_key = self._resolve_api_key()

    def _resolve_api_key(self) -> str | None:
        """Resolve API key from environment or known provider variables."""
        env_vars = [
            self.config.api_key_env,
            "AUTOSTRATEGY_LLM_API_KEY",
            f"{self.config.provider.upper()}_API_KEY",
            "OPENAI_API_KEY",
        ]
        for var in env_vars:
            value = os.environ.get(var)
            if value:
                return value
        return None

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Send a chat request and return the content string."""
        if self.config.provider == "openai" or self.config.base_url:
            return self._chat_openai_compatible(messages, **kwargs)
        raise NotImplementedError(f"Provider '{self.config.provider}' is not supported yet.")

    def _chat_openai_compatible(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        """Call an OpenAI-compatible API."""
        try:
            import openai
        except ImportError as e:
            raise RuntimeError(
                "The 'openai' package is required for LLM calls. "
                "Install it with: pip install openai"
            ) from e

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.config.base_url,
        )
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=kwargs.get("temperature", self.config.temperature),
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("LLM returned empty content.")
        return content
```

- [ ] **Step 4: 更新 pyproject.toml 添加可选依赖**

在 `[project.optional-dependencies]` 中添加：

```toml
llm = [
    "openai>=1.30.0",
]
```

并将 `dev` 改为：

```toml
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
    "openai>=1.30.0",
]
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/unit/test_llm_client.py -v
```

Expected: 3 PASS

- [ ] **Step 6: 提交**

```bash
git add src/autostrategy/llm/client.py tests/unit/test_llm_client.py pyproject.toml
git commit -m "feat: 添加统一 LLM Client

- 支持 OpenAI-compatible provider
- 自动从环境变量解析 API key
- 添加单元测试

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: 实现 Quality Check

**Files:**
- Create: `src/autostrategy/core/quality_check.py`
- Create: `tests/unit/test_quality_check.py`

- [ ] **Step 1: 写失败测试**

```python
"""Tests for design document quality check."""

from autostrategy.core.quality_check import DesignQualityCheck, QualityReport


def test_valid_design_passes():
    design = """
# 双均线策略

## 策略概述

基于均线交叉。

## 买入条件

- 金叉

## 卖出条件

- 死叉

## 止损

- 5% 止损

## 仓位管理

- 满仓
"""
    report = DesignQualityCheck.check(design)
    assert report.passed
    assert not report.errors


def test_missing_sections_fails():
    design = "# 策略\n\n## 策略概述\n\n"
    report = DesignQualityCheck.check(design)
    assert not report.passed
    assert any("买入条件" in error for error in report.errors)
    assert any("卖出条件" in error for error in report.errors)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/unit/test_quality_check.py -v
```

- [ ] **Step 3: 实现 QualityCheck**

```python
"""Quality checks for STRATEGY_DESIGN.md."""

from __future__ import annotations

from dataclasses import dataclass, field


REQUIRED_SECTIONS = [
    "策略概述",
    "买入条件",
    "卖出条件",
    "止损",
    "仓位管理",
]


@dataclass
class QualityReport:
    """Result of a design document quality check."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DesignQualityCheck:
    """Check whether a STRATEGY_DESIGN.md meets the required structure."""

    @classmethod
    def check(cls, design_text: str) -> QualityReport:
        """Run all checks and return a report."""
        errors = []
        warnings = []

        for section in REQUIRED_SECTIONS:
            if f"## {section}" not in design_text:
                errors.append(f"Missing required section: {section}")

        if len(design_text.strip()) < 50:
            errors.append("Design document is too short.")

        if "未来函数" in design_text:
            warnings.append("Document mentions future functions — review carefully.")

        return QualityReport(passed=len(errors) == 0, errors=errors, warnings=warnings)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/unit/test_quality_check.py -v
```

- [ ] **Step 5: 提交**

```bash
git add src/autostrategy/core/quality_check.py tests/unit/test_quality_check.py
git commit -m "feat: 添加设计文档质量检查

- 校验必填章节：策略概述、买入条件、卖出条件、止损、仓位管理
- 返回 QualityReport 对象
- 添加单元测试

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: 实现 DesignAgent

**Files:**
- Create: `src/autostrategy/agents/design_agent.py`
- Create: `tests/integration/test_design_agent.py`
- Copy/Migrate: `.claude/skills/autostrategy/prompts/design_agent.md` 内容

- [ ] **Step 1: 迁移 prompt 内容**

将 `.claude/skills/autostrategy/prompts/design_agent.md` 的核心要求提取到 `src/autostrategy/agents/prompts/design.py`：

```python
"""System prompt for the design agent."""

DESIGN_SYSTEM_PROMPT = """
You are an expert quantitative strategy designer.
Your task is to translate a user's trading idea into a precise strategy design document.

Output must be in Chinese and follow this exact structure:

# <策略名称>

## 策略概述
- 一句话描述策略逻辑
- 适用市场与周期

## 买入条件
- 逐条列出买入信号条件
- 每个条件必须可量化

## 卖出条件
- 逐条列出卖出信号条件
- 包含止盈止损规则

## 止损
- 明确止损触发条件与执行方式

## 仓位管理
- 每次建仓比例、加减仓规则

## 风控规则
- 单日最大亏损、最大持仓数等

## 已知局限
- 列出策略可能失效的市场状态

Rules:
1. Only use indicators and data sources available in the user's market.
2. Do not use future data.
3. Keep conditions simple; fewer than 10 total signal conditions.
4. Every number must have a specific value or formula.
"""
```

- [ ] **Step 2: 实现 DesignAgent**

```python
"""Design Agent: generate STRATEGY_DESIGN.md from natural language."""

from __future__ import annotations

import re
from pathlib import Path

from autostrategy.agents.prompts.design import DESIGN_SYSTEM_PROMPT
from autostrategy.config import LLMConfig
from autostrategy.core.quality_check import DesignQualityCheck
from autostrategy.core.strategy import Strategy
from autostrategy.core.template_registry import TemplateRegistry
from autostrategy.core.workspace import Workspace
from autostrategy.llm.client import ChatMessage, LLMClient


class DesignAgent:
    """Agent that generates strategy design documents."""

    def __init__(self, llm_config: LLMConfig | None = None) -> None:
        self.llm_config = llm_config or LLMConfig()
        self.llm_client = LLMClient(self.llm_config)

    def design(
        self,
        prompt: str,
        market: str = "A股",
        template: str | None = None,
    ) -> str:
        """Generate a STRATEGY_DESIGN.md string from a user prompt."""
        messages = [
            ChatMessage(role="system", content=DESIGN_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=self._build_user_prompt(prompt, market, template),
            ),
        ]
        design_text = self.llm_client.chat(messages)
        return self._normalize(design_text)

    def _build_user_prompt(
        self,
        prompt: str,
        market: str,
        template: str | None,
    ) -> str:
        parts = [
            f"目标市场：{market}",
            f"用户需求：{prompt}",
        ]
        if template:
            template_names = TemplateRegistry.list_templates()
            if template in template_names:
                parts.append(f"参考模板：{template}")
        return "\n".join(parts)

    @staticmethod
    def _normalize(text: str) -> str:
        """Clean up LLM output."""
        text = text.strip()
        if text.startswith("```markdown"):
            text = text[len("```markdown"):]
        if text.startswith("```"):
            text = text[len("```"):]
        if text.endswith("```"):
            text = text[:-len("```")]
        return text.strip()

    def design_and_save(
        self,
        workspace: Workspace,
        name: str,
        prompt: str,
        market: str = "A股",
        template: str | None = None,
    ) -> Strategy:
        """Create a strategy workspace, generate design, and save."""
        strategy = workspace.create_strategy(name, market=market, template=template)
        design_text = self.design(prompt, market=market, template=template)

        report = DesignQualityCheck.check(design_text)
        if not report.passed:
            workspace.delete_strategy(strategy.slug)
            raise ValueError(
                f"Generated design failed quality check: {'; '.join(report.errors)}"
            )

        design_path = workspace._strategy_dir(strategy.slug) / "STRATEGY_DESIGN.md"
        design_path.write_text(design_text, encoding="utf-8")
        workspace.update_strategy_status(strategy.slug, Strategy.Status.DESIGNED)
        return workspace.get_strategy(strategy.slug)
```

- [ ] **Step 3: 创建 prompts 包**

```bash
mkdir -p src/autostrategy/agents/prompts
touch src/autostrategy/agents/__init__.py
touch src/autostrategy/agents/prompts/__init__.py
```

- [ ] **Step 4: 写集成测试**

```python
"""Integration tests for DesignAgent."""

import pytest

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
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/integration/test_design_agent.py -v
```

- [ ] **Step 6: 提交**

```bash
git add src/autostrategy/agents/ tests/integration/test_design_agent.py
git commit -m "feat: 添加 DesignAgent

- 封装 LLM 调用生成 STRATEGY_DESIGN.md
- 集成质量检查，不通过则回滚工作区
- 添加集成测试

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: CLI 添加 design 命令

**Files:**
- Modify: `src/autostrategy/cli/main.py`
- Create: `tests/integration/test_design_cli.py`

- [ ] **Step 1: 写失败测试**

```python
"""Integration tests for design CLI."""

from typer.testing import CliRunner

from autostrategy.cli.main import app

runner = CliRunner()


def test_design_command_requires_prompt():
    result = runner.invoke(app, ["design"])
    assert result.exit_code != 0
    assert "prompt" in result.stdout.lower() or "Missing" in result.stdout
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/integration/test_design_cli.py -v
```

- [ ] **Step 3: 在 CLI 添加 design 命令**

在 `src/autostrategy/cli/main.py` 中添加：

```python
from autostrategy.agents.design_agent import DesignAgent


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


design_app = typer.Typer(help="Design a trading strategy with AI.")
app.add_typer(design_app, name="design")


@design_app.command("create")
def design_create(
    prompt: str = typer.Option(..., "--prompt", "-p", help="Strategy idea in natural language."),
    name: str = typer.Option(..., "--name", "-n", help="Strategy name."),
    market: str = typer.Option("A股", "--market", "-m", help="Target market."),
    template: str | None = typer.Option(None, "--template", "-t", help="Template name."),
    workspace_root: str | None = typer.Option(None, "--workspace-root", help="Workspace root directory."),
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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/integration/test_design_cli.py -v
```

- [ ] **Step 5: 提交**

```bash
git add src/autostrategy/cli/main.py tests/integration/test_design_cli.py
git commit -m "feat: CLI 支持 design 命令

- autostrategy design create -p/--prompt -n/--name
- 自动生成 STRATEGY_DESIGN.md 并通过质量检查
- 添加集成测试

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: 验证 Phase 2

- [ ] **Step 1: 运行全部测试**

```bash
pytest -v
```

Expected: 全部通过

- [ ] **Step 2: 手动验证 CLI**

设置一个测试 API key（如果可用）：

```bash
export AUTOSTRATEGY_LLM_API_KEY=your-key
autostrategy config set llm.provider deepseek
autostrategy config set llm.model deepseek-chat
autostrategy design create -p "帮我做一个 A 股双均线策略" -n dual-ma-test --template dual-ma
autostrategy strategy show dual-ma-test
```

- [ ] **Step 3: 提交验证**

```bash
git add -A
git commit -m "chore: 验证 Phase 2 AI 策略设计 Agent

- 全部测试通过
- CLI design 命令端到端可用

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review

- Spec coverage: Phase 2 设计 Agent 产品化全部覆盖。
- Placeholder scan: 无 TBD/TODO，代码完整。
- Type consistency: `DesignAgent` 使用 `Workspace` 和 `Strategy` 已有接口一致。

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-07-04-phase2-design-agent.md`.**

Two execution options:
1. Subagent-Driven (recommended)
2. Inline Execution

Which approach?
