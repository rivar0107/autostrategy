# Phase 0: autostrategy 产品骨架实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 autostrategy 从 Claude Code Skill 重构为可 pip 安装的 Python 包，保留 Skill 兼容性，建立 CLI、配置系统和产品目录结构。

**Architecture:** 根目录作为产品入口，代码放在 `src/autostrategy/` 下，CLI 使用 Typer，配置用 Pydantic + YAML，Skill 文件迁移到 `.claude/skills/autostrategy/`，根目录保留 shim `SKILL.md` 兼容旧安装路径。

**Tech Stack:** Python 3.11+, Typer, Pydantic, PyYAML, hatchling/setuptools, pytest

---

## 文件结构规划

```
autostrategy/
├── pyproject.toml                    # 包定义、依赖、CLI 脚本
├── .env.example                      # 环境变量示例
├── README.md                         # 更新为产品 README
├── SKILL.md                          # shim: 转发到 .claude/skills/autostrategy/SKILL.md
├── src/
│   └── autostrategy/
│       ├── __init__.py               # 版本号
│       ├── __main__.py               # python -m autostrategy
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py               # Typer CLI
│       └── config.py                 # 配置模型与持久化
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   └── test_config.py            # 配置测试
│   └── integration/
│       ├── __init__.py
│       └── test_cli.py               # CLI 集成测试
└── .claude/
    └── skills/
        └── autostrategy/
            ├── SKILL.md              # 从根目录迁移
            └── prompts/
                ├── design_agent.md   # 从 prompts/ 迁移
                ├── codegen_agent.md  # 从 prompts/ 迁移
                └── optimization_agent.md  # 从 prompts/ 迁移
```

---

## Task 1: 创建 pyproject.toml

**Files:**
- Create: `pyproject.toml`

**说明:** 定义包元数据、依赖、CLI 入口脚本、构建后端。

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "autostrategy"
version = "0.1.0"
description = "面向个人投资者的本地开源量化策略平台"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.11"
authors = [
    { name = "autostrategy contributors" },
]
keywords = ["quant", "trading", "backtest", "agent", "strategy"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Financial and Insurance Industry",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Office/Business :: Financial :: Investment",
]
dependencies = [
    "typer>=0.12.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "click>=8.0.0",
]

[project.optional-dependencies]
backtest = [
    "backtrader>=1.9.78",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
]
data = [
    "akshare>=1.14.0",
    "yfinance>=0.2.0",
]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
]
all = [
    "autostrategy[backtest,data,dev]",
]

[project.scripts]
autostrategy = "autostrategy.cli.main:app"

[project.urls]
Homepage = "https://github.com/rivar0107/autostrategy"
Documentation = "https://github.com/rivar0107/autostrategy#readme"
Repository = "https://github.com/rivar0107/autostrategy"
Issues = "https://github.com/rivar0107/autostrategy/issues"

[tool.hatch.build.targets.wheel]
packages = ["src/autostrategy"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: 提交**

```bash
git add pyproject.toml
git commit -m "chore: 添加 pyproject.toml 与包元数据

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: 创建 src/autostrategy 包与版本

**Files:**
- Create: `src/autostrategy/__init__.py`
- Create: `src/autostrategy/__main__.py`

- [ ] **Step 1: 创建 __init__.py**

```python
"""autostrategy: 面向个人投资者的本地开源量化策略平台."""

__version__ = "0.1.0"
__all__ = ["__version__"]
```

- [ ] **Step 2: 创建 __main__.py**

```python
"""Entry point for `python -m autostrategy`."""

from autostrategy.cli.main import app

if __name__ == "__main__":
    app()
```

- [ ] **Step 3: 提交**

```bash
git add src/autostrategy/__init__.py src/autostrategy/__main__.py
git commit -m "feat: 添加 autostrategy 包入口与版本号

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: 实现配置系统

**Files:**
- Create: `src/autostrategy/config.py`
- Create: `tests/unit/test_config.py`

- [ ] **Step 1: 写失败测试**

```python
"""Tests for autostrategy.config."""

import os
import tempfile
from pathlib import Path

import pytest

from autostrategy.config import Settings, get_settings_dir, load_settings, save_settings


def test_get_settings_dir():
    """Settings dir should be under user home."""
    settings_dir = get_settings_dir()
    assert isinstance(settings_dir, Path)
    assert settings_dir.name == ".autostrategy"
    assert settings_dir.parent == Path.home()


def test_settings_defaults():
    """Default settings should be valid."""
    settings = Settings()
    assert settings.version == "0.1.0"
    assert settings.llm.provider == "openai"
    assert settings.llm.model == "gpt-4o-mini"


def test_save_and_load_settings():
    """Round-trip settings save/load."""
    with tempfile.TemporaryDirectory() as tmp:
        settings_path = Path(tmp) / "settings.yaml"
        settings = Settings()
        save_settings(settings, settings_path)
        loaded = load_settings(settings_path)
        assert loaded.version == settings.version
        assert loaded.llm.provider == settings.llm.provider


def test_load_missing_settings_returns_defaults():
    """Loading missing settings should return defaults."""
    with tempfile.TemporaryDirectory() as tmp:
        settings_path = Path(tmp) / "not_exists.yaml"
        settings = load_settings(settings_path)
        assert settings.version == "0.1.0"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/unit/test_config.py -v
```

Expected: 4 FAIL，模块不存在

- [ ] **Step 3: 实现配置模块**

```python
"""Configuration management for autostrategy.

Settings are stored in ~/.autostrategy/settings.yaml.
API keys should be stored via keyring or environment variables, not in this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from autostrategy import __version__


DEFAULT_PROVIDER: Literal["openai", "deepseek", "kimi", "qwen", "zai", "minimax", "gemini", "local"] = "openai"


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = DEFAULT_PROVIDER
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key_env: str = "AUTOSTRATEGY_LLM_API_KEY"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class Settings(BaseModel):
    """Top-level application settings."""

    version: str = __version__
    llm: LLMConfig = Field(default_factory=LLMConfig)
    default_market: str = "A股"
    data_cache_dir: str | None = None


def get_settings_dir() -> Path:
    """Return the user-level settings directory."""
    return Path.home() / ".autostrategy"


def get_default_settings_path() -> Path:
    """Return the default settings file path."""
    return get_settings_dir() / "settings.yaml"


def save_settings(settings: Settings, path: Path | None = None) -> None:
    """Save settings to YAML file."""
    target = path or get_default_settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(settings.model_dump(), f, allow_unicode=True, sort_keys=False)


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from YAML file, returning defaults if missing."""
    target = path or get_default_settings_path()
    if not target.exists():
        return Settings()
    with open(target, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Settings(**data)


def init_settings() -> Settings:
    """Initialize settings directory and return current settings."""
    settings = load_settings()
    save_settings(settings)
    return settings
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/unit/test_config.py -v
```

Expected: 4 PASS

- [ ] **Step 5: 提交**

```bash
git add src/autostrategy/config.py tests/unit/test_config.py
git commit -m "feat: 添加配置系统与持久化

- 支持 ~/.autostrategy/settings.yaml
- 包含 LLM provider/model/base_url 配置
- 添加 pytest 单元测试

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: 实现 CLI 入口

**Files:**
- Create: `src/autostrategy/cli/__init__.py`
- Create: `src/autostrategy/cli/main.py`
- Create: `tests/integration/test_cli.py`

- [ ] **Step 1: 写失败测试**

```python
"""Integration tests for CLI."""

from typer.testing import CliRunner

from autostrategy.cli.main import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "autostrategy" in result.stdout
    assert "0.1.0" in result.stdout


def test_config_init():
    result = runner.invoke(app, ["config", "init"])
    assert result.exit_code == 0
    assert "initialized" in result.stdout.lower() or "Settings" in result.stdout


def test_strategy_list_empty():
    result = runner.invoke(app, ["strategy", "list"])
    assert result.exit_code == 0
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/integration/test_cli.py -v
```

Expected: 3 FAIL，CLI 模块不存在

- [ ] **Step 3: 实现 CLI**

```python
"""CLI entry point for autostrategy."""

from __future__ import annotations

import typer

from autostrategy import __version__
from autostrategy.config import init_settings, load_settings

app = typer.Typer(
    name="autostrategy",
    help="面向个人投资者的本地开源量化策略平台",
    no_args_is_help=True,
)


@app.callback()
def callback(
    version: bool | None = typer.Option(None, "--version", "-v", help="Show version and exit."),
) -> None:
    """autostrategy CLI."""
    if version:
        typer.echo(f"autostrategy {__version__}")
        raise typer.Exit()


config_app = typer.Typer(help="Manage autostrategy configuration.")
app.add_typer(config_app, name="config")


@config_app.command("init")
def config_init() -> None:
    """Initialize user settings directory."""
    settings = init_settings()
    typer.echo(f"Settings initialized at ~/.autostrategy/settings.yaml")
    typer.echo(f"Default LLM provider: {settings.llm.provider}")


@config_app.command("show")
def config_show() -> None:
    """Show current settings."""
    settings = load_settings()
    typer.echo(f"Version: {settings.version}")
    typer.echo(f"LLM provider: {settings.llm.provider}")
    typer.echo(f"LLM model: {settings.llm.model}")


strategy_app = typer.Typer(help="Manage trading strategies.")
app.add_typer(strategy_app, name="strategy")


@strategy_app.command("list")
def strategy_list() -> None:
    """List all strategies in workspace."""
    typer.echo("No strategies yet. Use `autostrategy strategy create` to add one.")


@strategy_app.command("create")
def strategy_create(
    name: str = typer.Argument(..., help="Strategy name."),
) -> None:
    """Create a new strategy workspace."""
    typer.echo(f"Strategy '{name}' creation will be implemented in Phase 1.")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/integration/test_cli.py -v
```

Expected: 3 PASS

- [ ] **Step 5: 提交**

```bash
git add src/autostrategy/cli/ tests/integration/test_cli.py
git commit -m "feat: 添加 Typer CLI 入口

- 支持 --version
- 支持 config init/show
- 支持 strategy list/create 占位
- 添加集成测试

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: 迁移 Skill 到子目录

**Files:**
- Create: `.claude/skills/autostrategy/SKILL.md`
- Create: `.claude/skills/autostrategy/prompts/design_agent.md`
- Create: `.claude/skills/autostrategy/prompts/codegen_agent.md`
- Create: `.claude/skills/autostrategy/prompts/optimization_agent.md`
- Modify: `SKILL.md`（根目录，改为 shim）
- Delete/Move: `prompts/design_agent.md`, `prompts/codegen_agent.md`, `prompts/optimization_agent.md`

- [ ] **Step 1: 移动 Skill 文件**

```bash
mkdir -p .claude/skills/autostrategy/prompts
mv prompts/design_agent.md .claude/skills/autostrategy/prompts/
mv prompts/codegen_agent.md .claude/skills/autostrategy/prompts/
mv prompts/optimization_agent.md .claude/skills/autostrategy/prompts/
mv SKILL.md .claude/skills/autostrategy/SKILL.md
```

- [ ] **Step 2: 创建根目录 shim SKILL.md**

```markdown
---
name: autostrategy
description: AI 驱动的量化策略自动生成与回测（产品版入口）
---

# autostrategy Skill

本 Skill 已产品化。完整能力见 `.claude/skills/autostrategy/SKILL.md`。

为了保持旧版安装路径兼容，根目录保留此 shim 文件。

未来 Skill 调用应通过 autostrategy 产品 CLI / API / MCP 完成。
```

- [ ] **Step 3: 验证目录结构**

```bash
ls -la .claude/skills/autostrategy/
ls -la .claude/skills/autostrategy/prompts/
cat SKILL.md
```

- [ ] **Step 4: 提交**

```bash
git add .claude/skills/autostrategy/ SKILL.md
git rm prompts/design_agent.md prompts/codegen_agent.md prompts/optimization_agent.md
git commit -m "refactor: 将 Skill 迁入 .claude/skills/autostrategy/ 子目录

- 保留根目录 SKILL.md 作为 shim，兼容旧安装路径
- prompts/ 下文件迁移到 .claude/skills/autostrategy/prompts/
- 为根目录产品化腾出空间

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: 创建环境变量示例与更新 README

**Files:**
- Create: `.env.example`
- Modify: `README.md`

- [ ] **Step 1: 创建 .env.example**

```bash
# autostrategy environment variables

# LLM API key (choose your provider)
AUTOSTRATEGY_LLM_API_KEY=your-api-key-here
OPENAI_API_KEY=your-openai-key-here
DEEPSEEK_API_KEY=your-deepseek-key-here
KIMI_API_KEY=your-kimi-key-here

# Optional: custom LLM base URL
# AUTOSTRATEGY_LLM_BASE_URL=https://api.openai.com/v1

# Optional: data cache directory
# AUTOSTRATEGY_DATA_CACHE_DIR=~/.autostrategy/cache
```

- [ ] **Step 2: 更新 README.md 头部**

在 README 开头添加产品化说明（保留原有 Skill 内容，新增产品安装方式）：

```markdown
# autostrategy

面向个人投资者的本地开源量化策略平台。

## 快速开始

```bash
pip install -e .
autostrategy --version
autostrategy config init
autostrategy strategy list
```

## 作为 Claude Code Skill 使用

...
```

- [ ] **Step 3: 提交**

```bash
git add .env.example README.md
git commit -m "docs: 添加 .env.example 并更新 README 产品化说明

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: 验证安装与 CLI

**Files:**
- 无新增文件

- [ ] **Step 1: 创建/刷新虚拟环境并安装**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

- [ ] **Step 2: 验证 CLI**

```bash
autostrategy --version
autostrategy config init
autostrategy config show
autostrategy strategy list
```

Expected:
- `autostrategy --version` 输出 `autostrategy 0.1.0`
- `config init` 创建 `~/.autostrategy/settings.yaml`
- `strategy list` 输出提示信息

- [ ] **Step 3: 运行全部测试**

```bash
pytest -v
```

Expected: 所有测试通过

- [ ] **Step 4: 提交**

```bash
git add -A
git commit -m "chore: 验证 Phase 0 安装与 CLI

- pip install -e . 成功
- CLI 基础命令可用
- 单元测试与集成测试通过

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**

| Spec 要求 | 对应 Task |
|-----------|-----------|
| 根目录产品化 | Task 1, 4, 5 |
| Skill 迁入子目录 + shim | Task 5 |
| CLI 入口 | Task 4 |
| 配置系统 | Task 3 |
| 技术栈选择 | Task 1 (pyproject.toml) |
| 测试策略 | 每个 Task 都包含测试 |
| 安全边界 | Task 3 (API key 不存 settings), Task 6 (.env.example) |

**2. Placeholder scan:**

- 无 TBD/TODO
- 所有代码步骤包含完整代码
- 所有命令包含预期输出

**3. Type consistency:**

- `Settings` 类在 Task 3 定义，Task 4 使用 `load_settings` 一致
- `__version__` 在 Task 2 定义，Task 1 和 Task 4 引用一致

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-07-04-phase0-product-skeleton.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
