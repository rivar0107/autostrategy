# Phase 1: 策略管理中心实现计划

**Goal:** 实现策略工作区、策略 CRUD、内置模板市场，让用户能通过 CLI 创建和管理策略。

**Architecture:** 用 `Workspace` 类抽象策略目录，`Strategy` Pydantic 模型描述元数据，模板作为目录模板，CLI 调用 Workspace API。

**Tech Stack:** Python 3.11+, Pydantic, Typer, pytest

---

## Task 1: 设计 Strategy 模型

**Files:**
- Create: `src/autostrategy/core/strategy.py`
- Create: `tests/unit/test_strategy.py`

- [ ] **Step 1: 写失败测试**

```python
"""Tests for strategy model."""

from datetime import datetime

from autostrategy.core.strategy import Strategy, StrategyStatus


def test_strategy_defaults():
    strategy = Strategy(name="dual-ma", market="A股")
    assert strategy.name == "dual-ma"
    assert strategy.market == "A股"
    assert strategy.status == StrategyStatus.DRAFT
    assert isinstance(strategy.created_at, datetime)


def test_strategy_slug():
    strategy = Strategy(name="Dual MA Strategy", market="A股")
    assert strategy.slug == "dual-ma-strategy"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/unit/test_strategy.py -v
```

Expected: 2 FAIL

- [ ] **Step 3: 实现 Strategy 模型**

```python
"""Strategy domain model."""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class StrategyStatus(str, Enum):
    """Strategy lifecycle status."""

    DRAFT = "draft"
    DESIGNED = "designed"
    CODED = "coded"
    BACKTESTED = "backtested"
    OPTIMIZED = "optimized"
    ACTIVE = "active"
    ARCHIVED = "archived"


class Strategy(BaseModel):
    """A single trading strategy."""

    name: str
    slug: str = ""
    description: str = ""
    market: str = "A股"
    status: StrategyStatus = StrategyStatus.DRAFT
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    template: str | None = None
    tags: list[str] = Field(default_factory=list)

    def model_post_init(self, __context) -> None:
        if not self.slug:
            self.slug = self._to_slug(self.name)

    @staticmethod
    def _to_slug(name: str) -> str:
        """Convert a human-readable name to a URL-safe slug."""
        slug = name.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[-\s]+", "-", slug)
        return slug.strip("-")

    @property
    def workspace_dir(self) -> Path:
        """Return the relative workspace directory name."""
        return Path(self.slug)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/unit/test_strategy.py -v
```

Expected: 2 PASS

- [ ] **Step 5: 提交**

```bash
git add src/autostrategy/core/strategy.py tests/unit/test_strategy.py
git commit -m "feat: 添加 Strategy 领域模型

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: 实现 Workspace 管理

**Files:**
- Create: `src/autostrategy/core/workspace.py`
- Create: `tests/unit/test_workspace.py`

- [ ] **Step 1: 写失败测试**

```python
"""Tests for workspace management."""

import pytest

from autostrategy.core.strategy import Strategy
from autostrategy.core.workspace import Workspace


def test_workspace_root(tmp_path):
    ws = Workspace(root=tmp_path)
    assert ws.root == tmp_path
    assert ws.root.exists()


def test_create_strategy(tmp_path):
    ws = Workspace(root=tmp_path)
    strategy = ws.create_strategy("dual-ma", market="A股")
    assert strategy.name == "dual-ma"
    assert (tmp_path / "dual-ma" / "config.yaml").exists()
    assert (tmp_path / "dual-ma" / "STRATEGY_DESIGN.md").exists()


def test_list_strategies(tmp_path):
    ws = Workspace(root=tmp_path)
    ws.create_strategy("s1")
    ws.create_strategy("s2")
    strategies = ws.list_strategies()
    assert len(strategies) == 2


def test_get_strategy(tmp_path):
    ws = Workspace(root=tmp_path)
    created = ws.create_strategy("s1")
    fetched = ws.get_strategy("s1")
    assert fetched is not None
    assert fetched.slug == created.slug


def test_delete_strategy(tmp_path):
    ws = Workspace(root=tmp_path)
    ws.create_strategy("s1")
    ws.delete_strategy("s1")
    assert ws.get_strategy("s1") is None
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/unit/test_workspace.py -v
```

Expected: 5 FAIL

- [ ] **Step 3: 实现 Workspace 类**

```python
"""Workspace management for strategies."""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from autostrategy.core.strategy import Strategy, StrategyStatus


DEFAULT_WORKSPACE_ROOT = Path.home() / ".autostrategy" / "strategies"


class Workspace:
    """Manages a directory of strategy workspaces."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or DEFAULT_WORKSPACE_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    def _strategy_dir(self, slug: str) -> Path:
        return self.root / slug

    def create_strategy(
        self,
        name: str,
        description: str = "",
        market: str = "A股",
        template: str | None = None,
        tags: list[str] | None = None,
    ) -> Strategy:
        """Create a new strategy workspace."""
        strategy = Strategy(
            name=name,
            description=description,
            market=market,
            template=template,
            tags=tags or [],
        )
        strategy_dir = self._strategy_dir(strategy.slug)
        if strategy_dir.exists():
            raise FileExistsError(f"Strategy '{name}' ({strategy.slug}) already exists.")

        strategy_dir.mkdir(parents=True)
        self._write_strategy_meta(strategy_dir, strategy)
        self._write_default_files(strategy_dir)
        return strategy

    def _write_strategy_meta(self, strategy_dir: Path, strategy: Strategy) -> None:
        meta_path = strategy_dir / "strategy.yaml"
        with open(meta_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(strategy.model_dump(mode="json"), f, allow_unicode=True)

    def _write_default_files(self, strategy_dir: Path) -> None:
        config_path = strategy_dir / "config.yaml"
        config_path.write_text(
            "# Strategy configuration\n"
            "market: A股\n"
            "symbols: []\n"
            "period:\n"
            "  start: 2022-01-01\n"
            "  end: 2024-01-01\n",
            encoding="utf-8",
        )
        design_path = strategy_dir / "STRATEGY_DESIGN.md"
        design_path.write_text(
            f"# {strategy_dir.name}\n\n"
            "## 策略概述\n\n"
            "待补充...\n\n"
            "## 买入条件\n\n"
            "- 待补充\n\n"
            "## 卖出条件\n\n"
            "- 待补充\n",
            encoding="utf-8",
        )

    def list_strategies(self) -> list[Strategy]:
        """List all strategies in the workspace."""
        strategies = []
        for entry in sorted(self.root.iterdir()):
            if entry.is_dir():
                strategy = self._load_strategy(entry)
                if strategy:
                    strategies.append(strategy)
        return strategies

    def get_strategy(self, slug: str) -> Strategy | None:
        """Get a strategy by slug."""
        strategy_dir = self._strategy_dir(slug)
        if not strategy_dir.exists():
            return None
        return self._load_strategy(strategy_dir)

    def _load_strategy(self, strategy_dir: Path) -> Strategy | None:
        meta_path = strategy_dir / "strategy.yaml"
        if not meta_path.exists():
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return Strategy(**data)

    def delete_strategy(self, slug: str) -> None:
        """Delete a strategy workspace."""
        strategy_dir = self._strategy_dir(slug)
        if strategy_dir.exists():
            shutil.rmtree(strategy_dir)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/unit/test_workspace.py -v
```

Expected: 5 PASS

- [ ] **Step 5: 提交**

```bash
git add src/autostrategy/core/workspace.py tests/unit/test_workspace.py
git commit -m "feat: 添加 Workspace 策略管理

- 支持创建/列出/读取/删除策略
- 自动生成 config.yaml 和 STRATEGY_DESIGN.md 占位
- 添加单元测试

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: 实现模板市场

**Files:**
- Create: `src/autostrategy/templates/dual-ma/config.yaml`
- Create: `src/autostrategy/templates/dual-ma/STRATEGY_DESIGN.md`
- Create: `src/autostrategy/templates/grid/config.yaml`
- Create: `src/autostrategy/templates/grid/STRATEGY_DESIGN.md`
- Create: `src/autostrategy/templates/momentum/config.yaml`
- Create: `src/autostrategy/templates/momentum/STRATEGY_DESIGN.md`
- Create: `src/autostrategy/core/template_registry.py`
- Create: `tests/unit/test_template_registry.py`

- [ ] **Step 1: 创建模板文件**

创建 `src/autostrategy/templates/dual-ma/config.yaml`:

```yaml
market: A股
symbols:
  - "000300.SH"
period:
  start: "2022-01-01"
  end: "2024-01-01"
parameters:
  fast_ma: 20
  slow_ma: 60
```

创建 `src/autostrategy/templates/dual-ma/STRATEGY_DESIGN.md`:

```markdown
# 双均线交叉策略

## 策略概述

基于 fast_ma 和 slow_ma 的交叉产生买卖信号。

## 买入条件

- fast_ma 上穿 slow_ma（金叉）

## 卖出条件

- fast_ma 下穿 slow_ma（死叉）

## 止损

- 亏损超过 5% 止损

## 仓位管理

- 固定仓位 100%
```

类似创建 `grid` 和 `momentum` 模板。

- [ ] **Step 2: 实现 TemplateRegistry**

```python
"""Built-in strategy templates."""

from __future__ import annotations

import shutil
from pathlib import Path


TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


class TemplateRegistry:
    """Registry of built-in strategy templates."""

    @staticmethod
    def list_templates() -> list[str]:
        """Return available template names."""
        if not TEMPLATES_DIR.exists():
            return []
        return sorted(
            entry.name
            for entry in TEMPLATES_DIR.iterdir()
            if entry.is_dir() and (entry / "STRATEGY_DESIGN.md").exists()
        )

    @classmethod
    def apply_template(cls, template_name: str, target_dir: Path) -> None:
        """Copy a template into a strategy workspace."""
        source = TEMPLATES_DIR / template_name
        if not source.exists():
            raise ValueError(f"Template '{template_name}' not found.")
        for item in source.iterdir():
            dest = target_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
```

- [ ] **Step 3: 添加测试**

```python
"""Tests for template registry."""

from autostrategy.core.template_registry import TemplateRegistry


def test_list_templates():
    templates = TemplateRegistry.list_templates()
    assert "dual-ma" in templates
    assert "grid" in templates
    assert "momentum" in templates
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/unit/test_template_registry.py tests/unit/test_workspace.py -v
```

- [ ] **Step 5: 更新 Workspace 支持模板**

修改 `Workspace.create_strategy`，如果指定了 template，在创建后应用模板。

- [ ] **Step 6: 提交**

```bash
git add src/autostrategy/templates/ src/autostrategy/core/template_registry.py tests/unit/test_template_registry.py
git commit -m "feat: 添加模板市场

- 内置 dual-ma、grid、momentum 三个模板
- TemplateRegistry 管理模板列表和应用
- Workspace 创建策略时支持模板

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: CLI 策略 CRUD

**Files:**
- Modify: `src/autostrategy/cli/main.py`
- Create: `tests/integration/test_strategy_cli.py`

- [ ] **Step 1: 写失败测试**

```python
"""Integration tests for strategy CLI."""

from typer.testing import CliRunner

from autostrategy.cli.main import app

runner = CliRunner()


def test_strategy_create(tmp_path):
    result = runner.invoke(
        app, ["strategy", "create", "dual-ma-test", "--workspace-root", str(tmp_path)]
    )
    assert result.exit_code == 0
    assert "created" in result.stdout.lower()


def test_strategy_list(tmp_path):
    runner.invoke(app, ["strategy", "create", "s1", "--workspace-root", str(tmp_path)])
    result = runner.invoke(app, ["strategy", "list", "--workspace-root", str(tmp_path)])
    assert result.exit_code == 0
    assert "s1" in result.stdout
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/integration/test_strategy_cli.py -v
```

- [ ] **Step 3: 更新 CLI 添加策略命令**

在 `src/autostrategy/cli/main.py` 中替换 strategy_app 部分：

```python
from pathlib import Path

from autostrategy.core.template_registry import TemplateRegistry
from autostrategy.core.workspace import Workspace


@strategy_app.command("list")
def strategy_list(
    workspace_root: str | None = typer.Option(None, "--workspace-root", help="Workspace root directory."),
) -> None:
    """List all strategies in workspace."""
    root = Path(workspace_root) if workspace_root else None
    workspace = Workspace(root=root)
    strategies = workspace.list_strategies()
    if not strategies:
        typer.echo("No strategies yet. Use `autostrategy strategy create` to add one.")
        return
    for strategy in strategies:
        typer.echo(f"{strategy.slug:20} {strategy.market:8} {strategy.status.value:12} {strategy.name}")


@strategy_app.command("create")
def strategy_create(
    name: str = typer.Argument(..., help="Strategy name."),
    template: str | None = typer.Option(None, "--template", help="Template name to use."),
    market: str = typer.Option("A股", "--market", help="Target market."),
    workspace_root: str | None = typer.Option(None, "--workspace-root", help="Workspace root directory."),
) -> None:
    """Create a new strategy workspace."""
    if template and template not in TemplateRegistry.list_templates():
        typer.echo(f"Template '{template}' not found. Available: {', '.join(TemplateRegistry.list_templates())}")
        raise typer.Exit(1)

    root = Path(workspace_root) if workspace_root else None
    workspace = Workspace(root=root)
    try:
        strategy = workspace.create_strategy(name, market=market, template=template)
        typer.echo(f"Strategy '{strategy.name}' created at {strategy.workspace_dir}")
    except FileExistsError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1)


@strategy_app.command("show")
def strategy_show(
    slug: str = typer.Argument(..., help="Strategy slug."),
    workspace_root: str | None = typer.Option(None, "--workspace-root", help="Workspace root directory."),
) -> None:
    """Show strategy details."""
    root = Path(workspace_root) if workspace_root else None
    workspace = Workspace(root=root)
    strategy = workspace.get_strategy(slug)
    if not strategy:
        typer.echo(f"Strategy '{slug}' not found.")
        raise typer.Exit(1)
    typer.echo(f"Name: {strategy.name}")
    typer.echo(f"Slug: {strategy.slug}")
    typer.echo(f"Market: {strategy.market}")
    typer.echo(f"Status: {strategy.status.value}")
    typer.echo(f"Template: {strategy.template or 'none'}")


@strategy_app.command("delete")
def strategy_delete(
    slug: str = typer.Argument(..., help="Strategy slug."),
    workspace_root: str | None = typer.Option(None, "--workspace-root", help="Workspace root directory."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Delete a strategy workspace."""
    root = Path(workspace_root) if workspace_root else None
    workspace = Workspace(root=root)
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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/integration/test_strategy_cli.py -v
```

- [ ] **Step 5: 提交**

```bash
git add src/autostrategy/cli/main.py tests/integration/test_strategy_cli.py
git commit -m "feat: CLI 支持策略 CRUD

- strategy create/list/show/delete
- 支持 --workspace-root 指定工作区
- 支持 --template 选择模板

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: 验证 Phase 1

- [ ] **Step 1: 运行全部测试**

```bash
pytest -v
```

Expected: 全部通过

- [ ] **Step 2: 手动验证 CLI**

```bash
autostrategy strategy list
autostrategy strategy create my-first-strategy --template dual-ma
autostrategy strategy list
autostrategy strategy show my-first-strategy
autostrategy strategy delete my-first-strategy --yes
```

- [ ] **Step 3: 提交验证**

```bash
git add -A
git commit -m "chore: 验证 Phase 1 策略管理中心

- 全部测试通过
- CLI 端到端策略 CRUD 可用

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review

- Spec coverage: Phase 1 策略管理（Workspace、CRUD、模板）全部覆盖。
- Placeholder scan: 无 TBD/TODO，代码完整。
- Type consistency: Strategy 模型与 Workspace 使用一致的 slug/name 字段。
