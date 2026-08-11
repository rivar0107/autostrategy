"""Integration tests for CodegenAgent."""

from autostrategy.agents.codegen_agent import CodegenAgent
from autostrategy.agents.prompts.codegen import CODEGEN_SYSTEM_PROMPT
from autostrategy.config import LLMConfig
from autostrategy.core.strategy import StrategyStatus
from autostrategy.core.workspace import Workspace


class FakeLLMClient:
    """Fake LLM client for testing without network."""

    def __init__(self, config):
        self.config = config

    def chat(self, messages, **kwargs):
        return """
=== FILE: strategy.py ===
```python
def run_backtest(config: dict) -> dict:
    return {
        "annual_return": 12.5,
        "max_drawdown": 8.0,
        "sharpe": 1.4,
        "win_rate": 52.0,
        "profit_loss_ratio": 1.8,
        "total_trades": 20,
        "daily_values": [
            {"date": "2024-01-01", "value": 1000000},
            {"date": "2024-01-02", "value": 1005000},
        ],
        "initial_cash": 1000000,
    }

def run_paper(config: dict) -> dict:
    return {"events": [], "paper": {"cash": config.get("initial_cash", 0)}}

def generate_intents(context: dict) -> dict:
    return {"decisions": [], "intents": [], "strategy_state": context.get("strategy_state", {})}
```

=== FILE: config.yaml ===
```yaml
initial_cash: 1000000
start_date: "2024-01-01"
end_date: "2024-12-31"
benchmark: "000300.SH"
commission: 0.0003
stamp_tax: 0.001
slippage: 0.001
market: "A股"
symbols:
  - "510500.SH"
data_source: "akshare"
data_cycle: "daily"
indicators:
  fast_ma: 5
  slow_ma: 20
risk:
  stop_loss_pct: 5
```

=== FILE: README.md ===
```markdown
# 双均线策略

## 策略概述

基于均线交叉的示例策略。
```

=== FILE: requirements.txt ===
```text
pandas
numpy
```

=== FILE: data/fetch_data.py ===
```python
def fetch(config: dict):
    return None
```
"""


class RecordingLLMClient(FakeLLMClient):
    """Record prompts while returning a complete valid response."""

    def __init__(self, config):
        super().__init__(config)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return super().chat(messages, **kwargs)


class RepairingConfigLLMClient(RecordingLLMClient):
    """Return an invalid config once so only that step needs repair."""

    def __init__(self, config):
        super().__init__(config)
        self.returned_invalid_config = False

    def chat(self, messages, **kwargs):
        prompt = messages[-1].content
        if "目标文件：config.yaml" in prompt and not self.returned_invalid_config:
            self.calls.append(messages)
            self.returned_invalid_config = True
            return """
=== FILE: config.yaml ===
```yaml
backtest:
  initial_cash: 1000000
```
"""
        return super().chat(messages, **kwargs)


def test_codegen_agent_generates_files(tmp_path, monkeypatch):
    workspace = Workspace(root=tmp_path)
    strategy = workspace.create_strategy("dual-ma")
    workspace.write_text_file(
        strategy.slug,
        "STRATEGY_DESIGN.md",
        "# 双均线策略\n\n## 策略概述\n\n基于均线交叉。\n\n"
        "## 买入条件\n\n- 金叉\n\n## 卖出条件\n\n- 死叉\n\n"
        "## 止损\n\n- 5% 止损\n\n## 仓位管理\n\n- 满仓\n",
    )

    agent = CodegenAgent(llm_config=LLMConfig())
    monkeypatch.setattr(agent, "llm_client", FakeLLMClient(agent.llm_config))

    updated = agent.codegen_and_save(workspace, strategy.slug)

    assert updated.status == StrategyStatus.CODED
    assert updated.version == 2
    assert updated.content_digest
    assert (tmp_path / "dual-ma" / "strategy.py").exists()
    assert (tmp_path / "dual-ma" / "config.yaml").exists()
    assert (tmp_path / "dual-ma" / "README.md").exists()
    assert (tmp_path / "dual-ma" / "requirements.txt").exists()
    assert (tmp_path / "dual-ma" / "data" / "fetch_data.py").exists()


def test_codegen_generates_files_step_by_step_in_dependency_order(tmp_path, monkeypatch):
    workspace = Workspace(root=tmp_path)
    strategy = workspace.create_strategy("repair-demo")
    workspace.write_text_file(
        strategy.slug,
        "STRATEGY_DESIGN.md",
        "# 修复示例\n\n## 策略概述\n\n严格按设计生成。\n",
    )
    agent = CodegenAgent(llm_config=LLMConfig())
    recording_client = RecordingLLMClient(agent.llm_config)
    monkeypatch.setattr(agent, "llm_client", recording_client)

    updated = agent.codegen_and_save(workspace, strategy.slug)

    assert updated.status == StrategyStatus.CODED
    expected_order = [
        "config.yaml",
        "data/fetch_data.py",
        "strategy.py",
        "requirements.txt",
        "README.md",
    ]
    assert len(recording_client.calls) == len(expected_order)
    for messages, expected_path in zip(recording_client.calls, expected_order, strict=True):
        assert f"目标文件：{expected_path}" in messages[-1].content
        assert "只输出这个文件" in messages[-1].content


def test_codegen_repairs_only_the_current_invalid_file(tmp_path, monkeypatch):
    workspace = Workspace(root=tmp_path)
    strategy = workspace.create_strategy("repair-config")
    workspace.write_text_file(
        strategy.slug,
        "STRATEGY_DESIGN.md",
        "# 修复配置\n\n## 策略概述\n\n严格按设计生成。\n",
    )
    agent = CodegenAgent(llm_config=LLMConfig())
    repairing_client = RepairingConfigLLMClient(agent.llm_config)
    monkeypatch.setattr(agent, "llm_client", repairing_client)

    updated = agent.codegen_and_save(workspace, strategy.slug)

    assert updated.status == StrategyStatus.CODED
    prompts = [messages[-1].content for messages in repairing_client.calls]
    assert ["目标文件：config.yaml" in prompt for prompt in prompts].count(True) == 2
    for relative_path in [
        "data/fetch_data.py",
        "strategy.py",
        "requirements.txt",
        "README.md",
    ]:
        assert [f"目标文件：{relative_path}" in prompt for prompt in prompts].count(True) == 1
    assert "config.yaml missing required key: start_date" in prompts[1]


def test_codegen_prompt_matches_quality_contract():
    for key in [
        "initial_cash",
        "start_date",
        "end_date",
        "commission",
        "slippage",
        "market",
    ]:
        assert key in CODEGEN_SYSTEM_PROMPT
    assert "config.yaml 顶层" in CODEGEN_SYSTEM_PROMPT
    assert "symbols" in CODEGEN_SYSTEM_PROMPT
    assert "## 策略概述" in CODEGEN_SYSTEM_PROMPT
    assert "## 核心逻辑" in CODEGEN_SYSTEM_PROMPT
    assert "不要输出未完成的代码" in CODEGEN_SYSTEM_PROMPT
    assert "fields" in CODEGEN_SYSTEM_PROMPT
    assert "market" in CODEGEN_SYSTEM_PROMPT
    assert "百分数" in CODEGEN_SYSTEM_PROMPT
    assert "def run_paper(config: dict)" in CODEGEN_SYSTEM_PROMPT
    assert "def generate_intents(context: dict) -> dict" in CODEGEN_SYSTEM_PROMPT
    assert "不得发起 HTTP" in CODEGEN_SYSTEM_PROMPT


def test_codegen_rejects_strategy_without_client_simulation_entry():
    agent = CodegenAgent(llm_config=LLMConfig())

    report = agent._check_generated_file(
        "strategy.py",
        {"strategy.py": "def run_backtest(config):\n    return {}\n"},
    )

    assert not report.passed
    assert any("generate_intents" in error for error in report.errors)


def test_codegen_rejects_strategy_without_local_paper_entry():
    agent = CodegenAgent(llm_config=LLMConfig())

    report = agent._check_generated_file(
        "strategy.py",
        {
            "strategy.py": (
                "def run_backtest(config):\n    return {}\n\n"
                "def generate_intents(context):\n    return {'intents': []}\n"
            )
        },
    )

    assert not report.passed
    assert any("run_paper" in error for error in report.errors)


def test_codegen_rejects_config_without_explicit_execution_universe():
    agent = CodegenAgent(llm_config=LLMConfig())

    report = agent._check_generated_file(
        "config.yaml",
        {
            "config.yaml": (
                "initial_cash: 1000000\nstart_date: '2024-01-01'\n"
                "end_date: '2024-12-31'\ncommission: 0.0003\n"
                "slippage: 0.001\nmarket: A股\n"
            )
        },
    )

    assert not report.passed
    assert any("symbols" in error for error in report.errors)


def test_codegen_rejects_empty_execution_universe():
    agent = CodegenAgent(llm_config=LLMConfig())
    config = (
        "initial_cash: 1000000\nstart_date: '2024-01-01'\n"
        "end_date: '2024-12-31'\ncommission: 0.0003\n"
        "slippage: 0.001\nmarket: A股\nsymbols: []\n"
    )

    report = agent._check_generated_file("config.yaml", {"config.yaml": config})

    assert not report.passed
    assert any("non-empty list" in error for error in report.errors)


def test_codegen_rejects_unsupported_ftshare_arguments():
    agent = CodegenAgent(llm_config=LLMConfig())
    files = {
        "data/fetch_data.py": (
            "from autostrategy.data.ftshare import fetch_daily_ohlc\n"
            "def fetch(config):\n"
            "    return fetch_daily_ohlc('000905.SH', fields=['close'], market='A股')\n"
        )
    }

    report = agent._check_generated_file("data/fetch_data.py", files)

    assert not report.passed
    assert any("unsupported" in error for error in report.errors)


def test_codegen_rejects_dangerous_python_patterns():
    agent = CodegenAgent(llm_config=LLMConfig())
    files = {
        "strategy.py": "import subprocess\n\ndef run_backtest(config):\n    return {}\n",
        "config.yaml": (
            "initial_cash: 1000000\nstart_date: '2024-01-01'\nend_date: '2024-12-31'\n"
            "commission: 0.0003\nslippage: 0.001\nmarket: A股\n"
        ),
        "README.md": "# Demo\n\n## 策略概述\n\nDemo\n",
        "requirements.txt": "pandas\nnumpy\n",
        "data/fetch_data.py": "def fetch(config):\n    return None\n",
    }

    report = agent.check_generated_files(files)

    assert not report.passed
    assert any("dangerous pattern" in error for error in report.errors)


def test_codegen_rejects_missing_design(tmp_path):
    workspace = Workspace(root=tmp_path)
    strategy = workspace.create_strategy("empty")
    workspace.write_text_file(strategy.slug, "STRATEGY_DESIGN.md", "")
    agent = CodegenAgent(llm_config=LLMConfig())

    try:
        agent.codegen_and_save(workspace, strategy.slug)
    except ValueError as exc:
        assert "STRATEGY_DESIGN.md" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
