"""Integration tests for CodegenAgent."""

from autostrategy.agents.codegen_agent import CodegenAgent
from autostrategy.config import LLMConfig
from autostrategy.core.strategy import StrategyStatus
from autostrategy.core.workspace import Workspace


class FakeLLMClient:
    """Fake LLM client for testing without network."""

    def __init__(self, config):
        self.config = config

    def chat(self, messages, **kwargs):
        return '''
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
symbol: "000300.SH"
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
'''


def test_codegen_agent_generates_files(tmp_path, monkeypatch):
    workspace = Workspace(root=tmp_path)
    strategy = workspace.create_strategy("dual-ma")
    workspace.write_text_file(
        strategy.slug,
        "STRATEGY_DESIGN.md",
        "# 双均线策略\n\n## 策略概述\n\n基于均线交叉。\n\n## 买入条件\n\n- 金叉\n\n## 卖出条件\n\n- 死叉\n\n## 止损\n\n- 5% 止损\n\n## 仓位管理\n\n- 满仓\n",
    )

    agent = CodegenAgent(llm_config=LLMConfig())
    monkeypatch.setattr(agent, "llm_client", FakeLLMClient(agent.llm_config))

    updated = agent.codegen_and_save(workspace, strategy.slug)

    assert updated.status == StrategyStatus.CODED
    assert (tmp_path / "dual-ma" / "strategy.py").exists()
    assert (tmp_path / "dual-ma" / "config.yaml").exists()
    assert (tmp_path / "dual-ma" / "README.md").exists()
    assert (tmp_path / "dual-ma" / "requirements.txt").exists()
    assert (tmp_path / "dual-ma" / "data" / "fetch_data.py").exists()


def test_codegen_rejects_dangerous_python_patterns():
    agent = CodegenAgent(llm_config=LLMConfig())
    files = {
        "strategy.py": "import subprocess\n\ndef run_backtest(config):\n    return {}\n",
        "config.yaml": "initial_cash: 1000000\nstart_date: '2024-01-01'\nend_date: '2024-12-31'\ncommission: 0.0003\nslippage: 0.001\nmarket: A股\n",
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
