# Autostrategy

[![Python >= 3.11](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](https://www.python.org/)
[![Market](https://img.shields.io/badge/market-A%E8%82%A1%20%7C%20%E6%B8%AF%E8%82%A1%20%7C%20%E7%BE%8E%E8%82%A1-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

用自然语言创建、回测和模拟运行量化策略的本地开源工具。

你描述一个想法，Autostrategy 帮你把它变成可检查的策略设计文档，再生成代码，跑回测，最后在本地模拟运行里观察它会如何做决策。

> 免责声明：本项目仅用于学习、研究和策略原型验证，不构成任何投资建议。量化交易有风险，回测收益不代表未来表现。

## 为什么做这个

很多个人投资者不是卡在“不会写 Python”，而是卡在更前面：

- 想法说不清，策略条件散在脑子里。
- 代码跑起来了，但不知道 AI 有没有偷偷改了逻辑。
- 回测结果看起来不错，但不知道风险在哪里。
- 想观察策略的逐步决策，却没有一个轻量的本地工作台。

Autostrategy 的核心思路是：**先把策略写成清楚的设计文档，再让代码严格跟随文档。**

这不是让 AI 直接自由发挥写一段交易脚本。它更像一个本地策略工坊：先画图纸，再施工，再验收。

## 它能帮你做什么

| 你想做的事 | Autostrategy 做什么 |
|---|---|
| “帮我做一个双均线策略” | 生成策略设计文档、策略代码和本地回测结果 |
| “我有个模糊想法，但不知道怎么量化” | 把想法整理成买入、卖出、止损、仓位规则 |
| “我想先从模板开始” | 用内置双均线、网格、动量模板创建策略工作区 |
| “我想检查 AI 生成的策略有没有依据” | 保留 `STRATEGY_DESIGN.md`，让策略逻辑可读、可审查 |
| “我想看策略每一步会怎么判断” | 本地模拟运行，保存决策事件和运行结果 |

## 安装

建议先在虚拟环境中使用：

```bash
git clone git@github.com:Zach0911/autostrategy.git
cd autostrategy

python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

如果你的 shell 对引号比较敏感，也可以用：

```bash
pip install -e '.[all]'
```

开发和测试时可安装：

```bash
pip install -e '.[dev,api,web,mcp]'
npm install
```

## 快速开始

### 1. 初始化配置

```bash
autostrategy config init
```

这会创建本地配置文件：

```text
~/.autostrategy/settings.yaml
```

### 2. 配置 LLM

Autostrategy 使用 OpenAI-compatible 接口。API key 不写入项目文件，只从环境变量读取。

```bash
autostrategy config set llm.provider openai
autostrategy config set llm.model gpt-4o-mini
export AUTOSTRATEGY_LLM_API_KEY="你的 API Key"
```

如果你只想体验模板和本地回测，可以先不配置 LLM。

### 3. 创建一个策略

用内置模板创建：

```bash
autostrategy strategy create dual-ma --template dual-ma
```

查看策略文件位置：

```bash
autostrategy strategy paths dual-ma
```

### 4. 用自然语言生成策略设计

```bash
autostrategy design create \
  --prompt "帮我做一个 A 股双均线策略，快线上穿慢线买入，跌破慢线卖出，并控制最大仓位" \
  --name dual-ma \
  --template dual-ma
```

生成后重点看这个文件：

```text
STRATEGY_DESIGN.md
```

它是策略图纸。买入条件、卖出条件、止损、仓位管理都应该在这里说清楚。

### 5. 生成代码

```bash
autostrategy codegen create dual-ma --force
```

生成的策略工作区会包含：

```text
strategy.py
config.yaml
README.md
requirements.txt
fetch_data.py
```

### 6. 回测

```bash
autostrategy backtest run dual-ma
```

回测结果会保存到策略工作区：

```text
backtest/results/backtest_result.json
```

查看策略当前状态：

```bash
autostrategy strategy show dual-ma
```

## 浏览器工作台

如果你不想一直在命令行里看文件，可以启动本地工作台：

```bash
npm run build
autostrategy serve --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000/
```

你可以在浏览器里查看策略、生成设计、触发代码生成、运行回测、预览产物，并查看 LLM 配置状态。

建议只监听 `127.0.0.1`。Autostrategy 是本地研究工具，不是多用户远程服务。

## 模拟运行

回测回答的是：“这套规则在历史数据上表现如何？”

模拟运行回答的是：“如果按时间顺序重放，策略每一步会怎么做决策？”

策略代码只要暴露：

```python
def run_paper(config):
    ...
```

就可以启动模拟运行：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/strategies/dual-ma/paper-run
```

结果会写入：

```text
paper_run/results/paper_run_result.json
paper_run/results/paper_run_events.jsonl
paper_run/logs/paper_run.log
```

当前模拟运行是 replay-first，本地重放历史数据和策略决策。它还不是实盘交易，也不连接真实 broker。

## 工作流

```text
Describe  →  Design  →  Generate  →  Backtest  →  Paper Run  →  Iterate
想法          图纸        代码          回测          模拟观察        继续改进
```

1. **Describe**：用自然语言描述策略。
2. **Design**：生成可审查的策略设计文档。
3. **Generate**：根据设计文档生成策略代码。
4. **Backtest**：在本地跑历史回测。
5. **Paper Run**：重放策略决策，查看事件流和运行结果。
6. **Iterate**：修改设计，再生成、再验证。

## 内置模板

当前内置三个策略模板：

| 模板 | 适合场景 |
|---|---|
| `dual-ma` | 双均线趋势跟随，新手最容易理解 |
| `grid` | 震荡行情里的网格思路 |
| `momentum` | 动量策略原型 |

查看模板列表：

```bash
autostrategy strategy create --help
```

## 支持市场

Autostrategy 的设计目标覆盖：

| 市场 | 状态 | 说明 |
|---|---|---|
| A 股 | 优先支持 | 适合个人投资者研究和验证想法 |
| 港股 | 规划支持 | 可通过 Futu OpenD 等数据源扩展 |
| 美股 | 规划支持 | 可通过 Futu OpenD、yfinance 等数据源扩展 |

数据源不是强绑定的。策略工作区可以按自己的方式提供历史数据，只要 `strategy.py` 暴露约定入口即可。

## 安全边界

Autostrategy 会尽量降低 AI 生成代码的风险：

- API key 只从环境变量读取，不写入项目配置。
- 生成代码会拒绝明显危险的模式，例如 `os.system`、`subprocess`、`eval(`、`exec(`。
- 工作区文件访问会阻止 `../` 和绝对路径穿越。
- 回测和模拟运行默认在本地执行，不提供远程多用户沙箱。

这仍然是一个会执行本地策略代码的研究工具。运行第三方策略前，请先读代码。

## 命令速查

```bash
# 配置
autostrategy config init
autostrategy config show
autostrategy config set llm.model gpt-4o-mini

# 策略管理
autostrategy strategy create dual-ma --template dual-ma
autostrategy strategy list
autostrategy strategy show dual-ma
autostrategy strategy paths dual-ma

# AI 工作流
autostrategy design create --prompt "帮我做一个双均线策略" --name dual-ma
autostrategy codegen create dual-ma --force

# 验证
autostrategy backtest run dual-ma

# 本地工作台
autostrategy serve --host 127.0.0.1 --port 8000
```

## 给开发者

常用验证命令：

```bash
python -m pytest
npm test -- --run
npm run build
npm run typecheck
```

项目主要目录：

```text
src/autostrategy/
├── cli/          # 命令行入口
├── agents/       # 设计与代码生成 Agent
├── core/         # 策略、工作区、模板、回测核心
├── services/     # 业务服务层
├── api/          # 本地 API
├── web/          # 本地浏览器工作台
├── mcp/          # 本地 Agent 工具适配
└── templates/    # 内置策略模板
```

更详细的产品设计和阶段计划在 [docs/superpowers/](docs/superpowers/) 中。

## License

MIT License
