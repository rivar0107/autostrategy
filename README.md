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

策略工作台的“研究流程”页提供完整的可复现实验闭环：冻结数据、运行基线、生成诊断、隔离评估候选、执行一次样本外验证，以及明确接受、拒绝或回滚版本。

建议只监听 `127.0.0.1`。Autostrategy 是本地研究工具，不是多用户远程服务。

## 可复现研究与自动优化

研究流程不是直接反复修改当前策略，而是由三个持久化对象共同约束：

- `StrategyVersion`：保存不可变的设计、代码、配置和数据适配器快照。候选版本在接受前不会覆盖当前策略。
- `DatasetManifest`：冻结数据文件、数据摘要、标的、复权、基准、手续费、滑点，以及 train/validation/test 三段无重叠日期区间。
- `ExperimentSession`：记录基线 run、结构化诊断、候选版本、样本外结果和人工决策原因；服务重启后仍可继续读取。

状态只能依次推进：

```text
created
→ baseline_completed
→ diagnosed
→ optimized
→ oos_validated
→ awaiting_decision
→ accepted / rejected
```

核心约束：

- 基线和自动优化只使用训练集与验证集，不能访问测试日期。
- 一个候选只允许改变一个配置叶子，默认最多评估 5 个候选。
- 候选至少需要超过基线分数、满足最小交易次数且不突破最大回撤门槛。
- 测试集从执行开始即视为已经揭晓，即使执行失败也不能在同一实验中重试。
- 默认样本外门槛要求至少 30 笔交易、最大回撤不超过 20%，候选分数相对基础版本退化不超过 5 分。
- 接受要求样本外通过、基础版本和当前工作区没有变化，并且必须填写决策原因。
- 拒绝不会修改当前策略；回滚恢复被接受的祖先版本，同时保留后续版本、回测和审计历史。

普通回测历史也会记录 `version_id`；实验回测还会记录 `manifest_id`、`session_id`、`phase` 和 `candidate_id`，因此每个结果都能追溯到确切代码、数据和研究阶段。

研究用 `ExperimentSession` 与 FT 客户端模拟设计中的 `SimulationSession` 完全独立：前者管理回测研究和版本决策，后者未来管理委托、成交、持仓和账户状态。研究会话不会写入或复用模拟交易状态。

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

模拟运行分为两种明确隔离的模式：默认的 `local_replay` 在本地重放历史数据；`ft_client_simulation` 只连接同机非凸智能交易终端中的白名单模拟账户。平台不提供实盘开关。

#### 本地 replay 数据（feed）

模拟运行的行情来源可以配置为本地 feed，完全不依赖网络。在 `config.yaml` 中加：

```yaml
feed:
  path: "data/feed.csv"     # 支持 .csv / .jsonl，列：date,symbol,open,high,low,close,volume
  start: "2024-01-01"       # 可选，时间窗口
  end: "2025-12-31"
```

配置后 workflow 会把 bar 事件注入 `config['feed_bars']` 供 `run_paper(config)` 消费；策略没有 `run_paper` 时会按 feed 自动逐 bar 重放虚拟账户。行情数据本身用 FTShare MCP 下载（见策略目录下的 `fetch_data.py`）。示例见 `examples/dynamic-grid-multi-market`。

模拟运行会同步维护一个虚拟账户：`paper_run_result.json` 中的 `paper` 字段给出账户快照（`initial_cash`、`cash`、`equity`、`final_value` 和 `positions` 持仓明细），买入/卖出决策会按事件价格成交并更新现金与持仓；现金不足或持仓不足的委托会被拒绝并记录在事件流中。前端 Paper Run 面板会直接展示账户摘要与持仓表。

运行结束后自动生成复盘摘要：`paper_run/results/paper_run_review.md` 汇总收益、回撤、成交笔数、已实现/未实现盈亏与关键买卖/拒绝事件，可直接阅读，也可作为后续优化（Learning Agent）的结构化输入；复盘过程只读，不会修改策略代码。

#### 非凸客户端模拟盘

策略完成回测并暴露本地回放入口 `run_paper(config)` 和纯计算入口 `generate_intents(context)` 后，可在 Web 工作台的“模拟运行”中选择“非凸客户端模拟盘”。页面会要求客户填写：

- 本机客户端地址（只允许 `localhost`/`127.0.0.1`）、客户端版本、非凸账号和密码处理方式。
- 非凸密码；只在当前服务进程内存使用，不写入设置、日志、artifact 或 API 响应。
- 已确认的模拟交易账户白名单、本次策略可能下单的全部沪深 A 股/ETF，以及逐标的客户端代码映射。动态选股策略须在启动前固化最大执行股票池；指数仅可作行情基准，不能下单。
- 观察/人工确认/自动模拟模式、TWAP 结构化参数、执行窗口和仓位风控上限。
- `external_id` 最大长度，以及可在母单查询中完整返回的联调确认（当前幂等 ID 需要至少 64 字符）。

最低支持客户端版本为 3.11.4。第一次接入建议先运行观察模式；连接、账户登录、下单引擎、资金持仓、代码映射、算法和 `external_id` 任一硬检查失败时，平台不会提交母单。客户端会话 artifacts 独立保存在 `paper_run/client_sessions/<session_id>/`，不会覆盖本地 replay 结果。

运行中的客户端模拟盘会通过 FTShare 读取当日一分钟价格并聚合为不跨午休的 10 分钟 K 线，以所有执行标的和基准指数共有的最新完成 K 线为计算时点。每根 10 分钟 K 线只触发一次 `generate_intents(context)`；`history_by_symbol` 仍保留截至上一交易日的日线历史，供 MA250 等中长期指标使用。自动模式仅在客户配置的执行窗口内提交，暂停状态继续对账但不生成新意图，行情时间倒退时会话进入 `needs_attention`。

## 工作流

```text
Describe → Design → Generate → Backtest → Diagnose → Optimize → OOS → Accept/Rollback
想法         图纸       代码        基线        诊断         优化      样本外       接受/回滚
```

1. **Describe**：用自然语言描述策略。
2. **Design**：生成可审查的策略设计文档。
3. **Generate**：根据设计文档生成策略代码。
4. **Backtest**：在本地跑历史回测。
5. **Diagnose**：把回测证据转换成结构化问题与改进假设。
6. **Optimize**：在隔离候选版本上使用训练集和验证集比较单变量改动。
7. **OOS**：候选确定后一次性揭晓测试集，检验泛化能力。
8. **Accept/Rollback**：人工填写理由后接受、拒绝或恢复已接受的祖先版本。

`Paper Run` 是独立的运行观察流程，不参与上述研究会话的样本切分或版本决策。

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
| A 股 | 优先支持 | 当前主要验证市场，支持股票、ETF 与指数历史行情 |
| 港股 | 实验性 | FTShare 数据适配已具备，策略与交易规则仍需逐项验证 |
| 美股 | 实验性 | FTShare 数据适配已具备，策略与交易规则仍需逐项验证 |

策略默认通过 [FTShare MCP](https://market.ft.tech/gateway/mcp) 获取历史行情：`data/fetch_data.py` 调用 `autostrategy.data.ftshare.fetch_daily_ohlc`，返回带 `date` 索引的 OHLCV DataFrame。也可通过环境变量 `AUTOSTRATEGY_FTSHARE_URL` 覆盖网关地址。旧策略仍可显式配置本地回放 feed，但生成器不会自动优先读取 `data/data.csv`。

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
