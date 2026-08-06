# Phase 5: Paper Trading / 模拟盘设计

**日期:** 2026-07-06  
**分支:** `docs/phase5-paper-trading-plan`  
**定位:** Phase 5 将 Phase 4 已经打通的本地 API、Web 工作台、任务系统，推进到可观察的模拟运行闭环。当前目标是 replay-first paper trading，不接真实 broker，不触发真实下单。

---

## 1. 背景

Autostrategy 已经完成：

1. 策略工作区与模板
2. AI 设计文档生成
3. 策略代码生成
4. 本地回测
5. REST API 与 Web 工作台

回测解决的是“历史上这套规则表现如何”。Phase 5 要解决的是“如果按时间顺序推进，策略每一步会如何决策、产生什么事件、用户如何观察和复盘”。

这个阶段不能直接跳到实盘交易。实盘会引入资金安全、broker API、网络失败、订单状态、合规与风控。对当前产品来说，正确顺序是先把本地模拟盘做扎实：数据可重放、事件可审计、状态可解释、UI 可观察。

---

## 2. 产品目标

Phase 5 的用户目标：

- 用户可以从 Web 或 API 启动一次模拟运行。
- 用户可以看到 replay 进度，而不是只能等最终 JSON。
- 用户可以复盘每一步策略决策和事件流。
- 用户可以在后续阶段看到虚拟账户、持仓、权益曲线和订单记录。
- 用户清楚知道这不是实盘，也不会连接真实 broker。

Phase 5 的工程目标：

- 复用现有 strategy workspace 与 job service。
- 让 `run_paper(config)` 成为策略模拟运行入口。
- 将模拟运行结果写到稳定 artifact 路径。
- 为后续虚拟账户、mock feed、Optimization Agent、Learning Agent 留出数据结构。
- 保持本地执行边界，不引入远程多用户系统。

---

## 3. 非目标

Phase 5 暂不做：

- 真实 broker 接入
- 实盘下单
- 多用户账户系统
- 托管式远程模拟盘服务
- 高频实时行情处理
- 撮合引擎级别的订单簿模拟
- 收益归因、自动调参、学习 Agent 的完整闭环

这些能力可以出现在 Phase 6 或更后阶段，但不能混进 Phase 5 的最小闭环。原因很简单：资金安全和真实交易链路的复杂度远高于 replay-first 模拟盘，过早引入会把产品主线打散。

---

## 4. 分阶段范围

### 4.1 Phase 5A: replay-first 模拟运行最小闭环

**状态:** 已完成  
**完成日期:** 2026-07-05  
**目标:** 策略暴露 `run_paper(config)` 后，可以通过 API/Web 触发模拟运行，并产出标准 artifacts。

核心能力：

- 后端支持 `POST /api/v1/strategies/{slug}/paper-run`。
- `PaperRunJobService` 后台执行模拟运行任务。
- `run_paper_replay_workflow()` 调用策略代码里的 `run_paper(config)`。
- 输出：
  - `paper_run/results/paper_run_result.json`
  - `paper_run/results/paper_run_events.jsonl`
  - `paper_run/logs/paper_run.log`
- 前端策略详情页可以启动 paper run 并查看结果。

验收标准：

- 单元测试覆盖 workflow 成功、缺少入口、停止请求。
- 集成测试覆盖 API 启动 paper run。
- README 说明用户如何启动模拟运行。

### 4.2 Phase 5B: 准实时 replay

**状态:** 已完成  
**完成日期:** 2026-07-06  
**目标:** 支持策略按事件逐步产出 replay 状态，前端可以观察进度和事件，而不是只读取最终结果。

核心能力：

- `run_paper(config)` 可以返回普通结果，也可以返回增量事件迭代器。
- workflow 能在 replay 过程中刷新结果文件。
- 支持 stop requested 后停止增量 replay。
- 前端展示 replay progress、current time、bars processed、events。

验收标准：

- 单元测试覆盖增量结果刷新。
- 单元测试覆盖增量 replay 停止。
- 集成测试覆盖 API paper run 的结果结构。
- 前端测试覆盖 paper run 状态展示。

### 4.3 Phase 5C: 虚拟账户与持仓模型

**状态:** 未开始  
**目标:** 让 paper run 不只是 replay 日志，而是维护虚拟账户状态。

计划能力：

- 维护 `cash`、`positions`、`equity`、`realized_pnl`、`unrealized_pnl`。
- 支持策略输出 `buy`、`sell`、`hold` 决策。
- 将决策转换为虚拟成交事件。
- 将账户快照写入事件流和最终结果。
- 在前端展示虚拟账户摘要与持仓表。

建议 artifact 结构：

```json
{
  "paper": {
    "initial_cash": 1000000,
    "final_value": 1012000,
    "cash": 820000,
    "equity": 1012000,
    "positions": [
      {
        "symbol": "000001.SZ",
        "quantity": 1000,
        "avg_price": 12.3,
        "market_value": 12500,
        "unrealized_pnl": 200
      }
    ]
  },
  "replay": {
    "progress": 1.0,
    "bars_processed": 240,
    "current_at": "2024-12-31"
  }
}
```

### 4.4 Phase 5D: 本地 mock 行情 feed

**状态:** 未开始  
**目标:** 将 replay 数据源从策略内部自带逻辑，推进到可替换的本地行情 feed。

计划能力：

- 支持从 CSV/JSON/Parquet 加载历史行情。
- 支持按时间窗口推进。
- 支持多 symbol replay。
- 统一 bar event 格式，供策略与 paper engine 消费。
- 避免依赖外部网络，测试使用 fixture 数据。

建议事件格式：

```json
{
  "type": "bar",
  "at": "2024-01-02T09:30:00+08:00",
  "symbol": "000001.SZ",
  "open": 10.0,
  "high": 10.5,
  "low": 9.8,
  "close": 10.2,
  "volume": 123456
}
```

### 4.5 Phase 5E: 复盘与优化前置

**状态:** 规划中  
**目标:** 为 Learning Agent 和 Optimization Agent 准备可读、可比、可追踪的模拟运行数据。

计划能力：

- 生成 paper run summary。
- 记录关键风险指标：最大回撤、胜率、换手、交易次数。
- 支持对比两次 paper run。
- 将事件流输入策略复盘 prompt。
- 不在本阶段自动改策略，只提供优化输入。

---

## 5. 系统边界

### 5.1 策略入口

Phase 5 使用策略工作区中的 `strategy.py` 作为执行入口。

约定：

```python
def run_paper(config):
    ...
```

允许两种返回形式：

1. 最终结果字典
2. 增量事件迭代器，每个事件包含 replay/paper 局部状态

这样做的好处是简单，策略作者不需要理解完整 engine 生命周期。代价是接口早期偏宽松，后续 Phase 5C/5D 需要收敛事件 schema。

### 5.2 Job Service

Paper run 继续复用后台 job 模式：

- API 创建 job
- 后台执行策略
- 前端轮询 job 状态和结果
- artifacts 写入 strategy workspace

暂不引入队列系统。当前是本地单用户工具，in-memory job 足够。等需要多进程、多用户、远程部署时，再引入持久化队列。

### 5.3 Artifact 契约

稳定路径：

```text
paper_run/results/paper_run_result.json
paper_run/results/paper_run_events.jsonl
paper_run/logs/paper_run.log
```

约定：

- `paper_run_result.json` 保存最新 summary，可被前端反复读取。
- `paper_run_events.jsonl` 保存事件流，一行一个 JSON 事件。
- `paper_run.log` 保存执行日志和错误信息。

### 5.4 安全边界

Phase 5 仍然执行本地策略代码，因此必须延续 Phase 4 的安全边界：

- workspace path 必须限制在策略目录内。
- AI 生成代码继续拒绝危险模式。
- API 默认监听 `127.0.0.1`。
- 不保存 broker token。
- 不提供真实交易接口。
- 文案中明确“模拟运行不是实盘交易”。

---

## 6. 用户体验

Web 工作台中的 Phase 5 体验应该按这个顺序演进：

1. 策略详情页显示 Paper Run 操作区。
2. 用户点击启动模拟运行。
3. UI 展示 job 状态：queued/running/succeeded/failed/stopped。
4. UI 展示 replay 进度、当前时间、处理 bar 数量。
5. UI 展示事件列表。
6. Phase 5C 后展示虚拟账户和持仓。
7. Phase 5E 后展示复盘摘要和优化建议入口。

关键原则：用户必须能看懂“现在发生到哪一步了”。模拟盘的价值不是最终 JSON，而是过程可解释。

---

## 7. 测试策略

Phase 5 影响策略执行、任务状态、文件 artifacts 和前端展示，测试要分层：

### 7.1 单元测试

覆盖：

- `run_paper_replay_workflow()` 成功写结果
- 缺少 `run_paper` 时失败
- stop requested 行为
- 增量 replay 刷新结果
- 增量 replay 停止
- 后续账户模型的买入、卖出、现金不足、持仓不足

### 7.2 集成测试

覆盖：

- `POST /api/v1/strategies/{slug}/paper-run`
- job 状态轮询
- paper run result 读取
- 错误响应结构

### 7.3 前端测试

覆盖：

- 策略详情页启动 paper run
- running/succeeded/failed 状态展示
- replay progress 和事件展示
- 后续账户摘要和持仓表展示

### 7.4 fixture 策略

测试策略应保持小而确定：

- 返回固定最终结果
- 产出两个增量 replay 事件
- 支持 stop requested 后可预测停止
- 不依赖外部网络和真实行情 API

---

## 8. 风险与取舍

| 风险 | 影响 | 取舍 |
|------|------|------|
| `run_paper(config)` 接口过宽松 | 后续策略事件格式可能分叉 | 先跑通闭环，Phase 5C 收敛 schema |
| 本地执行策略代码 | 可能执行用户不理解的代码 | 延续危险模式检查、路径限制和本地安全提示 |
| 轮询 job 状态 | 不如 SSE 实时 | 当前简单可靠，准实时 replay 已足够验证产品价值 |
| 无真实 broker | 不能验证真实订单生命周期 | 明确非目标，避免资金风险过早进入 |
| mock feed 还未完成 | 数据源仍偏策略自带 | Phase 5D 单独补 feed 层 |

---

## 9. 验收总表

| 阶段 | 状态 | 验收标准 |
|------|------|----------|
| Phase 5A replay-first 最小闭环 | 已完成 | API/Web 可启动 paper run，artifacts 写入稳定路径 |
| Phase 5B 准实时 replay | 已完成 | 增量 replay 可刷新结果，前端可展示进度和事件 |
| Phase 5C 虚拟账户与持仓 | 未开始 | paper run 维护 cash/positions/equity，并展示账户摘要 |
| Phase 5D 本地 mock 行情 feed | 未开始 | 可从本地历史行情按时间窗口推进 replay |
| Phase 5E 复盘与优化前置 | 规划中 | 可生成复盘摘要，并为优化 Agent 提供输入 |

---

## 10. 结论

Phase 5 的核心不是“假装实盘”，而是把策略从一次性回测推进到可观察、可复盘、可迭代的模拟运行。当前 5A/5B 已经证明 replay-first 路径成立。下一步应优先补 Phase 5C 的虚拟账户与持仓模型，再做 Phase 5D 的本地 mock 行情 feed。这样用户会从“看一个回测分数”进化到“看策略在时间里如何做决定”。
