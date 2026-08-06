# Phase 5: Paper Trading / 模拟盘执行计划

**日期:** 2026-07-06  
**分支:** `docs/phase5-paper-trading-plan`  
**对应设计:** `docs/superpowers/specs/2026-07-06-phase5-paper-trading-design.md`  
**定位:** 补齐 Phase 5 的计划沉淀，记录已完成的 5A/5B，并给 5C/5D/5E 后续实现提供明确路径。

---

## 1. 成功标准

Phase 5 完成后，Autostrategy 应满足：

1. 用户可以从 Web/API 启动模拟运行。
2. 模拟运行支持 replay-first 和准实时进度展示。
3. 每次运行产出稳定 artifacts，方便前端展示和后续复盘。
4. paper run 维护虚拟账户、持仓和权益状态。
5. replay 数据可以来自本地 mock 行情 feed。
6. 测试覆盖核心 workflow、API、账户模型和前端状态展示。
7. 产品文案清楚说明：模拟运行不是实盘交易，不连接真实 broker。

---

## 2. 当前状态

### 已完成

- Phase 5A: replay-first 模拟运行最小闭环
- Phase 5B: 准实时 replay

### 未完成

- Phase 5C: 虚拟账户与持仓模型
- Phase 5D: 本地 mock 行情 feed
- Phase 5E: 复盘与优化前置

### 当前待办对应关系

`TODOS.md` 中仍有两项 Phase 5 待办：

- 虚拟账户与持仓模型 → Phase 5C
- 本地 mock 行情 feed → Phase 5D

---

## 3. Phase 5A 验收记录: replay-first 最小闭环

**状态:** 已完成  
**完成日期:** 2026-07-05  
**提交:** `bd9f599 feat: Phase 5A replay-first 模拟运行闭环`

### 3.1 交付内容

- `run_paper_replay_workflow()` 支持调用策略 `run_paper(config)`。
- paper run artifacts 写入：
  - `paper_run/results/paper_run_result.json`
  - `paper_run/results/paper_run_events.jsonl`
  - `paper_run/logs/paper_run.log`
- API 支持启动 paper run job。
- Web 工作台可以触发 paper run。
- 测试覆盖 workflow 与 API。

### 3.2 验收标准

- [x] 策略有 `run_paper(config)` 时，workflow 能成功写入结果。
- [x] 策略缺少 `run_paper(config)` 时，workflow 返回失败状态。
- [x] stop requested 可以中断运行。
- [x] API paper run 集成测试通过。
- [x] 前端可触发 paper run。

### 3.3 验证命令

```bash
python -m pytest tests/unit/test_paper_run_workflow.py tests/unit/test_paper_run_job_service.py tests/integration/test_api_paper_run.py -q
```

---

## 4. Phase 5B 验收记录: 准实时 replay

**状态:** 已完成  
**完成日期:** 2026-07-06  
**提交:** `fe8675f feat: 添加 Phase 5B 准实时 replay`

### 4.1 交付内容

- `run_paper(config)` 支持增量 replay 输出。
- workflow 在 replay 过程中刷新 `paper_run_result.json`。
- workflow 将 replay event 追加到 `paper_run_events.jsonl`。
- 支持增量 replay 过程中的 stop requested。
- 前端展示 replay progress、current time、bars processed 和事件列表。
- README 和 TODOS 更新 Phase 5 状态。

### 4.2 验收标准

- [x] 增量 replay 会刷新最终结果文件。
- [x] 增量 replay 会写事件流。
- [x] stop requested 后能停止增量 replay。
- [x] API 返回可供前端展示的 replay 状态。
- [x] 前端测试覆盖 paper run 状态展示。

### 4.3 验证命令

```bash
python -m pytest tests/unit/test_paper_run_workflow.py tests/integration/test_api_paper_run.py -q
npm test -- --run src/App.test.tsx
```

---

## 5. Phase 5C 计划: 虚拟账户与持仓模型

**状态:** 未开始  
**优先级:** P2  
**建议先做:** 是。它是 mock feed 和复盘能力之前最关键的产品增量。

### 5.1 用户故事

作为策略研究者，我希望模拟运行能显示现金、持仓、权益和盈亏，这样我不只是看到策略输出了什么信号，还能知道这些信号在虚拟账户里造成了什么结果。

### 5.2 实现步骤

1. **定义 paper account 数据结构**
   - 新增或扩展核心模型，包含 cash、positions、equity、realized_pnl、unrealized_pnl。
   - positions 至少包含 symbol、quantity、avg_price、market_value、unrealized_pnl。
   - 验证：新增单元测试覆盖初始化账户。

2. **定义 order/decision 事件 schema**
   - 支持 buy、sell、hold。
   - 输入包含 symbol、quantity 或 target_percent、price、at。
   - 输出标准 event，写入 `paper_run_events.jsonl`。
   - 验证：测试 buy/sell/hold 三类事件序列化。

3. **实现虚拟成交规则**
   - buy: 现金足够时增加持仓、减少 cash。
   - sell: 持仓足够时减少持仓、增加 cash。
   - hold: 只记录决策，不改变账户。
   - 先使用 bar close price 成交，不做滑点和手续费，除非 config 显式传入。
   - 验证：测试现金不足、持仓不足、正常买卖。

4. **接入 `run_paper_replay_workflow()`**
   - workflow 从 replay event 中识别决策。
   - 更新账户快照。
   - 每个事件后刷新 summary。
   - 验证：测试 replay 两个 bar 后账户状态正确。

5. **更新 API response 与前端展示**
   - paper result 中增加 account summary。
   - 前端展示 cash、equity、positions 表格。
   - 验证：前端测试覆盖账户摘要和空持仓状态。

6. **更新 README/TODOS**
   - README 补充虚拟账户输出说明。
   - TODOS 将“虚拟账户与持仓模型”移动到 Completed。
   - 验证：人工检查文档链接和示例路径。

### 5.3 验收清单

- [ ] paper run result 包含 cash、positions、equity。
- [ ] buy/sell/hold 事件写入 events jsonl。
- [ ] cash 不足不会产生非法负现金，除非未来显式支持融资。
- [ ] sell 超过持仓不会产生非法负持仓，除非未来显式支持融券。
- [ ] 前端能展示账户摘要和持仓表。
- [ ] 单元测试覆盖正常路径和边界情况。
- [ ] 集成测试覆盖 API 结果结构。

---

## 6. Phase 5D 计划: 本地 mock 行情 feed

**状态:** 未开始  
**优先级:** P2  
**建议顺序:** Phase 5C 之后。

### 6.1 用户故事

作为策略研究者，我希望 replay 数据来自统一的本地行情 feed，而不是每个策略自己随意生成数据，这样同一组历史数据可以复用、测试结果更稳定、后续也能接入真实数据源。

### 6.2 实现步骤

1. **定义 bar event schema**
   - 字段包含 at、symbol、open、high、low、close、volume。
   - 时间使用 ISO 8601 字符串。
   - 验证：schema 单元测试覆盖必填字段和排序。

2. **实现本地 fixture feed**
   - 从 CSV 或 JSONL 读取 bars。
   - 支持按 start/end 时间过滤。
   - 支持单 symbol 和多 symbol。
   - 验证：测试空文件、单 symbol、多 symbol、时间窗口。

3. **接入 paper run config**
   - config 支持指定 feed path、start、end、symbols。
   - 默认仍允许策略自带 replay，避免一次性打断现有示例。
   - 验证：测试 config 指定 feed 后 workflow 使用 feed events。

4. **更新示例策略**
   - 提供一个使用本地 feed 的示例策略。
   - 不依赖外部网络。
   - 验证：示例 paper run 可稳定复现。

5. **前端展示 feed 元信息**
   - 展示 replay 数据来源、时间范围、symbol 数量。
   - 验证：前端测试覆盖 feed metadata。

### 6.3 验收清单

- [ ] 本地 feed 可按时间窗口输出 bar events。
- [ ] workflow 可消费 feed events。
- [ ] 测试 fixture 不依赖外部网络。
- [ ] README 说明如何准备本地 replay 数据。
- [ ] 前端显示数据源和 replay 时间范围。

---

## 7. Phase 5E 计划: 复盘与优化前置

**状态:** 规划中  
**优先级:** P3  
**建议顺序:** Phase 5C/5D 之后。

### 7.1 用户故事

作为策略研究者，我希望模拟运行结束后得到一份复盘摘要，知道策略在哪些时点做了关键决策、收益和回撤来自哪里，以及下一步该优化什么。

### 7.2 实现步骤

1. **生成 paper run summary**
   - 汇总收益、回撤、交易次数、胜率、换手。
   - 验证：指标计算单元测试。

2. **生成关键事件摘要**
   - 从 events jsonl 中提取买入、卖出、止损、异常事件。
   - 验证：事件摘要测试。

3. **为 Learning Agent 准备输入**
   - 输出结构化 markdown/json。
   - 暂不自动调用 LLM 修改策略。
   - 验证：snapshot 测试保证格式稳定。

4. **前端展示复盘入口**
   - 展示 summary。
   - 提供“复制复盘上下文”或后续“生成优化建议”的入口。
   - 验证：前端测试覆盖 summary 展示。

### 7.3 验收清单

- [ ] 每次 paper run 生成 summary。
- [ ] summary 可被人直接阅读。
- [ ] summary 可作为 Learning Agent 输入。
- [ ] 不自动修改策略代码。

---

## 8. 文件影响范围

预计后续 Phase 5C/5D 会触达：

- `src/autostrategy/core/backtest_engine.py`
- `src/autostrategy/services/paper_run_job_service.py`
- `src/autostrategy/api/routers/strategies.py`
- `src/autostrategy/web/frontend/src/App.tsx`
- `tests/unit/test_paper_run_workflow.py`
- `tests/unit/test_paper_run_job_service.py`
- `tests/integration/test_api_paper_run.py`
- `src/autostrategy/web/frontend/src/App.test.tsx`
- `README.md`
- `TODOS.md`

如果账户模型或 feed 逻辑开始变大，应新建独立模块，例如：

- `src/autostrategy/core/paper_account.py`
- `src/autostrategy/core/paper_feed.py`

不要继续把所有逻辑塞进 `backtest_engine.py`。`backtest_engine.py` 可以保留 workflow 编排，但账户状态和 feed 解析应该独立测试。

---

## 9. 验证策略

每个后续子阶段至少运行：

```bash
python -m pytest tests/unit/test_paper_run_workflow.py tests/unit/test_paper_run_job_service.py tests/integration/test_api_paper_run.py -q
```

如果改动前端：

```bash
npm test -- --run src/App.test.tsx
```

如果改动 README 或用户流程，还要手工启动本地服务验证主路径：

```bash
autostrategy serve --host 127.0.0.1 --port 8000
```

然后在浏览器中验证：

1. 打开策略列表。
2. 进入策略详情。
3. 启动 paper run。
4. 查看 job 状态、replay 进度、events、账户摘要。

---

## 10. 里程碑顺序建议

建议按这个顺序做，避免过早复杂化：

1. **Phase 5C.1:** 账户模型纯单元测试，不接 API。
2. **Phase 5C.2:** workflow 接账户模型，写 result/events。
3. **Phase 5C.3:** API/前端展示账户摘要。
4. **Phase 5D.1:** 本地 fixture feed。
5. **Phase 5D.2:** workflow 消费 feed。
6. **Phase 5E.1:** paper run summary。
7. **Phase 5E.2:** Learning/Optimization 输入格式。

这条路径的好处是每一步都能单独验收。用户每次都能看到产品变得更有用，而不是等一个大而全的“模拟盘系统”最后才露面。

---

## 11. 本次补文档验收

本次只补设计与计划文档，不改运行时代码。

验收标准：

- [x] 新增 Phase 5 设计文档。
- [x] 新增 Phase 5 执行计划。
- [x] 明确 5A/5B 已完成状态。
- [x] 明确 5C/5D/5E 后续范围。
- [x] 对齐 `TODOS.md` 中剩余 Phase 5 待办。

验证命令：

```bash
git diff -- docs/superpowers/specs/2026-07-06-phase5-paper-trading-design.md docs/superpowers/plans/2026-07-06-phase5-paper-trading-plan.md
```
