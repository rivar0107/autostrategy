# FT Client Ten-Minute Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让非凸客户端模拟盘使用沪深 A 股、ETF 与基准指数的已完成 10 分钟行情实时评估策略，并确保后台持续携带策略依赖的基准标的。

**Architecture:** 新增 FTShare 分钟价格 REST 客户端，将交易时段内的一分钟价格聚合为不跨午休的 10 分钟 OHLCV，并仅发布所有策略标的共有的最新已完成桶。模拟服务继续以 3 秒同步账户和订单，以短轮询发现新 K 线，但通过 `last_evaluated_bar_at` 保证每根 10 分钟 K 线最多评估一次；日线历史保留为独立上下文，避免改变 MA250 的含义。

**Tech Stack:** Python 3.11、urllib、pandas、FastAPI、pytest、React 18、TypeScript、Vitest。

---

## File map

- Modify `src/autostrategy/data/ftshare.py`: 增加沪深代码转换、分钟价格 REST 请求和交易时段 10 分钟聚合。
- Modify `src/autostrategy/services/client_simulation_service.py`: 新增 10 分钟行情上下文 provider，保留日线历史，并修复后台基准标的集合。
- Modify `src/autostrategy/api/app.py`: 默认注入 10 分钟 provider。
- Modify `src/autostrategy/api/dependencies.py`: 测试/依赖容器默认注入 10 分钟 provider。
- Modify `tests/unit/test_ftshare.py`: 覆盖代码转换、完整桶、未完成桶和午休边界。
- Modify `tests/unit/test_client_simulation_service.py`: 覆盖共同完成时间、每桶一次评估与后台基准保留。
- Modify `tests/integration/test_api_client_simulation.py`: 更新默认 provider 替换点。
- Modify `src/autostrategy/web/frontend/src/ClientSimulationPanel.tsx`: 把日线运行说明改为 10 分钟实时模拟说明。
- Modify `src/autostrategy/web/frontend/src/ClientSimulationPanel.test.tsx`: 锁定客户可见的 10 分钟提示。
- Modify `docs/superpowers/specs/2026-08-10-ft-client-simulation-design.md`: 将首版日线口径更新为 10 分钟已完成 K 线口径。

### Task 1: FTShare minute prices and ten-minute aggregation

**Files:**
- Modify: `src/autostrategy/data/ftshare.py`
- Test: `tests/unit/test_ftshare.py`

- [ ] **Step 1: Write failing symbol-conversion and aggregation tests**

覆盖 `510500.SH -> 510500.XSHG`、`159915.SZ -> 159915.XSHE`，以及 `09:30-09:39` 聚合为时间戳 `09:40` 的 OHLCV。

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest tests/unit/test_ftshare.py -q`

Expected: 新的分钟价格与聚合函数尚不存在，测试失败。

- [ ] **Step 3: Implement the REST client and completed-bucket aggregation**

请求 `/app/api/v2/{stocks|etfs|indices}/{symbol}/prices?since=TODAY`，使用 `X-Client-Name: ft-web`。只接受 `09:30-11:30`、`13:00-15:00` 的分钟点，分别以半日开盘为锚点聚合，不让桶跨越午休；以当前上海时间判断桶是否完成。

- [ ] **Step 4: Verify aggregation GREEN**

Run: `pytest tests/unit/test_ftshare.py -q`

Expected: 完整桶被返回，当前未完成桶被排除，午休前后数据不混合。

### Task 2: Synchronized ten-minute market context

**Files:**
- Modify: `src/autostrategy/services/client_simulation_service.py`
- Test: `tests/unit/test_client_simulation_service.py`

- [ ] **Step 1: Write failing provider tests**

构造 ETF 与基准指数分钟数据，断言 `bars_by_symbol` 截止到二者共有的最新完成时间，并断言 `history_by_symbol` 仍提供独立日线历史。

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest tests/unit/test_client_simulation_service.py -q`

Expected: `FtshareTenMinuteMarketContextProvider` 尚不存在，测试失败。

- [ ] **Step 3: Implement the provider**

分钟数据聚合写入 `intraday_history_by_symbol` 和最新 `bars_by_symbol`；日线 fetcher 继续写入 `history_by_symbol`，使已有 MA250 策略不改变周期语义。缺少任一标的共同完成桶时返回 `completed_bar_at=None`。

- [ ] **Step 4: Verify provider GREEN**

Run: `pytest tests/unit/test_client_simulation_service.py -q`

Expected: 多标的同步和日线历史测试通过。

### Task 3: Background benchmark and once-per-bar evaluation

**Files:**
- Modify: `src/autostrategy/services/client_simulation_service.py`
- Test: `tests/unit/test_client_simulation_service.py`

- [ ] **Step 1: Write the benchmark regression test**

启动执行标的为 `510500.SH`、基准为 `000905.SH` 的会话，调用 `evaluate_latest_bar()`，断言 provider 收到两个标的。

- [ ] **Step 2: Run the regression test and verify RED**

Run: `pytest tests/unit/test_client_simulation_service.py -q`

Expected: provider 只收到执行标的，断言失败。

- [ ] **Step 3: Use the complete strategy market symbol set**

在后台评估中用 `_strategy_market_symbols(strategy_config, execution_symbols)` 构造行情集合，并维持 `last_evaluated_bar_at` 去重。

- [ ] **Step 4: Verify background evaluation GREEN**

Run: `pytest tests/unit/test_client_simulation_service.py -q`

Expected: 基准始终存在，同一完成桶不重复评估，下一个桶恰好评估一次。

### Task 4: Runtime wiring and customer-facing copy

**Files:**
- Modify: `src/autostrategy/api/app.py`
- Modify: `src/autostrategy/api/dependencies.py`
- Modify: `tests/integration/test_api_client_simulation.py`
- Modify: `src/autostrategy/web/frontend/src/ClientSimulationPanel.tsx`
- Modify: `src/autostrategy/web/frontend/src/ClientSimulationPanel.test.tsx`

- [ ] **Step 1: Write or update failing wiring and UI tests**

断言默认服务使用 10 分钟 provider，面板明确显示“已完成 10 分钟 K 线”，并继续暴露连接地址、FT 账号、密码、客户端版本、模拟账户白名单和标的白名单等客户输入字段。

- [ ] **Step 2: Run backend and frontend focused tests and verify RED**

Run: `pytest tests/integration/test_api_client_simulation.py -q`

Run: `npm test -- --run src/ClientSimulationPanel.test.tsx`

- [ ] **Step 3: Wire the provider and update copy**

默认应用与依赖容器注入 `FtshareTenMinuteMarketContextProvider`，市场轮询保持在 30 至 60 秒内；界面说明信号只在完整 10 分钟桶形成后触发，账户/订单同步仍为约 3 秒。

- [ ] **Step 4: Verify focused tests GREEN**

重复运行上一步命令，预期全部通过。

### Task 5: Design document and full verification

**Files:**
- Modify: `docs/superpowers/specs/2026-08-10-ft-client-simulation-design.md`

- [ ] **Step 1: Update the design contract**

将“上一根已完成日线”更新为“最新共有已完成 10 分钟 K 线”，补充午休分桶、共同时间戳、每桶一次、日线研究历史独立保留和分钟 REST 来源。

- [ ] **Step 2: Run complete verification**

Run: `pytest -q`

Run: `ruff check src tests`

Run: `npm test -- --run`

Run: `npm run typecheck`

Run: `npm run build`

Run: `git diff --check`

- [ ] **Step 3: Preserve the staged design document and restart the app**

只重新暂存 `docs/superpowers/specs/2026-08-10-ft-client-simulation-design.md`，重启 8000 端口后检查 `/api/health` 和前端页面。
