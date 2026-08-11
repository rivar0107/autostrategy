# FT Client China A-Market Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将非凸客户端模拟盘从两只固定 ETF 扩展为沪深市场可交易 A 股和 ETF，并让客户在前端确认每个策略的实际执行标的与客户端代码映射。

**Architecture:** 新增统一的沪深证券分类器，产品级能力允许沪深 A 股和 ETF，但继续用会话级 `allowed_symbols + symbol_mapping` 控制策略实际可下单范围。前端改为动态标的列表和映射 JSON；预检验证市场范围、映射完整性和策略声明标的，订单风控再次校验，FTShare 按股票/ETF 分类取数。指数、债券、B 股、北交所和非 A 股市场继续硬阻断。

**Tech Stack:** Python 3.11+、Pydantic 2、FastAPI、React 18、TypeScript、Ant Design、pytest、Vitest。

---

### Task 1: 沪深 A 股/ETF 证券分类

**Files:**
- Create: `src/autostrategy/brokers/cn_symbols.py`
- Modify: `src/autostrategy/config.py`
- Test: `tests/unit/test_cn_symbols.py`

- [ ] **Step 1: 写失败测试**

```python
def test_supports_sh_sz_a_shares_and_etfs_only():
    assert classify_cn_security("600519.SH") == "stock"
    assert classify_cn_security("300750.SZ") == "stock"
    assert classify_cn_security("510500.SH") == "etf"
    assert classify_cn_security("159915.SZ") == "etf"
    assert classify_cn_security("000905.SH") is None
    assert classify_cn_security("200002.SZ") is None
    assert classify_cn_security("430047.BJ") is None
```

- [ ] **Step 2: 运行 RED**

Run: `.venv/bin/python -m pytest tests/unit/test_cn_symbols.py -q`
Expected: `autostrategy.brokers.cn_symbols` 不存在。

- [ ] **Step 3: 实现分类器与空的默认执行白名单**

实现 `classify_cn_security(symbol) -> Literal["stock", "etf"] | None`、`is_supported_cn_security` 和 `ftshare_asset_type`。`FtClientConfig.allowed_symbols` 默认改为空列表，禁止把两个历史 ETF 当成产品硬编码。

- [ ] **Step 4: 运行 GREEN**

Run: `.venv/bin/python -m pytest tests/unit/test_cn_symbols.py tests/unit/test_config.py -q`
Expected: 全部通过。

### Task 2: 动态执行标的预检与风控

**Files:**
- Modify: `src/autostrategy/services/client_simulation_service.py`
- Test: `tests/unit/test_client_simulation_service.py`
- Test: `tests/integration/test_api_client_simulation.py`

- [ ] **Step 1: 写失败测试**

```python
def test_preflight_accepts_a_share_and_etf_execution_universe(...):
    config = _ft_config(
        allowed_symbols=["600519.SH", "510500.SH"],
        symbol_mapping={"600519.SH": "600519.SH", "510500.SH": "510500.SH"},
    )
    assert service.preflight(slug, request).ready is True

def test_preflight_rejects_index_and_missing_mapping(...):
    assert "unsupported_market_symbol" in failed_codes
    assert "symbol_mapping_confirmed" in failed_codes
```

- [ ] **Step 2: 运行 RED**

Run: `.venv/bin/python -m pytest tests/unit/test_client_simulation_service.py -q`
Expected: `510500.SH` 或 A 股标的被旧的两 ETF 限制拒绝。

- [ ] **Step 3: 实现动态会话白名单**

预检要求 `allowed_symbols` 非空、全部属于沪深 A 股/ETF、映射完整；策略明确声明的可交易标的必须是该集合子集。行情/基准指数可进入数据上下文，但不得进入订单意图。`_risk_rejection` 对每个意图再次执行市场分类、会话白名单和映射校验。

- [ ] **Step 4: 验证 GREEN**

Run: `.venv/bin/python -m pytest tests/unit/test_client_simulation_service.py tests/integration/test_api_client_simulation.py -q`
Expected: 动态 A 股/ETF 通过，指数/B 股/北交所失败。

### Task 3: FTShare 股票/ETF 分类取数

**Files:**
- Modify: `src/autostrategy/services/client_simulation_service.py`
- Test: `tests/unit/test_client_simulation_service.py`

- [ ] **Step 1: 写失败测试**

```python
def test_market_provider_routes_stock_and_etf_to_correct_ftshare_type():
    provider(["600519.SH", "510500.SH"], {})
    assert calls["600519.SH"]["type_"] == "stock"
    assert calls["510500.SH"]["type_"] == "etf"
```

- [ ] **Step 2: 运行 RED、实现分类路由并运行 GREEN**

Run: `.venv/bin/python -m pytest tests/unit/test_client_simulation_service.py::test_market_provider_routes_stock_and_etf_to_correct_ftshare_type -q`
Expected before: 两者都使用 `etf`；after: 测试通过。

### Task 4: 前端动态标的与代码映射

**Files:**
- Modify: `src/autostrategy/web/frontend/src/ClientSimulationPanel.tsx`
- Modify: `src/autostrategy/web/frontend/src/types.ts`
- Test: `src/autostrategy/web/frontend/src/ClientSimulationPanel.test.tsx`

- [ ] **Step 1: 写失败测试**

```tsx
expect(screen.getByLabelText('本次策略允许标的')).toBeInTheDocument()
expect(screen.getByLabelText('客户端代码映射')).toBeInTheDocument()
expect(screen.queryByLabelText('588000.SH 客户端代码')).not.toBeInTheDocument()
```

- [ ] **Step 2: 运行 RED**

Run: `npm test -- --run ClientSimulationPanel.test.tsx`
Expected: 动态字段不存在。

- [ ] **Step 3: 实现动态输入与校验**

“本次策略允许标的”使用逗号/换行输入；“客户端代码映射”使用结构化 JSON。前端校验证券代码格式、映射键与允许标的一致，并将结果写入现有 `allowed_symbols`、`symbol_mapping` 请求字段。

- [ ] **Step 4: 运行 GREEN**

Run: `npm test -- --run ClientSimulationPanel.test.tsx && npm run typecheck`
Expected: 全部通过。

### Task 5: 策略生成兼容本地回放与客户端模拟盘

**Files:**
- Modify: `src/autostrategy/agents/prompts/codegen.py`
- Modify: `src/autostrategy/agents/codegen_agent.py`
- Test: `tests/integration/test_codegen_agent.py`

- [ ] **Step 1: 写失败测试**

```python
assert "def run_paper(config: dict)" in CODEGEN_SYSTEM_PROMPT
assert "def generate_intents(context: dict) -> dict" in CODEGEN_SYSTEM_PROMPT
```

- [ ] **Step 2: 运行 RED、补齐双入口契约并运行 GREEN**

Run: `.venv/bin/python -m pytest tests/integration/test_codegen_agent.py -q`
Expected before: `run_paper` 断言失败；after: 两种模拟入口均为生成验收项。

### Task 6: 文档与全量验证

**Files:**
- Modify: `docs/superpowers/specs/2026-08-10-ft-client-simulation-design.md`
- Modify: `README.md`

- [ ] **Step 1: 更新范围与客户操作说明**

删除两只 ETF 的产品硬编码，明确“产品级全沪深 A 股/ETF + 会话级客户确认标的白名单”，并说明指数仅可做行情基准、不能下单。

- [ ] **Step 2: 完整验证**

Run: `.venv/bin/python -m pytest tests/unit tests/integration -q`

Run: `.venv/bin/ruff check src tests`

Run: `npm test -- --run && npm run typecheck && npm run build`

Run: `git diff --check`

Expected: 全部退出码为 0，无凭证或临时 PDF 进入仓库。
