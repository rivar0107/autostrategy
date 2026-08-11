# FT Client Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留本地历史回放的前提下，让 autostrategy 能通过非凸智能交易终端 API v0.0.23 在白名单模拟账户上观察、人工确认或自动执行策略。

**Architecture:** 新增独立的 broker 领域层和 `FtClientBroker`，将非凸 v1 HTTP/WebSocket 响应归一化；`ClientSimulationService` 负责预检、策略意图、风控、幂等、持久化和会话生命周期，FastAPI 只暴露脱敏 DTO。前端把“本地回放”和“非凸客户端模拟盘”明确分开，客户端事实不会写入现有 `PaperAccount`。

**Tech Stack:** Python 3.11、Pydantic 2、FastAPI、httpx、websockets、pytest、React 18、TypeScript、Ant Design、Vitest。

---

## File map

- Create `src/autostrategy/brokers/models.py`: 账户、资金、持仓、母单、子单、监控、意图及状态归一化模型。
- Create `src/autostrategy/brokers/base.py`: broker 能力协议。
- Create `src/autostrategy/brokers/ft_client.py`: API v0.0.23 HTTP/WebSocket 适配器、认证、脱敏、请求映射。
- Create `src/autostrategy/services/client_simulation_service.py`: 预检、风险、会话、幂等、artifacts 和恢复。
- Create `src/autostrategy/api/routers/client_simulation.py`: `/api/v1` 客户端模拟盘路由。
- Create `src/autostrategy/web/frontend/src/ClientSimulationPanel.tsx`: 非凸模拟盘工作台。
- Modify `src/autostrategy/config.py`: 非凸连接安全配置。
- Modify `src/autostrategy/agents/prompts/codegen.py`: 要求生成纯计算 `generate_intents(context)`。
- Modify `src/autostrategy/api/{app,dependencies,errors,schemas}.py`: 注册服务、路由和 DTO。
- Modify `src/autostrategy/services/__init__.py`: 导出新服务。
- Modify `src/autostrategy/web/frontend/src/{types.ts,api/client.ts,StrategyWorkbench.tsx,App.css}`: 接入面板。
- Modify `pyproject.toml`: 增加 HTTP/WebSocket 客户端依赖。
- Create `tests/unit/test_ft_client_broker.py`: 适配器契约测试。
- Create `tests/unit/test_client_simulation_service.py`: 预检、风险、幂等、生命周期和持久化测试。
- Create `tests/integration/test_api_client_simulation.py`: REST 端到端测试。
- Create `src/autostrategy/web/frontend/src/ClientSimulationPanel.test.tsx`: 前端关键门禁测试。

### Task 1: Configuration and broker domain models

**Files:**
- Create: `src/autostrategy/brokers/__init__.py`
- Create: `src/autostrategy/brokers/models.py`
- Create: `src/autostrategy/brokers/base.py`
- Modify: `src/autostrategy/config.py`
- Test: `tests/unit/test_ft_client_broker.py`

- [ ] **Step 1: Write failing configuration and status tests**

```python
def test_ft_config_rejects_non_loopback_base_url():
    with pytest.raises(ValidationError):
        FtClientConfig(enabled=True, base_url="https://broker.example.com")

def test_order_status_namespaces_are_independent():
    assert normalize_algorithm_child_status(3) == "cancelled"
    assert normalize_direct_order_status(3) == "filled"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/unit/test_ft_client_broker.py -q`
Expected: collection fails because `autostrategy.brokers` does not exist.

- [ ] **Step 3: Implement strict config and models**

```python
class FtClientConfig(BaseModel):
    enabled: bool = False
    base_url: str = "http://127.0.0.1:11356"
    min_client_version: str = "3.11.4"
    confirmed_client_version: str | None = None
    ft_account_env: str = "AUTOSTRATEGY_FT_ACCOUNT"
    password_env: str = "AUTOSTRATEGY_FT_PASSWORD"
    password_transform: Literal["plain", "md5_32_lower"] = "plain"
    allowed_simulation_accounts: list[str] = Field(default_factory=list)
    allowed_symbols: list[str] = Field(default_factory=lambda: ["588000.SH", "563300.SH"])
    symbol_mapping: dict[str, str] = Field(default_factory=dict)
    allowed_algorithms: list[str] = Field(default_factory=lambda: ["TWAP"])
    external_id_max_length: int | None = None
    external_id_scope_confirmed: bool = False
    poll_interval_seconds: float = 3.0
    auto_resume: bool = False
```

Implement complete mother 0-11/21-25, algorithm child 0-9, and direct order 0-6 mappings while preserving `raw_status`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest tests/unit/test_ft_client_broker.py -q`
Expected: configuration and model tests pass.

### Task 2: FT Client v0.0.23 adapter

**Files:**
- Create: `src/autostrategy/brokers/ft_client.py`
- Modify: `pyproject.toml`
- Test: `tests/unit/test_ft_client_broker.py`

- [ ] **Step 1: Write failing adapter tests**

```python
def test_login_uses_account_level_login_status_and_redacts_password(fake_transport):
    broker = FtClientBroker(config, transport=fake_transport)
    accounts = broker.connect()
    assert accounts[0].login_status is False
    assert "secret" not in repr(fake_transport.requests)

def test_submit_parent_uses_external_id_and_safe_algorithm_params(fake_transport):
    order = broker.submit_order(intent)
    assert fake_transport.last_json["mudans"][0]["external_id"] == intent.intent_id
    assert fake_transport.last_json["mudans"][0]["algo_param"] == "delay_end_time=10"
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest tests/unit/test_ft_client_broker.py -q`
Expected: `FtClientBroker` import or behavior assertions fail.

- [ ] **Step 3: Implement the v1 adapter**

Implement `connect`, `disconnect`, `health`, `get_funds`, `get_positions`, `submit_order`, `cancel_orders`, `get_orders`, `get_child_orders`, `get_monitoring`, and `stream_events`. Use `ft_acc_login`, `query_acc_status`, `get_fund_by_acc`, `get_position_by_acc`, `api_upload_mudan`, `op_batch_mudan`, `get_mudan_by_acc`, `get_zidan_by_mudan_id`, `get_zidan_by_acc`, and `get_algo_monitoring_info`. Retry only idempotent queries; never blind-retry mother-order POST. Ping messages must produce Pong with identical data.

- [ ] **Step 4: Verify adapter GREEN**

Run: `pytest tests/unit/test_ft_client_broker.py -q`
Expected: parsing, password transform, URL redaction, field mapping, monitoring and state tests pass.

### Task 3: Session service, preflight, risk, and idempotency

**Files:**
- Create: `src/autostrategy/services/client_simulation_service.py`
- Modify: `src/autostrategy/services/__init__.py`
- Test: `tests/unit/test_client_simulation_service.py`

- [ ] **Step 1: Write failing service tests**

```python
def test_preflight_blocks_unverified_client_and_failed_account_login(service):
    result = service.preflight("grid", request)
    assert {item.code for item in result.failed} == {
        "client_version_unverified", "account_login_failed"
    }

def test_submission_unknown_reconciles_by_external_id_without_retry(service, broker):
    session = service.start("grid", auto_request)
    assert broker.submit_calls == 1
    assert session.orders[0].external_id == session.intents[0].intent_id
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest tests/unit/test_client_simulation_service.py -q`
Expected: service import fails.

- [ ] **Step 3: Implement service behavior**

Load strategy modules in isolation and require `generate_intents(context)`. Compute `intent_id=sha256(strategy_slug + strategy_version + trade_account + intent_key)`. Apply whitelist, version, account status, engine status, symbol, lot, available cash/position, 5% order, 20% symbol, 80% total-position, T+1 and algorithm gates. Support `observe`, `manual`, and `auto`; manual intents persist as `validated` until approval.

- [ ] **Step 4: Implement atomic artifacts and recovery**

Persist `session.json`, `strategy_state.json`, `events.jsonl`, `order_intents.jsonl`, `broker_orders.jsonl`, `child_orders.jsonl`, `fills.jsonl`, `account_snapshots.jsonl`, `monitoring_snapshots.jsonl`, and `reconciliation.jsonl` under `paper_run/client_sessions/<session_id>/`. On recovery, correlate by `external_id` first and resume paused unless `auto_resume=true`.

- [ ] **Step 5: Verify service GREEN**

Run: `pytest tests/unit/test_client_simulation_service.py -q`
Expected: preflight, observe/manual/auto, risk, dedupe, stop and recovery tests pass.

### Task 4: REST API

**Files:**
- Create: `src/autostrategy/api/routers/client_simulation.py`
- Modify: `src/autostrategy/api/app.py`
- Modify: `src/autostrategy/api/dependencies.py`
- Modify: `src/autostrategy/api/errors.py`
- Modify: `src/autostrategy/api/schemas.py`
- Test: `tests/integration/test_api_client_simulation.py`

- [ ] **Step 1: Write failing API tests**

```python
def test_connection_check_and_preflight_never_return_credentials(client, monkeypatch):
    response = client.post("/api/v1/broker-connections/ft-client/check")
    assert response.status_code == 200
    assert "password" not in response.text.lower()
    assert "token" not in response.text.lower()

def test_manual_session_approval_and_stop(client):
    session = client.post("/api/v1/strategies/grid/client-simulation/sessions", json=payload)
    intent_id = session.json()["intents"][0]["intent_id"]
    assert client.post(f".../intents/{intent_id}/approve").status_code == 200
    assert client.post(f".../{session.json()['session_id']}/stop").status_code == 200
```

- [ ] **Step 2: Run API tests and verify RED**

Run: `pytest tests/integration/test_api_client_simulation.py -q`
Expected: routes return 404.

- [ ] **Step 3: Implement the designed `/api/v1` surface**

Expose connection check, filtered accounts, preflight, create/list/get session, pause/resume/stop, approve/reject intent, events, and account snapshot. Return only safe account metadata and structured errors.

- [ ] **Step 4: Verify API GREEN**

Run: `pytest tests/integration/test_api_client_simulation.py -q`
Expected: all routes and credential-leak assertions pass.

### Task 5: Code generation contract

**Files:**
- Modify: `src/autostrategy/agents/prompts/codegen.py`
- Test: `tests/integration/test_codegen_agent.py`

- [ ] **Step 1: Add a failing prompt-contract test**

```python
def test_codegen_prompt_requires_pure_generate_intents_contract():
    assert "def generate_intents(context: dict) -> dict" in CODEGEN_SYSTEM_PROMPT
    assert "must not perform HTTP" in CODEGEN_SYSTEM_PROMPT
```

- [ ] **Step 2: Verify RED, update prompt, verify GREEN**

Run: `pytest tests/integration/test_codegen_agent.py -q`
Expected before change: prompt assertion fails; after change: pass.

### Task 6: Web workbench

**Files:**
- Create: `src/autostrategy/web/frontend/src/ClientSimulationPanel.tsx`
- Create: `src/autostrategy/web/frontend/src/ClientSimulationPanel.test.tsx`
- Modify: `src/autostrategy/web/frontend/src/types.ts`
- Modify: `src/autostrategy/web/frontend/src/api/client.ts`
- Modify: `src/autostrategy/web/frontend/src/StrategyWorkbench.tsx`
- Modify: `src/autostrategy/web/frontend/src/App.css`

- [ ] **Step 1: Write failing UI tests**

```tsx
it('separates local replay from FT client simulation', async () => {
  render(<ClientSimulationPanel slug="grid" />)
  expect(screen.getByText('非凸客户端模拟账户')).toBeInTheDocument()
  expect(screen.getByText('本地历史回放')).toBeInTheDocument()
})

it('does not enable auto start when preflight has hard failures', async () => {
  expect(await screen.findByRole('button', { name: '启动非凸模拟盘' })).toBeDisabled()
})
```

- [ ] **Step 2: Run UI tests and verify RED**

Run: `npm test -- --run ClientSimulationPanel.test.tsx`
Expected: component import fails.

- [ ] **Step 3: Implement panel and API types**

Render connection, client version, account login, mode, preflight, risk summary, funds, positions, intents, orders, completion/exposure/cancel/error rates, and lifecycle buttons. Auto mode requires explicit confirmation.

- [ ] **Step 4: Verify UI GREEN and build**

Run: `npm test -- --run ClientSimulationPanel.test.tsx && npm run build`
Expected: test and TypeScript production build pass.

### Task 7: Full verification and documentation

**Files:**
- Modify: `README.md`
- Verify: all files above

- [ ] **Step 1: Document safe setup**

Document environment variables, loopback-only base URL, simulation-account whitelist, client 3.11.4 minimum, symbol mapping, external-ID confirmation and observe-first workflow. Do not include credential values.

- [ ] **Step 2: Run focused and regression suites**

Run: `pytest tests/unit/test_ft_client_broker.py tests/unit/test_client_simulation_service.py tests/integration/test_api_client_simulation.py -q`

Run: `pytest tests/unit tests/integration -q`

Run: `ruff check src tests`

Run: `npm test -- --run && npm run build` from `src/autostrategy/web/frontend`.

Expected: zero failures and no credential strings in generated artifacts or logs.

- [ ] **Step 3: Inspect exact change scope**

Run: `git diff --check` and `git status --short`.

Expected: only intended FT simulation additions plus the user's pre-existing changes; no PDF temporary files and no secrets.
