# Phase 4A: 本地 Agent 服务平台后端骨架验收记录

**日期:** 2026-07-04  
**分支:** `feature/productize-agent-platform`  
**定位:** Phase 4 的第一段收敛实现，不做完整 React/WebSocket/多用户系统，先交付本地 REST API + Dashboard + MCP adapter 的后端骨架。

---

## 1. 完成范围

### 1.1 Service Layer

新增 `src/autostrategy/services/`，将原本 CLI 直接编排的核心能力抽成可复用服务：

- `StrategyService`: 策略创建、列表、详情、路径、删除、模板列表
- `DesignService`: 调用 `DesignAgent` 创建 `STRATEGY_DESIGN.md`
- `CodegenService`: 调用 `CodegenAgent` 生成策略实现文件
- `BacktestService`: 调用 `run_backtest_workflow()` 并读取回测结果
- 统一 service exception 与 Pydantic response model

**意义:** CLI / REST API / MCP 后续可以复用同一套业务层，避免三套逻辑分叉。

### 1.2 FastAPI REST API

新增 `src/autostrategy/api/`，提供本地 API：

- `GET /api/v1/health`
- `GET /api/v1/info`
- `GET /api/v1/config`
- `GET /api/v1/templates`
- `GET /api/v1/strategies`
- `POST /api/v1/strategies`
- `GET /api/v1/strategies/{slug}`
- `DELETE /api/v1/strategies/{slug}`
- `GET /api/v1/strategies/{slug}/paths`
- `POST /api/v1/designs`
- `POST /api/v1/strategies/{slug}/codegen`
- `POST /api/v1/strategies/{slug}/backtest`
- `GET /api/v1/strategies/{slug}/backtest-result`

错误响应统一为：

```json
{
  "error": {
    "code": "strategy_not_found",
    "message": "Strategy 'demo' not found.",
    "details": {}
  }
}
```

### 1.3 本地 Dashboard

新增 `src/autostrategy/web/static/`：

- `index.html`
- `app.css`
- `app.js`

当前 Dashboard 是轻量入口，功能包括：

- 策略列表展示
- 策略 slug / market / status / template 展示
- OpenAPI、health、templates 快捷入口

**设计取舍:** Phase 4A 不引入 React/Vite，避免前端构建链路过早膨胀。

### 1.4 CLI Serve 命令

新增：

```bash
autostrategy serve --host 127.0.0.1 --port 8000
```

支持：

```bash
autostrategy serve --workspace-root /path/to/strategies
```

### 1.5 MCP Adapter

新增 `src/autostrategy/mcp/`：

- `tools.py`
- `server.py`

Phase 4A 暴露保守工具范围：

- `list_strategies`
- `get_strategy`
- `get_strategy_paths`
- `list_templates`
- `get_backtest_result`
- `create_strategy`
- `run_backtest`

暂不开放 `design_strategy` / `codegen_strategy`，避免 MCP 自动形成“生成代码 → 执行代码”的高风险链路。

### 1.6 安全边界

已补充两类基础防线：

1. `Workspace.resolve_strategy_path()`
   - 拒绝 `../` path traversal
   - 拒绝绝对路径
   - 读写文件统一限制在 strategy workspace 内

2. `CodegenAgent` 危险模式检查
   - 拒绝 `os.system`
   - 拒绝 `subprocess`
   - 拒绝 `socket`
   - 拒绝 `eval(`
   - 拒绝 `exec(`
   - 拒绝 `shutil.rmtree`

---

## 2. 暂缓范围

Phase 4A 明确暂缓：

- React/Vite 前端工程
- WebSocket/SSE 长任务进度流
- 任务队列 / 后台 job 系统
- 多用户、登录、鉴权
- 远程托管部署
- 完整容器/进程沙箱化执行
- MCP `design_strategy` / `codegen_strategy` 写操作
- Web UI 内置策略编辑器

这些能力应进入 Phase 4B 或 Phase 5 分阶段实现。

---

## 3. 验收标准

| 验收项 | 状态 | 说明 |
|--------|------|------|
| service layer 可复用 | ✅ | CLI/API/MCP 已具备共享服务基础 |
| REST API 可用 | ✅ | FastAPI app 与核心 routers 已实现 |
| Dashboard 可访问 | ✅ | `/` 提供最小静态页面 |
| `autostrategy serve` 可用 | ✅ | 已新增本地服务启动命令 |
| MCP adapter 可测试 | ✅ | tool 函数可直接单测，server 可按需创建 |
| 路径穿越防护 | ✅ | `tests/unit/test_workspace_security.py` 覆盖 |
| 生成代码危险模式检查 | ✅ | `test_codegen_rejects_dangerous_python_patterns` 覆盖 |
| 全量测试通过 | ✅ | `63 passed, 1 warning` |

---

## 4. 验证命令

安装依赖：

```bash
. .venv/bin/activate
pip install -e ".[dev,api,web,mcp]"
```

测试：

```bash
python -m pytest -q
```

结果：

```text
63 passed, 1 warning in 0.34s
```

CLI smoke：

```bash
autostrategy serve --help
```

结果：

```text
serve help ok
```

可选手工 API smoke：

```bash
autostrategy serve --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/api/v1/health
curl http://127.0.0.1:8000/api/v1/strategies
curl http://127.0.0.1:8000/api/v1/templates
```

---

## 5. 后续建议

### Phase 4B

- Dashboard 增加策略详情页
- 支持在 Web UI 触发 codegen/backtest
- 引入 SSE 展示长任务进度
- 增加更完整的 API 文档与示例
- 若需要远程访问，设计最小 API auth

### Phase 5 前置

- 将回测执行迁移到 subprocess/container executor
- Optimization Agent 产品化
- Paper Trading Service
- Learning Agent / 交易日志复盘

---

## 6. 结论

Phase 4A 已完成“本地 Agent 服务平台后端骨架”：现有 CLI 产品能力已经具备 REST API、最小 Dashboard 和 MCP adapter 的基础入口，并补上了最关键的路径安全与生成代码危险模式防线。
