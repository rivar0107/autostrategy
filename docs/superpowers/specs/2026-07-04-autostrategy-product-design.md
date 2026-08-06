# autostrategy 产品化设计文档

**版本:** v0.1-draft  
**日期:** 2026-07-04  
**分支:** `feature/productize-agent-platform`  
**作者:** Claude Code + 用户协作  

---

## 1. 背景与目标

### 1.1 现状

`autostrategy` 目前是一个 Claude Code Skill，提供 AI 驱动的量化策略生成与回测工作流：

- **Phase 1**: 设计 Agent 将自然语言需求翻译为 `STRATEGY_DESIGN.md`
- **Phase 2**: 代码 Agent 将设计文档翻译为 `strategy.py` 并运行回测
- **Phase 3**: 优化 Agent 对回测不达标的策略进行结构化优化

核心优势：强制"设计文档先行"，避免 AI 自由发挥导致策略不可控。

### 1.2 目标

参考 [Vibe-Trading](https://github.com/HKUDS/Vibe-Trading) 和 [Data-Analysis-Agent](https://github.com/Zafer-Liu/Data-Analysis-Agent)，将 `autostrategy` 从一个 Claude Code Skill 升级为**开源产品**，让个人投资者能够在本地完成：

1. **策略回测**: 自然语言设计策略、生成代码、运行回测、查看报告
2. **策略模拟执行**: 在模拟盘上跟踪策略信号和虚拟收益
3. **策略学习**: 上传交易日志，分析行为偏差，提取规则并与策略对比
4. **策略管理**: 工作区、模板、版本、搜索与分类

### 1.3 非目标

- 不托管用户资金
- 不推荐实盘交易
- 不内置 LLM API key
- 首期不支持加密货币与衍生品

---

## 2. 核心决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 产品形态 | CLI + Web UI + REST API + MCP 并重 | 覆盖开发者与普通用户，参考 Vibe-Trading |
| 部署方式 | 本地优先，Docker 可选 | 降低合规与运维压力，保护用户数据 |
| 实盘边界 | 仅模拟盘 | 与现有 Skill "不推实盘" 原则一致，降低风险 |
| 目标市场 | A股、港股、美股 | 与个人投资者需求最匹配 |
| 目标用户 | 个人投资者 | 自然语言驱动，降低使用门槛 |
| 与现有 Skill 关系 | Skill 变为助手层 | 主要交互在产品中完成，Skill 作为 MCP/AI 入口 |
| 仓库组织 | 方式 1: 根目录产品化，Skill 迁入子目录 | 符合开源产品惯例，参考 Vibe-Trading |
| 数据源策略 | 免费为主，可选付费 key | 降低入门成本，高级用户可扩展 |
| 商业模式 | 开源 + 官方 SaaS（远期） | 社区驱动，未来可选云端托管 |
| 开源许可证 | 待定（建议 MIT 或 Apache-2.0） | 便于社区采用与二次开发 |

---

## 3. 产品定位

> **一句话描述**: autostrategy 是一个面向个人投资者的本地开源量化策略平台，用自然语言生成、回测、管理和学习交易策略。

### 3.1 用户画像

- **角色**: 个人投资者、量化爱好者
- **能力**: 不一定精通 Python，但愿意用自然语言描述交易思路
- **场景**: 想快速验证一个策略想法，管理多个策略，复盘自己的交易记录
- **痛点**: 现有工具要么太复杂（需要写代码），要么太黑盒（AI 直接生成不可控）

### 3.2 核心价值主张

1. **设计文档先行**: 策略逻辑先写入 `STRATEGY_DESIGN.md`，再生成代码，可审计、可复现
2. **人在回路**: 每个关键阶段都需要用户确认，避免 AI 自主决策
3. **本地优先**: 数据与策略都保存在用户本地，保护隐私
4. **渐进增强**: 从 CLI 到 Web UI，从回测到模拟盘，逐步扩展

---

## 4. 架构设计

### 4.1 整体架构

autostrategy 采用分层架构：

```
┌─────────────────────────────────────────────────────────────┐
│  用户入口层                                                   │
│  CLI │ Web UI (React) │ REST API (FastAPI) │ MCP Server     │
├─────────────────────────────────────────────────────────────┤
│  应用服务层                                                   │
│  Strategy Service │ Backtest Service │ Paper Trading        │
│  Learning Service │ Report Service                            │
├─────────────────────────────────────────────────────────────┤
│  AI Agent 层                                                  │
│  Design Agent │ Codegen Agent │ Optimization Agent           │
│  Learning Agent                                               │
├─────────────────────────────────────────────────────────────┤
│  核心引擎层                                                   │
│  Data Registry │ Backtest Engine │ Paper Engine              │
│  Market Calendar │ LLM Client                                │
├─────────────────────────────────────────────────────────────┤
│  持久化层                                                     │
│  Workspace │ Run DB (SQLite/DuckDB) │ Artifacts │ Memory     │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 模块职责

| 模块 | 职责 |
|------|------|
| `autostrategy.cli` | CLI 入口与命令解析 |
| `autostrategy.web` | FastAPI 应用与路由 |
| `autostrategy.api` | REST API 端点 |
| `autostrategy.mcp` | MCP Server 工具注册 |
| `autostrategy.core` | 策略、回测、模拟盘、数据源等核心引擎 |
| `autostrategy.agents` | Design/Codegen/Optimization/Learning Agent |
| `autostrategy.config` | 配置模型与持久化 |
| `autostrategy.persistence` | Workspace、Run DB、Artifacts、Memory 抽象 |
| `.claude/skills/autostrategy` | Claude Code Skill 入口（兼容层） |

---

## 5. 数据流

以用户说"帮我做一个 A 股双均线策略，回测 2022-2024"为例：

1. **请求解析**: CLI/Web/MCP 接收请求，识别意图 `create_strategy + backtest`
2. **创建工作区**: Strategy Service 在 `~/.autostrategy/strategies/<id>/` 创建目录
3. **策略设计**: Design Agent 读取 prompt，调用 LLM 生成 `STRATEGY_DESIGN.md`
4. **用户确认**: Web/CLI 展示设计摘要，用户点击确认（审批点 1）
5. **代码生成**: Codegen Agent 按 DESIGN 生成 `strategy.py`、`config.yaml` 等
6. **质量检查**: Quality Check 校验 DESIGN 与代码一致性
7. **数据加载**: Data Registry 按市场选择数据源（akshare / tushare / yfinance）
8. **回测执行**: Backtest Engine 运行回测，输出结果到 Run DB 和 Artifacts
9. **评分报告**: `score_strategy()` 计算分数，Report Service 生成 HTML/Markdown 报告
10. **后续路径**: 根据结果进入优化、模拟盘或学习

---

## 6. 阶段路线图

| 阶段 | 名称 | 周期 | 核心交付 | 验收标准 |
|------|------|------|---------|---------|
| Phase 0 | 仓库重构与产品骨架 | 2-3 周 | `pyproject.toml`、`src/autostrategy/`、CLI 入口、配置系统、Skill 迁移 | `pip install -e .` 后 CLI 可用，Skill 仍能安装 |
| Phase 1 | 策略管理中心 | 2-3 周 | Workspace 抽象、策略 CRUD、模板市场、版本管理 | CLI 能创建/列出/查看策略 |
| Phase 2 | AI 策略设计 Agent 产品化 | 2-3 周 | `DesignAgent`、统一 LLM Client、CLI 自然语言设计 | CLI 能生成并通过 quality check 的 DESIGN |
| Phase 3 | 代码生成 + 回测引擎 | 3-4 周 | `CodegenAgent`、Data Registry、Backtest Engine、Run DB | 端到端跑通示例策略并生成报告 |
| Phase 4 | Web UI + REST API + MCP | 4-6 周 | FastAPI、React、SSE、MCP Server | 浏览器能完成全流程 |
| Phase 5 | 模拟盘 + 策略学习 + 优化 | 4-5 周 | Paper Trading Service、Optimization Agent、Learning Agent | 能跑模拟盘、上传日志复盘、自动优化 |
| Phase 6 | 发布与生态 | 持续 | PyPI、Docker、文档、模板市场、官方 SaaS（远期） | 非技术用户能按 README 安装并跑通示例 |

**MVP（Phase 0-3）周期**: 9-13 周  
**完整本地产品（Phase 0-5）周期**: 13-19 周

---

## 7. 技术栈

| 层级 | 工具/库 |
|------|--------|
| 语言 | Python 3.11+ |
| CLI | Typer 或 Click |
| API | FastAPI |
| Web UI | React 19 + TypeScript |
| 图表 | ECharts 或 Plotly |
| 数据校验 | Pydantic |
| 数据库 | SQLite（默认）+ DuckDB（可选） |
| ORM | SQLAlchemy |
| 回测引擎 | Backtrader（复用）+ 自定义 CompositeEngine |
| 数据源 | akshare, tushare, yfinance, mootdx, futu |
| 实时进度 | SSE / WebSocket |
| MCP | Python MCP SDK |
| 部署 | Docker（可选） |
| 包管理 | uv 或 hatch |
| 分发 | PyPI |

---

## 8. LLM Provider 配置

### 8.1 设计原则

- 用户必须自行配置 LLM API key
- key 不出本地，不上传服务器
- 支持多 provider，可为不同任务配置不同模型

### 8.2 配置方式

1. **Web UI Settings 页面**
2. **CLI**: `autostrategy config set llm.provider=deepseek llm.api_key=xxx`
3. **环境变量**: `.env` 文件

### 8.3 存储位置

- 配置文件: `~/.autostrategy/settings.yaml`
- API key: 可选系统 keyring 加密存储
- 不写入仓库

### 8.4 支持 Provider

- OpenAI
- DeepSeek
- Kimi / Moonshot
- Qwen
- Z.ai
- MiniMax
- Gemini
- Local: Ollama / vLLM
- 任意 OpenAI-compatible API

---

## 9. 安全边界

### 9.1 资金安全

- 不托管用户资金
- 不支持实盘交易（仅模拟盘）
- 所有"执行"都是本地虚拟账户

### 9.2 数据安全

- 策略与数据保存在本地 `~/.autostrategy/`
- API key 本地存储，加密可选
- 不上传策略代码到云端

### 9.3 代码安全

- AI 生成的代码在执行前经过 AST 预检
- 回测引擎在沙箱化进程中运行（参考 Vibe-Trading 做法）
- 不执行用户上传的任意代码

### 9.4 访问安全

- 本地 Web UI 默认仅监听 `127.0.0.1`
- 远程访问需要显式配置 `API_AUTH_KEY`
- 上传文件大小与类型限制

---

## 10. 测试策略

### 10.1 单元测试

- 核心引擎（评分函数、Data Registry、配置解析）必须有 pytest 覆盖
- 边界情况：空数据、异常参数、未来数据检测

### 10.2 集成测试

- "创建策略 → 生成代码 → 回测"端到端测试
- 使用 mock 数据源，避免依赖网络

### 10.3 回归测试

- 示例策略回测结果变化时告警
- Skill 安装路径兼容性测试

### 10.4 UI 测试

- Phase 4 引入后，核心用户流程用 Playwright 或 Cypress 覆盖

---

## 11. 仓库结构

```
autostrategy/
├── pyproject.toml
├── README.md
├── LICENSE
├── .env.example
├── .gitignore
├── src/
│   └── autostrategy/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py
│       ├── config.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── workspace.py
│       │   ├── strategy.py
│       │   ├── data_registry.py
│       │   ├── backtest_engine.py
│       │   ├── paper_engine.py
│       │   └── market_calendar.py
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── design_agent.py
│       │   ├── codegen_agent.py
│       │   ├── optimization_agent.py
│       │   └── learning_agent.py
│       ├── web/
│       │   ├── __init__.py
│       │   ├── app.py
│       │   └── routers/
│       ├── api/
│       │   ├── __init__.py
│       │   └── routes/
│       ├── mcp/
│       │   ├── __init__.py
│       │   └── server.py
│       ├── persistence/
│       │   ├── __init__.py
│       │   ├── run_db.py
│       │   └── memory.py
│       └── llm/
│           ├── __init__.py
│           └── client.py
├── frontend/
│   ├── package.json
│   └── src/
├── examples/
│   └── dynamic-grid-multi-market/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-07-04-autostrategy-product-design.md
└── .claude/
    └── skills/
        └── autostrategy/
            ├── SKILL.md
            └── prompts/
                ├── design_agent.md
                ├── codegen_agent.md
                └── optimization_agent.md
```

---

## 12. 风险与局限

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 工程量过大 | 延期 | 按 Phase 拆分，MVP 聚焦 CLI + 核心回测 |
| 数据源不稳定 | 回测失败 | Data Registry 多源 fallback |
| AI 生成代码质量不稳定 | 策略不可靠 | 设计文档先行 + 质量检查 + 人在回路 |
| 用户期望实盘 | 合规风险 | 明确仅模拟盘，文档与 UI 反复强调 |
| Skill 兼容性破坏 | 老用户受影响 | 根目录保留 shim `SKILL.md` |

---

## 13. 验收标准

### 13.1 设计文档验收

- [x] 产品定位清晰
- [x] 架构分层明确
- [x] 数据流完整
- [x] 阶段路线可执行
- [x] 技术栈已确认
- [x] 安全边界明确

### 13.2 Phase 0 骨架验收

- [ ] `pip install -e .` 成功
- [ ] `autostrategy --version` 正常输出
- [ ] `autostrategy config init` 创建 `~/.autostrategy/`
- [ ] 现有 Skill 安装路径仍可用
- [ ] 单元测试能通过基础配置测试

---

## 14. 下一步

1. 用户 review 并批准本设计文档
2. 调用 `writing-plans` skill 生成 Phase 0-1 的详细实现计划
3. 按实现计划搭建 Phase 0 产品骨架
4. 提交到 `feature/productize-agent-platform` 分支

---

**待确认事项:**

1. 开源许可证选择：MIT vs Apache-2.0
2. CLI 框架选择：Typer vs Click
3. Web UI 图表库：ECharts vs Plotly
4. 包管理工具：uv vs hatch
5. Run DB 默认：SQLite vs DuckDB

这些可以在实现 Phase 0 时最终确定。
