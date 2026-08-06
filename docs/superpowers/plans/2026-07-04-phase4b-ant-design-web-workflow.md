# Phase 4B: Ant Design Web 核心工作流验收记录

**日期:** 2026-07-04  
**分支:** `feature/productize-agent-platform`  
**定位:** 在 Phase 4A 本地服务骨架基础上，将 Web 端升级为基于 Ant Design 官方组件与默认主题的策略工作台。

---

## 1. 设计约束

用户明确要求：

- Web 端基于 Ant Design 组件开发
- 参考 https://ant.design/components/overview-cn
- 参考 https://github.com/ant-design/ant-design
- 不得擅自使用其他主题

本阶段采用：

- React
- Vite
- TypeScript
- 官方 `antd` npm package
- `antd/dist/reset.css`
- Ant Design 默认主题

本阶段不引入：

- Tailwind
- Bootstrap
- shadcn/ui
- DaisyUI
- 其他主题包
- 自定义品牌主题系统

---

## 2. 完成范围

### 2.1 Artifact Preview API

新增命名 artifact API，避免任意路径读取：

- `GET /api/v1/strategies/{slug}/artifacts`
- `GET /api/v1/strategies/{slug}/artifacts/{artifact_key}`

允许的 artifact key：

| Key | 文件 |
|-----|------|
| `design` | `STRATEGY_DESIGN.md` |
| `strategy_code` | `strategy.py` |
| `config` | `config.yaml` |
| `readme` | `README.md` |
| `requirements` | `requirements.txt` |
| `fetch_data` | `data/fetch_data.py` |
| `backtest_result` | `backtest/results/backtest_result.json` |

新增：

- `src/autostrategy/services/artifact_service.py`
- `src/autostrategy/api/routers/artifacts.py`
- `ArtifactMetaResponse`
- `ArtifactListResponse`
- `ArtifactContentResponse`

安全规则：

- 只读白名单 artifact
- 不暴露 arbitrary path API
- 仍通过 `Workspace.resolve_strategy_path()` 限制 workspace 边界
- 非法 key 返回结构化 `artifact_not_found`

### 2.2 Ant Design 前端工程

新增前端工程：

- `package.json`
- `package-lock.json`
- `src/autostrategy/web/frontend/`

核心文件：

- `src/autostrategy/web/frontend/index.html`
- `src/autostrategy/web/frontend/vite.config.ts`
- `src/autostrategy/web/frontend/tsconfig.json`
- `src/autostrategy/web/frontend/src/main.tsx`
- `src/autostrategy/web/frontend/src/App.tsx`
- `src/autostrategy/web/frontend/src/App.css`
- `src/autostrategy/web/frontend/src/api/client.ts`
- `src/autostrategy/web/frontend/src/types.ts`

构建输出：

- `src/autostrategy/web/static/index.html`
- `src/autostrategy/web/static/assets/*`

### 2.3 Web 工作台能力

Ant Design 工作台已支持：

- 策略列表
- 策略状态 Tag
- 创建空策略 Modal/Form
- 自然语言生成设计 Modal/Form
- 策略详情 Drawer
- Artifact Tabs
- Design / Code / Config / README / Backtest Result 预览
- Codegen 操作
- Backtest 操作
- Backtest Score 与关键指标摘要
- loading / success / error message

使用的 Ant Design 组件包括：

- `Layout`
- `Typography`
- `Card`
- `Table`
- `Tag`
- `Button`
- `Space`
- `Modal`
- `Form`
- `Input`
- `Select`
- `Drawer`
- `Descriptions`
- `Tabs`
- `Alert`
- `Spin`
- `Empty`
- `Statistic`
- `Progress`
- `Popconfirm`
- `message`

### 2.4 FastAPI 静态托管

Vite 构建产物输出到 `src/autostrategy/web/static/`，继续由 `autostrategy serve` 托管。

入口仍然是：

```bash
autostrategy serve --host 127.0.0.1 --port 8000
```

---

## 3. 暂缓范围

Phase 4B 暂缓：

- React Router 深链路
- SSE/WebSocket 长任务进度
- 后台任务队列
- Monaco Editor
- Markdown 富渲染
- 自定义主题系统
- MCP `design_strategy` / `codegen_strategy`
- 多用户鉴权
- 远程部署安全边界

---

## 4. 验收标准

| 验收项 | 状态 | 说明 |
|--------|------|------|
| 使用官方 Ant Design 组件 | ✅ | 前端依赖 `antd`，入口导入 `antd/dist/reset.css` |
| 不引入其他主题系统 | ✅ | 未引入 Tailwind/Bootstrap/shadcn 等 |
| Artifact API 白名单读取 | ✅ | 通过 artifact key 映射固定文件 |
| 策略列表 Web 展示 | ✅ | AntD `Table` |
| 创建策略 Web 表单 | ✅ | AntD `Modal` + `Form` |
| 自然语言设计 Web 表单 | ✅ | AntD `Modal` + `Input.TextArea` |
| 策略详情工作台 | ✅ | AntD `Drawer` + `Descriptions` + `Tabs` |
| Artifact 预览 | ✅ | Artifact Tabs + preview panel |
| Codegen/Backtest 操作 | ✅ | AntD Button loading + message |
| 回测摘要 | ✅ | AntD `Statistic` + `Progress` |
| 前端构建 | ✅ | `npm run build` 通过 |
| 前端测试 | ✅ | `npm test` 通过 |

---

## 5. 验证命令

Python 后端：

```bash
. .venv/bin/activate
python -m pytest -q
```

前端：

```bash
npm install
npm run build
npm test
```

本地服务：

```bash
autostrategy serve --host 127.0.0.1 --port 8000
```

API smoke：

```bash
curl http://127.0.0.1:8000/api/v1/health
curl http://127.0.0.1:8000/api/v1/strategies
curl http://127.0.0.1:8000/api/v1/strategies/demo/artifacts
```

浏览器 smoke：

1. 打开 `http://127.0.0.1:8000/`
2. 确认 Ant Design 工作台加载
3. 创建策略
4. 打开策略详情
5. 查看 Artifacts
6. 触发 codegen/backtest，或确认错误信息以 AntD message/Alert 展示

---

## 6. 已知说明

- Ant Design + React 构建产物当前单 chunk 超过 500KB，Vite 会提示 code splitting 建议。这不影响 Phase 4B 功能，后续可在 Phase 4C 做动态加载优化。
- Web 端 design/codegen/backtest 仍为同步请求，长耗时任务会通过按钮 loading 表示。SSE/job queue 后续再做。

---

## 7. 结论

Phase 4B 将 Web 端从 Phase 4A 的静态最小 Dashboard 升级为基于 Ant Design 的策略工作台。用户可以通过浏览器完成策略列表、创建、详情查看、artifact 预览、代码生成和回测触发等核心动作，同时保持 Ant Design 默认视觉体系和本地优先安全边界。
