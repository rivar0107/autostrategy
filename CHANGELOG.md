# Changelog

## [0.1.0.0] - 2026-07-05

### Added

- 本地 REST API 与 Ant Design Web 工作台：策略 CRUD、设计生成、代码生成、回测、artifact 预览。
- 本地 MCP adapter：list_strategies、get_strategy、get_strategy_paths、list_templates、get_backtest_result、create_strategy、run_backtest。
- 回测任务化：后台 in-memory job + 子进程隔离执行生成策略代码，支持轮询状态。
- Replay-first 模拟运行：策略暴露 `run_paper(config)` 即可在本地重放并产出 paper_run_result.json / events.jsonl / paper_run.log。
- LLM 配置安全边界：HTTP 更新接口限制 base_url host 与 api_key_env 白名单，防止 API key 被导向非预期主机。
- CodegenAgent 危险代码模式检查：拒绝 `os.system`、`subprocess`、`eval(`、`exec(`、`socket`、`shutil.rmtree` 等生成内容。
- Workspace 路径安全校验：拒绝 `../` 与绝对路径，防止 path traversal。

### Changed

- CLI 新增 `serve` 命令启动本地 API 与 Dashboard。
- README 改为更适合用户阅读的产品介绍、快速开始和浏览器工作台说明。
- `pyproject.toml` 增加 `api`、`web`、`mcp` 可选依赖组。

### Fixed

- LLM API key 缺失时返回结构化错误（428），前端自动弹出配置引导。
