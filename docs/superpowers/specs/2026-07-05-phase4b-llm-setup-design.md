# Phase 4B LLM 配置引导设计

## 背景

Phase 4B 已经把 autostrategy 推进到本地 Web 工作台形态。用户可以在浏览器里创建策略、生成设计、生成代码、查看 artifacts、运行回测。

现在的问题是：LLM 配置仍然停留在开发者心智。API key 只从本地环境变量读取，非敏感配置保存在 `~/.autostrategy/settings.yaml`。这个安全边界是对的，但 Web 用户第一次打开产品时不知道自己需要先配置大模型，常见结果是第一次点击“自然语言生成设计”或“生成代码”就失败。

对一个 startup 产品来说，这不是小瑕疵。首次使用的失败会直接破坏信任。用户不会先研究配置系统，他们只会觉得“这个产品坏了”。

## 目标

本次设计要解决三个痛点，优先级为：

1. 避免用户第一次点击 LLM 功能时才看到错误。
2. 让配置过程尽量短，用户能快速回到生成策略的主流程。
3. 明确安全边界，让用户知道 autostrategy 不保存 API key。

也就是：A > B > C。先不让用户撞墙，再减少步骤，最后把安全说明讲清楚。

## 非目标

本阶段不做这些事：

- 不引入账号系统。
- 不保存 API key 或任何 secret。
- 不做多 provider 凭证管理。
- 不做在线密钥校验或“测试连接”。
- 不做完整 provider/model 兼容矩阵。

这些能力以后可能有价值，但会显著扩大范围。Phase 4B 的正确边界是“本地环境变量引导 + 非敏感配置持久化”。

## 用户体验

### 首次进入

用户打开 Web 工作台时，前端请求后端配置状态。

如果后端判断 LLM 未就绪，页面仍正常加载策略列表等非 LLM 功能，同时自动弹出 LLM 设置引导。

引导里展示：

- 当前 provider。
- 当前 model。
- 可选 base_url。
- API key 应该读取的环境变量名。
- 一段明确说明：API key 不会保存在 autostrategy 中，只从本机环境变量读取。
- 一段可复制的 shell 示例，例如：

```bash
export AUTOSTRATEGY_LLM_API_KEY="your-api-key"
autostrategy serve
```

这里的关键不是把所有配置知识讲完，而是让用户知道下一步做什么。

### 用户主动配置

Header 增加 “LLM 设置”入口。用户可以随时重新打开同一套设置界面，修改非敏感配置。

可保存字段只有：

- provider
- model
- base_url
- api_key_env

保存后后端写入 `~/.autostrategy/settings.yaml`。如果环境变量仍然不存在，UI 不应假装已经配置完成，而是继续提示“还需要在本机 shell 中 export API key 并重启服务或刷新环境”。

### 运行时兜底

如果用户启动时配置是好的，但后来环境变量被移除，或者服务换了启动环境，LLM 动作可能仍然失败。

这时后端应该返回结构化错误：

```json
{
  "error": {
    "code": "llm_configuration_required",
    "message": "LLM API key is not configured.",
    "details": {
      "provider": "openai",
      "api_key_env": "AUTOSTRATEGY_LLM_API_KEY",
      "llm_ready": false,
      "setup_hint": "Set AUTOSTRATEGY_LLM_API_KEY in your local shell before starting autostrategy."
    }
  }
}
```

前端收到这个错误码时，打开同一套 LLM 设置引导，而不是只 toast 一个失败消息。

## 后端设计

### 配置状态

扩展 `GET /api/v1/config`，返回安全配置摘要和派生状态：

- version
- default_market
- llm_provider
- llm_model
- llm_base_url
- llm_api_key_env
- llm_ready
- llm_missing_api_key
- llm_setup_hint

`llm_ready` 必须由后端推导，不能由前端猜。原因很简单：只有后端进程知道自己启动时能读到哪些环境变量。

### 配置更新

新增 `PUT /api/v1/config/llm`。

请求体只允许非敏感字段：

- provider
- model
- base_url
- api_key_env

服务端使用现有 `load_settings()` / `save_settings()`。不要新增 `api_key` 字段，不要把 secret 写进 YAML。

### API key 解析

现有 `LLMClient._resolve_api_key()` 已经定义了解析顺序：

1. 配置里的 `api_key_env`
2. `AUTOSTRATEGY_LLM_API_KEY`
3. `{PROVIDER}_API_KEY`
4. `OPENAI_API_KEY`

配置状态和运行时错误都应该复用这套规则，避免启动检查说“未配置”，运行时却能跑，或者反过来。

### 结构化错误

把当前缺 key 的 raw `RuntimeError` 改为服务层异常，例如 `LLMConfigurationRequiredError`。

错误码固定为：

```text
llm_configuration_required
```

API 层继续复用现有结构化错误处理，不给前端增加特殊解析路径。

## 前端设计

### API 类型

新增配置响应和更新 payload 类型：

- `ConfigResponse`
- `LlmConfigUpdate`

API client 新增：

- `config()` 或 `getConfig()`
- `updateLlmConfig(payload)`

### App 状态

`App.tsx` 增加：

- 当前 config 状态
- LLM 设置 modal 是否打开
- LLM 设置 form

启动时和现有 strategies/templates 一起加载 config。加载失败不应阻塞整个工作台，但需要展示错误。

### Modal 内容

复用 Ant Design `Modal` + `Form` 风格，和现有“创建策略”“自然语言生成设计”保持一致。

建议文案重点：

- “autostrategy 不保存 API key。”
- “请在启动服务的终端中设置环境变量。”
- “修改环境变量后，通常需要重启 `autostrategy serve`。”

这比抽象地说“配置未完成”更有用。用户需要的是下一步动作。

### LLM 动作兜底

至少处理两个入口：

- 自然语言生成设计
- 生成代码

如果捕获到 `ApiError.code === 'llm_configuration_required'`：

1. 更新本地 config 状态。
2. 打开 LLM 设置 modal。
3. 展示后端返回的 setup hint。

普通错误仍按现有 toast 处理。

## 验收标准

### 用户层面

- 没有 API key 时，用户第一次打开 Web 就能看到配置引导。
- 用户仍然能浏览策略列表、查看详情和 artifacts。
- 用户知道 API key 不会被保存。
- 用户知道应该 export 哪个环境变量。
- 用户在 LLM 操作失败时看到同一套引导，而不是一个无法行动的错误。

### 工程层面

- 后端不保存 secret。
- readiness 和 runtime 使用同一套 API key 解析规则。
- `llm_configuration_required` 是稳定错误码。
- 前端没有复制多套 onboarding/error-recovery UI。
- 测试覆盖缺 key、保存非敏感配置、结构化错误和前端 modal 打开逻辑。

## 测试建议

后端：

- 无 settings、无 env var 时，`GET /config` 返回 `llm_ready=false`。
- 有 settings、无 env var 时，返回缺少指定 env var。
- `PUT /config/llm` 只持久化非敏感字段。
- LLM-backed endpoint 缺 key 时返回 `llm_configuration_required`。

前端：

- 启动时 `/config` 返回未就绪，自动打开 LLM 设置 modal。
- 保存配置时调用 `updateLlmConfig`。
- LLM action 返回 `llm_configuration_required`，重新打开同一 modal。

真实验证：

1. 不设置任何 API key，启动 Web，确认 modal 自动出现。
2. 保存 provider/model/api_key_env，确认没有 secret 写入 settings。
3. 设置对应环境变量并重启服务，确认 LLM action 能继续。
4. 移除环境变量后重试 LLM action，确认结构化错误触发同一引导。

## 决策

采用最小但完整的本地配置引导方案。

它不解决所有 LLM provider 管理问题，但解决当前最伤害用户体验的问题：第一次使用时不知道怎么开始。这个范围够小，可以在 Phase 4B 内稳定交付；边界也够清楚，不会把本地工具过早做成一套账号和密钥管理系统。
