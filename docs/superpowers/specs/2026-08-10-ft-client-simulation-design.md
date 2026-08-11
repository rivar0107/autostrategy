# 非凸智能交易终端客户端模拟盘接入需求设计

**版本:** v0.4-draft
**日期:** 2026-08-11
**产品:** autostrategy
**目标版本:** Phase 6A - FT Client Simulation
**依据文档:** `非凸智能交易终端API-v0.0.23 .pdf`
**最低客户端版本:** 3.11.4

### 文档修订摘要

v0.2 根据 2026-07-29 发布的 API v0.0.23 更新：

- 登录响应采用交易账户级 `data.accs.login_status` 做第一层预检。
- 算法母单使用新增的 `external_id` 承载平台幂等 ID，并显式配置 `strategy_type`、`algo_param` 和 `reach_limit_continue`。
- 母单和算法子单按 v0.0.23 的完整状态枚举归一化。
- 接入新增的拆单监控接口，用于展示完成率、暴露、撤单率和错废单率。
- 评估新增的普通委托接口后，首发仍选择算法母单；普通委托因混合 v1/v2 认证和下单响应缺少订单 ID 而暂不进入自动执行链路。
- 根据 v0.0.22 的资金字段注释，明确 `balance`、`asset` 和风险资产基数的归一化规则。

v0.3 补充产品输入要求：Web 工作台显式提供非凸账号、密码处理、客户端版本、模拟账户白名单、代码映射、算法参数、执行窗口和风控字段。密码允许通过浏览器到本机 API 的一次性请求输入，但只能以 `SecretStr` 保存在服务进程内存，禁止写入设置、artifact、日志或响应；环境变量仍作为无 UI 使用方式。

v0.4 扩展首发证券范围：产品能力允许上海、深圳市场全部可交易 A 股和 ETF，不再硬编码两只 ETF。每个会话仍必须由客户确认策略实际执行标的白名单和客户端代码映射；指数、债券、B 股、北交所、港股和美股不得作为订单标的。

---

## 1. 结论与范围决策

autostrategy 将在保留现有“本地历史回放”的基础上，新增“非凸客户端模拟盘”运行模式。用户在平台中完成策略设计、代码生成和回测后，可以选择一个已授权的非凸模拟账户，执行连接检查并启动策略。策略产生的买卖意图由 autostrategy 风控后转换成非凸算法母单，成交、资金和持仓以非凸客户端返回结果为唯一事实来源。

本需求一次性交付此前定义的第 1-4 期能力：

1. 客户端连接、模拟账户白名单、资金与持仓展示。
2. 标准化订单意图、预执行检查、观察模式和人工确认模式。
3. 自动提交、WebSocket 回报、轮询补偿、撤单、幂等和重启恢复。
4. 当前策略在客户确认的沪深 A 股/ETF 执行白名单内完成客户端模拟账户端到端运行。

只有以上四部分全部通过验收，功能才可以标记为“完成”。

### 1.1 首发支持范围

| 项目 | 首发范围 |
|---|---|
| 账户类型 | 非凸智能交易终端中的白名单模拟账户 |
| 市场 | A 股 |
| 标的 | 沪深市场可交易 A 股和 ETF；每个会话使用客户确认的执行标的白名单 |
| 策略周期 | 10 分钟；日线历史可作为中长期状态输入 |
| 运行方式 | 沪深市场形成所有策略行情标的共有的已完成 10 分钟 K 线后评估一次 |
| 订单类型 | 非凸算法母单 `api_upload_mudan`，首发默认 `TWAP` |
| 执行模式 | 观察、人工确认、自动模拟 |
| 部署方式 | autostrategy 与非凸客户端运行在同一台本地机器 |
| 用户模型 | 本地单用户、单模拟账户、同一账户同一时刻一个活动策略会话 |

### 1.2 明确不在首发范围

- 真实资金账户和实盘交易。
- 港股、美股、期货、期权。
- T0 母单、融资融券和卖空。
- 普通委托 `place_zidan` 自动执行链路。
- 低于 10 分钟的高频、逐笔或 Level 2 行情策略。
- 多用户、远程托管、云端交易执行。
- 同一交易账户同时运行多个策略。
- 自动策略优化或根据模拟盘结果自动修改策略。

指数、债券、B 股、北交所、港美股标的可保留在研究、回测和本地回放中，但在本版本的客户端模拟盘订单预检中必须被拒绝。指数可以作为行情基准进入策略上下文，不能成为订单意图。

---

## 2. 产品目标

### 2.1 用户目标

用户可以完成如下闭环：

```text
创建策略
  -> 确认 STRATEGY_DESIGN.md
  -> 生成策略代码
  -> 回测通过
  -> 选择“非凸客户端模拟盘”
  -> 检查客户端与模拟账户
  -> 预执行检查
  -> 启动策略
  -> 查看信号、母单、子单、成交、资金和持仓
  -> 暂停、恢复或停止
  -> 查看完整模拟交易复盘
```

### 2.2 工程目标

- 策略逻辑与客户端 API 解耦，生成代码不得自行发 HTTP 请求或保存 token。
- 同一份策略信号逻辑可以用于历史回放和客户端模拟盘。
- 所有订单提交可审计、可去重、可恢复。
- WebSocket 断线时通过轮询保持订单状态最终一致。
- 平台重启后可以恢复活动会话，不重复下单。
- 任何非白名单账户都不能进入客户端模拟交易流程。

### 2.3 成功指标

- 至少一只沪深 A 股和一只 ETF 能完成“信号 -> 母单 -> 子单 -> 成交 -> 持仓同步”闭环。
- 相同信号在重试、断线或重启后最多产生一个有效母单。
- 客户端账户快照与 autostrategy 展示一致，不使用本地虚拟成交覆盖客户端数据。
- 停止会话后不再产生新订单，未完成母单按规则撤销并记录结果。
- 密码和 token 不出现在配置文件、artifact、日志、API 响应或前端状态中。

---

## 3. 产品模式与术语

### 3.1 模拟运行模式

| 模式 | 标识 | 说明 |
|---|---|---|
| 本地回放 | `local_replay` | 现有能力；历史 bar 顺序重放，使用本地 `PaperAccount` |
| 客户端模拟盘 | `ft_client_simulation` | 新能力；策略信号通过非凸客户端模拟账户执行 |

UI 不再笼统使用“模拟运行”描述两种完全不同的行为。用户启动前必须明确看到当前模式。

### 3.2 客户端模拟盘执行模式

| 模式 | 标识 | 行为 |
|---|---|---|
| 观察 | `observe` | 生成订单意图和风控结果，不提交客户端 |
| 人工确认 | `manual` | 每个通过风控的意图由用户确认后提交 |
| 自动模拟 | `auto` | 通过风控后自动提交到白名单模拟账户 |

首次连接某个账户时默认使用 `observe`。切换到 `auto` 必须在 UI 中明确确认目标账户、标的白名单和风险上限。

### 3.3 核心术语

- **策略决策:** 策略在一次数据触发后给出的 `buy`、`sell` 或 `hold` 判断。
- **订单意图:** 平台从策略决策生成的、尚未发送给客户端的标准化交易请求。
- **母单:** 非凸客户端接受的算法订单。
- **子单:** 母单执行过程中产生的实际报单。
- **成交:** 子单产生的成交数量和成交价格。
- **交易会话:** 一个策略与一个模拟账户在一段时间内的运行实例。
- **账户事实来源:** 非凸客户端资金、持仓、母单、子单和成交查询结果。

---

## 4. 外部接口契约

### 4.1 已确认使用的非凸接口

| 能力 | 方法与路径 | 用途 |
|---|---|---|
| 登录 | `GET /api/ft_acc_login` | 获取交易账户、账户级登录状态、broker ID 和 v1 token |
| 登出 | `GET /api/logout_all` | 会话结束时释放客户端登录状态 |
| 账户状态 | `GET /api/query_acc_status` | 检查账户登录状态和下单引擎状态 |
| 资金 | `GET /api/get_fund_by_acc` | 获取总资产、可用资金、冻结资金和盈亏 |
| 持仓 | `GET /api/get_position_by_acc` | 获取总持仓、可用持仓和在途数量 |
| 上传母单 | `POST /api/api_upload_mudan` | 提交普通买入或卖出算法母单 |
| 查询账户母单 | `GET /api/get_mudan_by_acc` | 对账和断线恢复 |
| 查询母单子单 | `GET /api/get_zidan_by_mudan_id` | 获取报单、成交数量和成交均价 |
| 查询账户子单 | `GET /api/get_zidan_by_acc` | 补偿查询与会话恢复 |
| 操作母单 | `POST /api/op_batch_mudan` | 启动、暂停或取消指定母单 |
| 拆单监控 | `GET /api/get_algo_monitoring_info` | 获取账户/篮子的完成率、暴露、撤单率和错废单率 |
| 推送 | `ws://127.0.0.1:11356/ws/{token}` | 接收 Ping、Mudan、Zidan 和 Trade 消息 |

客户端基地址默认是 `http://127.0.0.1:11356`，由 autostrategy 后端访问。浏览器不得直接访问非凸客户端。

### 4.2 新增普通委托的执行路径决策

v0.0.23 新增以下普通委托能力：

| 能力 | 方法与路径 | 特征 |
|---|---|---|
| 普通委托下单 | `GET /api/place_zidan` | v1 token；指定价格；成功响应只返回“下单成功” |
| 普通委托查询 | `POST /api/v2/get_strategy_sub_order` | 使用 `strategy_order_id=[-1]`；要求 `ft-lp-auth` v2 JWT |
| 普通委托撤单 | `POST /api/v2/cancel_strategy_sub_order` | 按查询所得 `local_id` 撤单 |

首发不使用普通委托作为自动执行主路径，原因是：

1. 下单使用 v1 token，查询使用 v2 JWT，形成混合认证会话。
2. 下单成功响应没有返回 `local_id` 或其他订单 ID，需要事后查询关联。
3. 下单是带交易参数的 GET 请求，token、账户、价格和数量都必须做更严格的 URL 脱敏。
4. 普通委托子单使用独立的 0-6 状态语义，不能复用算法子单状态表。
5. 算法母单已经提供不被内部修改的 `external_id`，更适合幂等、恢复和审计。

普通委托只作为后续可选 `direct_order` 执行路由保留在适配器能力设计中；未单独完成双 token、订单关联和状态映射设计前，预检必须拒绝该路由。

### 4.3 母单字段映射

| autostrategy 字段 | 非凸字段 | 规则 |
|---|---|---|
| `session_id` | `basket_name` | 生成稳定、可追踪且长度受控的会话/策略分组名称 |
| `intent_id` | `external_id` | 平台幂等主键；提交、对账和重启恢复均优先使用该字段 |
| `quantity` | `order_vol` | A 股按 100 股整数手校验 |
| `broker_symbol` | `stock_code` | 通过显式代码映射表转换，不在适配器中猜测 |
| `execution_window.start` | `begin_time` | 使用 `HHMMSS` 字符串 |
| `execution_window.end` | `end_time` | 使用 `HHMMSS` 字符串 |
| `side=buy` | `bs_flag=BUY` | 首发只允许普通买入 |
| `side=sell` | `bs_flag=SELL` | 首发只允许普通卖出 |
| `trade_account` | `trade_acc` | 必须属于模拟账户白名单 |
| `execution.algorithm` | `strategy_type` | 首发默认 `TWAP`；只允许客户端联调确认的算法白名单 |
| `execution.algorithm_params` | `algo_param` | 由结构化参数安全序列化；禁止透传任意字符串 |
| `false` | `reach_limit_continue` | 首发固定为 false，涨跌停时不继续交易 |
| `false` | `over_time_continue` | 首发固定为 false，超过结束时间不继续交易 |

v0.0.23 允许通过 `algo_param` 传入 `limit_price` 等算法参数，但策略中的 `signal_price` 不得自动变成限价。首发默认不设置 `limit_price`；只有设计文档明确要求、用户确认且目标算法支持时，平台才从结构化配置生成该参数。最终成交价始终读取子单或成交回报中的 `trade_price`。

`external_id` 是首发幂等主键。真实联调必须验证它在上传响应、账户母单查询和 WebSocket Mudan 推送中完整往返，并确认最大长度和唯一性范围。

### 4.4 状态归一化

平台内部订单状态统一为：

```text
created
  -> validated | rejected
  -> submitting
  -> submission_unknown
  -> submitted
  -> working
  -> pause_pending
  -> paused
  -> partially_filled
  -> cancel_pending
  -> stopping
  -> completed
  -> filled
  -> cancelled
  -> stopped
  -> expired
  -> residual
  -> failed
  -> unknown
```

算法母单状态归一化规则：

- `0 Init` 映射为 `submitted`。
- 母单 `status=1` 映射为 `working`。
- 母单 `status=2` 映射为 `paused`。
- 母单 `status=3` 映射为 `completed`；成交量完整时同时标记成交结果 `filled=true`。
- 母单 `status=4` 映射为 `cancelled`。
- 母单 `status=5` 映射为 `expired`。
- 母单 `status=6` 映射为 `failed`。
- 母单 `status=7` 映射为 `stopping`。
- 母单 `status=8` 映射为 `cancel_pending`。
- 母单 `status=9` 映射为 `pause_pending`。
- 母单 `status=10` 映射为 `stopped`。
- 母单 `status=11` 映射为 `residual`，需要人工关注零碎股处理。
- T0 专用状态 `21-25` 不在首发范围，保留原始值并按 `unknown` 处理，不能触发 A 股普通算法母单的自动动作。

算法子单状态归一化规则：

- `0 Init` 映射为 `submitted`。
- `1 Insert` 映射为 `working`。
- `2 PartTrade` 映射为 `partially_filled`。
- `3 PartCancel` 映射为 `cancelled`，同时保留已成交数量。
- `4 AllTrade` 映射为 `filled`。
- `5 AllCancel` 映射为 `cancelled`。
- `6 Error`、`7 Invalid`、`9 FtError` 映射为 `failed`。
- `8 Finish` 映射为 `expired`。

普通委托查询返回的 0-6 子单状态属于另一套枚举。若未来启用 `direct_order`，必须使用独立映射表：`0 未报 -> submitted`、`1 已报 -> working`、`2 部分成交 -> partially_filled`、`3 全部成交 -> filled`、`4 已撤 -> cancelled`、`5 废单 -> failed`、`6 待撤 -> cancel_pending`。平台必须先按执行路由选择状态命名空间，禁止按算法子单状态解释普通委托。

所有状态都保存 `raw_status`、`raw_status_msg`、接口来源和采集时间。订单状态以成交事实优先，其次是子单，最后是母单状态。WebSocket 和轮询结果进入同一个归一化入口。

### 4.5 认证与客户端版本约束

- 首发统一使用 v1 登录 token，与 v1 订单接口保持同一认证体系。
- 登录响应中的 `data.accs.login_status` 必须为 true；该字段为 false 的交易账户不能进入资金、持仓和下单预检。
- `query_acc_status` 的 `login_status=1` 和 `order_engine_status=1` 仍是下单前第二层连接检查。
- v1 登录密码默认按原值发送；券商版本要求 MD5 时，必须通过显式 `password_transform=md5_32_lower` 配置转换，禁止自动猜测。
- v2 JWT 不用于首发算法母单链路；它只在后续普通委托查询能力中启用。
- 非凸账户名和密码可以从环境变量/系统安全存储读取，也可以由用户在 Web 工作台一次性输入。
- Web 输入的密码只允许通过同机 loopback autostrategy API 传递，并只以掩码类型保存在当前服务进程内存；页面刷新或服务重启后必须重新输入。
- 配置文件只保存环境变量名称和非敏感连接参数，不保存账号密码值。
- 由于 v1 登录将密码置于 query string，HTTP 客户端、异常信息和日志必须对 URL 参数完全脱敏。
- `code=1001` 时允许重新登录一次；重新登录失败后会话进入 `needs_attention`，禁止继续提交。
- 使用拆单监控接口时客户端版本必须不低于 3.11.4。现有文档没有版本查询接口，因此版本通过受控配置和联调清单确认；无法确认时预检返回 `client_version_unverified`，客户端模拟盘会话不得启动。

### 4.6 模拟账户硬边界

当前文档没有返回可靠的“模拟/实盘”字段，因此平台采用双重门禁：

1. 交易账户 ID 必须存在于本地配置的 `allowed_simulation_accounts` 白名单。
2. 启动 `manual` 或 `auto` 会话时，用户必须再次确认显示的账户 ID 和账户昵称。

不在白名单中的账户，即使客户端登录成功，也只能查看连接结果，不能查询资产详情、提交订单或操作母单。产品不得提供绕过该门禁的前端按钮或普通 API 参数。

长期建议由非凸 API 增加不可伪造的 `account_environment=simulation|production` 字段；在该字段可用前，白名单是发布的硬前提。

### 4.7 资金字段归一化

v0.0.23 将股票账户资金字段说明更新为：

- `available`: 可用资金，是买入风控的现金上限。
- `frozen`: 冻结资金，只展示和审计，不计入可用资金。
- `balance`: 余额。
- `asset`: 总资产。

平台规范化账户快照同时保留 `balance` 和 `asset`。风险仓位比例使用 `asset > 0` 时的 `asset`；若 `asset` 为 0 而 `balance > 0`，允许以 `balance` 作为降级的 `risk_equity`，但必须记录 `fund_asset_fallback` 诊断并在 UI 显示。两个字段均非正数时禁止买入。

拆单监控响应按账户或 `basket_name` 保存以下字段，不重命名原始字段：

| 原始字段 | 平台含义 |
|---|---|
| `plan_buy` | 计划买入金额 |
| `plan_sale` | 计划卖出金额 |
| `trade_buy` | 已成交买入金额 |
| `trade_sale` | 已成交卖出金额 |
| `buy_rate` | 买入完成率 |
| `sale_rate` | 卖出完成率 |
| `exposure` | 当前执行暴露 |
| `cancel_rate` | 撤单率 |
| `total_rate` | 总完成率 |
| `error_rate` | 错废单率 |

平台解析 `trade_acc_infos` 和 `basket_infos` 两个数组，同时保存原始响应、作用范围和采集时间。根据文档公式和示例，完成率、撤单率、总成交率和错废单率按 0-1 原始比率处理，UI 乘以 100 后展示百分比；`exposure` 按 `(买交易额 - 卖交易额) / (买交易额 + 卖交易额)` 展示有符号百分比。超出预期范围或分母为零的异常值保留原值并显示数据诊断，不得静默修正。上述指标只用于执行质量监控，不能替代资金、持仓、母单或子单事实。

---

## 5. 策略执行契约

### 5.1 设计原则

- 生成策略只负责计算，不负责网络通信和订单生命周期。
- 账户资金和持仓通过只读上下文提供给策略。
- 策略不得假设意图已经成交。
- 只有成交回报可以改变客户端模拟盘持仓事实。
- 风控可以缩小或拒绝订单，但不能放宽 `STRATEGY_DESIGN.md` 中定义的限制。

### 5.2 新增标准入口

今后支持客户端模拟盘的策略必须暴露：

```python
def generate_intents(context: dict) -> dict:
    """Return decisions, order intents, and the next serializable strategy state."""
```

输入 `context` 至少包含：

```json
{
  "session": {
    "session_id": "sess_...",
    "mode": "ft_client_simulation",
    "execution_mode": "auto",
    "now": "2026-08-11T09:35:00+08:00"
  },
  "market": {
    "bars_by_symbol": {},
    "history_by_symbol": {},
    "completed_bar_at": "2026-08-10T15:00:00+08:00"
  },
  "account": {
    "trade_account": "SIM_ACCOUNT_ID",
    "available_cash": 1000000,
    "positions": []
  },
  "strategy_state": {},
  "config": {}
}
```

返回结构：

```json
{
  "decisions": [
    {
      "symbol": "588000.SH",
      "action": "buy",
      "signal_price": 1.08,
      "reason": "grid -1 below base 1.12"
    }
  ],
  "intents": [
    {
      "intent_key": "588000.SH:2026-08-10:grid:-1:buy",
      "symbol": "588000.SH",
      "side": "buy",
      "quantity": 46000,
      "signal_price": 1.08,
      "reason": "grid -1 below base 1.12",
      "execution_window": {
        "start": "093500",
        "end": "145000"
      }
    }
  ],
  "strategy_state": {}
}
```

### 5.3 幂等键

策略提供业务稳定的 `intent_key`，平台生成最终 `intent_id`：

```text
sha256(strategy_slug + strategy_version + trade_account + intent_key)
```

同一个模拟账户上，同一个 `intent_id` 跨会话只允许存在一个有效母单。平台提交时固定令 `external_id=intent_id`，本地去重表以 `trade_account + external_id` 建立唯一约束。提交超时或响应丢失时，不得直接重发 POST；必须先查询账户母单并按 `external_id` 对账，再使用已记录的母单 ID、`basket_name` 和提交时间窗口辅助核验。只有确认客户端不存在对应母单后，才能由对账器补交。用户确需再次执行同一业务信号时，必须产生新的、可审计的 `intent_key`，不能通过重启或新建会话绕过去重。

真实客户端尚未验证 `external_id` 最大长度和唯一性范围时，预检返回 `external_id_unverified`，`manual` 和 `auto` 均不得启动。平台不得截断哈希；如客户端长度有限，应在联调后确定一种稳定的短 ID 编码，并把编码版本纳入会话快照。

### 5.4 兼容现有策略

- `run_backtest(config)` 保持不变。
- `run_paper(config)` 继续服务于 `local_replay`。
- 旧策略没有 `generate_intents(context)` 时，客户端模拟盘入口显示“不兼容”，并提供重新生成代码操作。
- 代码生成 Agent 必须从 `STRATEGY_DESIGN.md` 同时生成 `run_backtest`、`run_paper` 和 `generate_intents`；`config.yaml` 顶层 `symbols` 必须列出有限的可下单证券池，行情基准指数单独配置，不得混入 `symbols`。不得在 `generate_intents` 中嵌入凭证、URL 或客户端调用。

### 5.5 当前动态网格策略迁移

当前策略内部维护的 `cash`、`positions` 和假定成交逻辑必须移出客户端模拟路径。迁移后的策略状态只保留策略自身信息，例如：

- 每个标的的历史 bar。
- 当前网格线和最近一次重平衡日期。
- 网格层级与对应的已确认持仓关联。
- 连续亏损次数和暂停截止日期。

持仓数量、可卖数量、现金、成交均价和订单在途数量由账户上下文和客户端回报提供。

首发启用沪深市场可交易 A 股与 ETF。产品级分类器验证交易所、证券代码和资产类型；每个策略会话另外使用显式 `allowed_symbols` 和 `symbol_mapping` 限制实际执行范围。客户只需配置该策略可能下单的标的，不需要录入全市场证券。动态选股策略必须在启动前固化本会话最大执行股票池，运行中生成的意图不得越过该集合。

---

## 6. 系统架构

```text
Web UI / REST API
        |
SimulationSessionService
        |
        +-- PreflightService
        |     +-- account whitelist
        |     +-- client status
        |     +-- strategy compatibility
        |     +-- data freshness
        |     +-- risk configuration
        |
        +-- StrategyRuntime
        |     +-- market context
        |     +-- generate_intents(context)
        |     +-- strategy state
        |
        +-- OrderIntentService
        |     +-- normalize
        |     +-- risk gate
        |     +-- idempotency
        |
        +-- BrokerAdapter
        |     +-- LocalPaperBroker
        |     +-- FtClientSimulationBroker
        |
        +-- OrderReconciler
        |     +-- WebSocket events
        |     +-- polling fallback
        |     +-- account snapshots
        |
        +-- SessionStore / Artifacts
```

### 6.1 组件职责

| 组件 | 职责 |
|---|---|
| `SimulationSessionService` | 创建、启动、暂停、恢复、停止和恢复交易会话 |
| `PreflightService` | 在任何订单提交前完成全部硬校验 |
| `StrategyRuntime` | 构建上下文、调用策略、保存策略状态 |
| `OrderIntentService` | 规范化意图、计算幂等 ID、执行风控 |
| `BrokerAdapter` | 定义登录、账户、订单、撤单、执行监控和事件的统一接口 |
| `FtClientSimulationBroker` | 实现非凸客户端 v1 HTTP、算法母单和 WebSocket 协议；首发禁用普通委托路由 |
| `OrderReconciler` | 合并推送与轮询结果，维护最终一致的订单状态 |
| `SessionStore` | 原子保存会话状态并追加不可变事件 |

### 6.2 BrokerAdapter 能力契约

适配器至少提供：

```text
connect()
disconnect()
health()
list_accounts()
get_funds(account)
get_positions(account)
submit_order(intent)
cancel_orders(order_ids)
get_orders(account, filters)
get_child_orders(parent_order_id)
get_monitoring(ft_account)
stream_events()
```

`get_monitoring(ft_account)` 一次读取该非凸总账户下的 `trade_acc_infos` 和 `basket_infos`，由平台按交易账户和 `basket_name` 过滤。`submit_order(intent)` 内部按 `execution_route` 分派。首发只注册 `algorithm_parent`，映射到 `api_upload_mudan`；`direct_order` 是禁用的能力标识，不能通过配置绕过预检。平台服务只依赖此契约，不直接依赖非凸响应字段。

---

## 7. 会话生命周期

### 7.1 会话状态

```text
draft
  -> preflight_failed
  -> ready
  -> starting
  -> running
  -> paused
  -> stopping
  -> stopped
  -> completed
  -> needs_attention
  -> failed
```

### 7.2 启动前检查

所有项目必须通过：

1. 策略状态至少为 `backtested`。
2. 策略存在 `generate_intents(context)`。
3. 非凸客户端基地址是 loopback 地址。
4. 已通过受控配置确认客户端版本不低于 3.11.4；无法确认时返回 `client_version_unverified`。
5. 密码转换配置只能是 `plain` 或 `md5_32_lower`，且凭证来源和 URL 脱敏检查通过。
6. v1 登录成功，登录响应中目标交易账户的 `data.accs.login_status=true`，且账户在模拟账户白名单。
7. `query_acc_status` 返回目标账户 `login_status=1` 且 `order_engine_status=1`。
8. 资金和持仓查询成功，资金字段能够生成有效 `risk_equity`。
9. `execution_route=algorithm_parent`；`strategy_type` 在已联调算法白名单中，结构化算法参数可安全序列化，`reach_limit_continue=false` 且 `over_time_continue=false`。
10. `external_id` 的长度、字符集、唯一性范围和查询往返规则已经联调确认，生成的 ID 不需要截断。
11. 策略只包含首发支持的 A 股标的。
12. 每个标的存在已确认的客户端代码映射和 100 股手数规则。
13. 行情数据已完成、未过期且没有时间倒退。
14. 风控上限完整，且不宽于策略设计文档。
15. 同一账户没有其他活动策略会话。
16. 系统时间和交易日历使用 `Asia/Shanghai`。

任何一项失败都返回结构化原因，不允许“忽略并继续”。

### 7.3 暂停、恢复与停止

- **暂停:** 停止产生和提交新意图；不自动取消已工作的母单；继续同步订单、成交、资金和持仓。
- **恢复:** 重新执行账户状态、数据新鲜度和幂等检查后恢复信号计算。
- **停止:** 停止新意图，对本会话全部未完成母单发起取消，继续对账直到终态或达到 30 秒等待上限，然后持久化结果。
- **紧急停止:** 与停止相同，但跳过正常调度等待，立即取消本会话全部可取消母单。

停止操作不得调用“按账户取消所有母单”的宽范围接口，除非能够证明该账户只存在本会话订单。默认使用已记录的母单 ID 精确取消。

### 7.4 重启恢复

应用启动时扫描状态为 `starting`、`running`、`paused`、`stopping` 或 `needs_attention` 的会话：

1. 重新登录客户端。
2. 查询资金、持仓、母单和子单。
3. 优先按 `external_id` 关联本地意图与客户端母单，再以母单 ID、`basket_name` 和时间窗口辅助核验并重建内部订单状态。
4. 对已提交但响应未知的意图执行对账。
5. 恢复 WebSocket 订阅。
6. 默认以 `paused` 恢复，等待用户确认；配置显式允许 `auto_resume=true` 时才恢复自动提交。

恢复过程中不调用策略生成新意图，直到所有旧订单完成对账。

---

## 8. 行情与调度

### 8.1 首发行情口径

- 使用 FTShare 日线接口获取截至上一交易日的历史数据，供 MA250 等中长期状态计算。
- 使用 FTShare 股票、ETF 和指数 `prices?since=TODAY` REST 接口获取当日一分钟价格点，并聚合为 10 分钟 OHLCV。
- 上午以 `09:30` 为分桶锚点，下午以 `13:00` 为分桶锚点；任何 K 线不得跨越 `11:30-13:00` 午休。
- 只发布已经完成的 10 分钟 K 线；当前未完成桶不得进入 `bars_by_symbol`。
- 执行标的、基准指数及策略声明的其他行情依赖必须取时间戳交集，只使用所有标的共有的最新完成桶。
- 每个 bar 以 `symbol + completed_bar_at` 唯一标识。
- 行情约每 30 秒轮询一次；新完成桶到达后先校验时间递增，再触发一次策略计算，同一桶最多评估一次。
- `history_by_symbol` 保持日线语义，`intraday_history_by_symbol` 保存 10 分钟历史，`bars_by_symbol` 保存共同时间戳上的最新 10 分钟 bar，禁止把 250 根 10 分钟 K 线解释为 MA250 日线。

### 8.2 当前策略执行时序

默认盘中时序：

```text
交易日 09:40 确认 09:30-09:39 K 线完成
  -> 同步账户资金和持仓
  -> 使用截至上一交易日的日线计算 MA250 等状态
  -> 使用 09:40 完成的共同 10 分钟 K 线作为触发时间与执行参考价
  -> 生成订单意图并风控
  -> 按执行模式观察、确认或自动提交
随后每形成一根新的共同 10 分钟 K 线
  -> 重复账户同步、策略评估和意图处理
  -> 相同 completed_bar_at 不重复计算或下单
```

休市、停牌、行情缺失、bar 未完成或数据时间倒退时不产生订单。

### 8.3 实时能力边界

本版本的“实时”定义为在完整 10 分钟 K 线形成后由 30 秒级轮询发现并评估，不承诺逐笔、秒级或未完成 K 线内触发。账户、母单、子单和成交仍通过 WebSocket 与约 3 秒轮询补偿同步，不受行情轮询周期影响。

---

## 9. 风控需求

### 9.1 账户级硬风控

- 目标账户必须在 `allowed_simulation_accounts`。
- 单账户只允许一个活动策略会话。
- 可用资金查询失败时禁止买入。
- 可用持仓查询失败时禁止卖出。
- 客户端或下单引擎异常时禁止提交。
- 连接恢复后必须重新做账户快照和订单对账。

### 9.2 订单级硬风控

- 只允许 `BUY` 和 `SELL`。
- 标的必须是沪深市场可交易 A 股或 ETF，并存在于当前会话的 `allowed_symbols` 和 `symbol_mapping`。
- 数量必须大于 0 且为 100 的整数倍。
- 买入估算金额不得超过可用资金。
- 卖出数量不得超过 `avail_vol`。
- 单笔金额不得超过账户总资产的 5%。
- 单标的持仓不得超过账户总资产的 20%。
- 总持仓不得超过账户总资产的 80%。
- 每个标的每天最多提交一个同方向、同 `intent_key` 的母单。
- 买入当日新增持仓不得用于当日卖出。
- `reach_limit_continue` 固定为 false。
- `over_time_continue` 固定为 false。

策略文档如果配置了更严格阈值，使用更严格值。UI 只能收紧阈值，不能放宽到高于设计文档。

### 9.3 自动模式门禁

切换到 `auto` 前必须显示并确认：

- 策略名称与版本。
- 模拟账户 ID 与昵称。
- 允许标的。
- 单笔、单标的和总仓位上限。
- 母单执行时间窗口。
- 停止行为。

确认只对当前策略版本和当前账户有效。策略代码、设计文档或风险配置变化后，自动模式授权失效并退回 `observe`。

---

## 10. 网络、重试与一致性

### 10.1 HTTP 规则

- 连接超时 3 秒，单次读取超时 10 秒。
- GET 查询失败采用 1、2、5 秒退避，最多三次。
- POST 上传母单不做无条件自动重试。
- POST 响应超时先进入 `submission_unknown`，随后查询账户母单并优先按 `external_id` 确认；母单 ID、`basket_name` 和提交时间窗口只做辅助核验。
- 对账尚未证明不存在对应 `external_id` 时禁止再次提交；禁止因 HTTP 断线、进程重启或切换会话而盲目重试。
- 所有非零 `code` 转换为结构化错误，保留脱敏后的原始响应。

### 10.2 WebSocket 规则

- 收到 Ping 立即返回同数据的 Pong。
- Mudan、Zidan、Trade 事件进入统一对账器。
- 断线后以 1、2、5、10、30 秒退避重连。
- 断线期间每 3 秒轮询本会话未完成订单。
- WebSocket 恢复后先执行一次完整订单查询，再恢复仅事件驱动更新。

### 10.3 数据一致性

- 资金、持仓和订单快照带采集时间。
- 旧快照不能覆盖新快照。
- 所有内部状态写入使用临时文件加原子替换。
- JSONL 事件只追加，不原地修改。
- 原始客户端 ID、状态码和时间戳必须保留，便于审计。

---

## 11. 持久化与 artifacts

每个客户端模拟盘会话使用独立目录：

```text
paper_run/client_sessions/<session_id>/
  session.json
  strategy_state.json
  events.jsonl
  order_intents.jsonl
  broker_orders.jsonl
  child_orders.jsonl
  fills.jsonl
  account_snapshots.jsonl
  monitoring_snapshots.jsonl
  reconciliation.jsonl
  logs/session.log
  review.md
```

### 11.1 `session.json`

至少保存：

- 会话 ID、策略 slug 和策略版本摘要。
- 模式、执行模式、状态和状态时间。
- 脱敏账户标识、broker ID 和客户端基地址。
- 标的白名单、代码映射和风控配置快照。
- 执行路由、`strategy_type`、结构化 `algo_param`、`reach_limit_continue` 和 `over_time_continue` 快照。
- 最低客户端版本要求、实际/人工确认的客户端版本及确认来源。
- `external_id` 编码版本和已确认的长度、字符集、唯一性范围。
- 最近处理 bar、最近对账时间和最近错误。
- 活动母单 ID 列表。
- 自动恢复授权状态。

不得保存密码、token、完整登录 URL 或任何凭证值。

### 11.2 复盘内容

`review.md` 至少包含：

- 会话时间、策略版本、账户和执行模式。
- 信号数量、通过/拒绝意图数量及拒绝原因。
- 母单、子单、成交和撤单数量。
- 信号价格与实际成交均价偏差。
- 账户和篮子维度的买卖完成率、暴露、撤单率、总完成率和错废单率摘要。
- 起止资产、已实现/未实现盈亏和最大回撤。
- WebSocket 断线、重试、未知状态和人工干预记录。

---

## 12. autostrategy REST API

新增 API 采用 `/api/v1`：

| 方法 | 路径 | 用途 |
|---|---|---|
| `POST` | `/broker-connections/ft-client/check` | 脱敏连接检查 |
| `GET` | `/broker-connections/ft-client/accounts` | 返回可用且经过白名单过滤的账户 |
| `POST` | `/strategies/{slug}/client-simulation/preflight` | 返回全部预检结果 |
| `POST` | `/strategies/{slug}/client-simulation/sessions` | 创建并启动会话 |
| `GET` | `/strategies/{slug}/client-simulation/sessions` | 列出历史会话 |
| `GET` | `/strategies/{slug}/client-simulation/sessions/{session_id}` | 获取会话状态、客户端版本/账户登录状态、执行质量指标和摘要 |
| `POST` | `/strategies/{slug}/client-simulation/sessions/{session_id}/pause` | 暂停新信号 |
| `POST` | `/strategies/{slug}/client-simulation/sessions/{session_id}/resume` | 重新预检并恢复 |
| `POST` | `/strategies/{slug}/client-simulation/sessions/{session_id}/stop` | 停止并撤销未完成母单 |
| `POST` | `/strategies/{slug}/client-simulation/sessions/{session_id}/intents/{intent_id}/approve` | 人工确认意图 |
| `POST` | `/strategies/{slug}/client-simulation/sessions/{session_id}/intents/{intent_id}/reject` | 人工拒绝意图 |
| `GET` | `/strategies/{slug}/client-simulation/sessions/{session_id}/events` | 获取决策、订单和成交事件 |
| `GET` | `/strategies/{slug}/client-simulation/sessions/{session_id}/account` | 获取最新资金、持仓、账户登录状态和账户级拆单监控快照 |

API 响应只包含脱敏账户信息。连接检查 API 可以接收 Web 工作台提交的一次性账号和密码，但仅限同机 loopback 部署；请求体不得记录到 access log，密码转换完成后只保留内存中的掩码对象，任何响应、异常、artifact 和日志都不得包含密码或 token。无 UI 场景仍可由后端从配置的安全来源读取凭证。

---

## 13. Web 工作台需求

### 13.1 模拟运行入口

策略详情页将现有“模拟运行”拆分为：

- 本地历史回放。
- 非凸客户端模拟盘。

只有 `backtested` 且兼容新策略契约的策略可以进入客户端模拟盘配置。

### 13.2 客户端模拟盘页面

页面包含：

1. 客户端连接状态。
2. 客户输入的非凸账号、密码、密码处理方式和已确认客户端版本。
3. 客户输入或确认的模拟交易账户白名单。
4. 模拟账户选择、账户登录和下单引擎状态。
5. 执行模式选择。
6. 本次策略允许标的（逗号/换行列表）与客户端代码映射（JSON）输入；两者必须完整对应。
7. 算法类型、结构化算法参数、开始时间和结束时间输入。
8. 单笔、单标的和总仓位上限输入与摘要。
9. `external_id` 长度和查询往返确认项。
10. 预检结果列表。
11. 启动、暂停、恢复、停止和紧急停止操作。
12. 资金与持仓卡片。
13. 账户/篮子执行质量卡片：买卖完成率、总完成率、暴露、撤单率和错废单率。
14. 策略决策与订单意图列表。
15. 母单、子单和成交状态表。
16. WebSocket、轮询、错误和人工干预事件时间线。

### 13.3 状态文案

- 页面持续显示“非凸客户端模拟账户，不是本地回放”。
- 未确认模拟账户身份时不得出现可点击的自动提交按钮。
- `submission_unknown`、`unknown` 和 `needs_attention` 必须使用醒目告警，不能显示为成功。
- `client_version_unverified`、账户 `login_status=false` 和 `fund_asset_fallback` 必须显示明确原因；前两者禁止启动客户端模拟盘会话。
- 信号价格和成交价格分列展示。

---

## 14. 配置模型

建议用户级配置：

```yaml
broker_connections:
  ft_client:
    enabled: true
    base_url: "http://127.0.0.1:11356"
    min_client_version: "3.11.4"
    confirmed_client_version: null
    ft_account_env: "AUTOSTRATEGY_FT_ACCOUNT"
    password_env: "AUTOSTRATEGY_FT_PASSWORD"
    password_transform: "plain"
    allowed_simulation_accounts:
      - "SIM_ACCOUNT_ID"
    allowed_symbols:
      - "600519.SH"
      - "510500.SH"
    symbol_mapping:
      "600519.SH": "CONFIRMED_CLIENT_CODE"
      "510500.SH": "CONFIRMED_CLIENT_CODE"
    poll_interval_seconds: 3
    auto_resume: false
```

策略工作区配置只保存非敏感执行参数：

```yaml
client_simulation:
  execution_mode: observe
  execution_route: algorithm_parent
  execution_window:
    start: "093500"
    end: "145000"
  algorithm:
    strategy_type: "TWAP"
    params: {}
    reach_limit_continue: false
  over_time_continue: false
  risk:
    max_order_pct: 5
    max_symbol_position_pct: 20
    max_total_position_pct: 80
```

`confirmed_client_version` 必须来自受控安装记录或联调确认，不允许仅由浏览器请求声明。`CONFIRMED_CLIENT_CODE` 必须在联调后替换为非凸客户端实际接受的股票代码，不允许带着示例值启动会话。`algorithm.params` 只接受各算法白名单 schema 中定义的结构化字段，由适配器生成 `algo_param`；禁止用户直接填写分号拼接字符串。

---

## 15. 错误分类

| 分类 | 示例 | 会话行为 |
|---|---|---|
| `configuration_error` | 缺少凭证环境变量 | 预检失败 |
| `account_not_allowed` | 账户不在模拟白名单 | 硬阻断 |
| `client_unavailable` | 11356 端口不可达 | 暂停并重连 |
| `authentication_error` | token 失效且重新登录失败 | `needs_attention` |
| `account_login_failed` | 登录响应中目标账户 `login_status=false` | 预检失败 |
| `client_version_unverified` | 无法证明客户端版本不低于 3.11.4 | 客户端模拟盘预检失败 |
| `order_engine_unavailable` | `order_engine_status=0` | 暂停提交 |
| `algorithm_config_invalid` | 路由、算法名或结构化参数不在白名单 | 预检失败 |
| `external_id_unverified` | 长度、唯一性范围或查询往返尚未确认 | `manual`/`auto` 预检失败 |
| `fund_asset_fallback` | `asset` 无效而使用 `balance` 作为风险资产基数 | 允许继续但显示诊断 |
| `market_data_stale` | 任一策略行情标的缺少共同完成的 10 分钟 K 线或时间倒退 | 不生成意图 |
| `risk_rejected` | 资金、持仓或仓位超限 | 拒绝单个意图 |
| `submission_unknown` | POST 超时且尚未查到母单 | 禁止重发并持续对账 |
| `broker_rejected` | 母单或子单错误 | 记录失败，不自动放宽条件 |
| `reconciliation_error` | 订单状态互相矛盾 | `needs_attention` |
| `strategy_error` | `generate_intents` 抛出异常 | 暂停会话，保留客户端订单同步 |

策略异常、行情异常或 UI 关闭都不能停止订单对账；只有明确结束会话后才结束对账进程。

---

## 16. 测试策略

### 16.1 单元测试

- 非凸响应解析和错误码映射。
- 母单 0-11、21-25 与算法子单 0-9 的完整状态归一化和原始值保留。
- 普通委托 0-6 状态使用独立命名空间，不能复用算法子单映射。
- 登录响应中账户 `login_status=false` 的硬阻断。
- `plain`/`md5_32_lower` 密码转换、登录 URL 和异常信息脱敏。
- 客户端最低版本门禁和 `client_version_unverified`。
- 账户白名单硬阻断。
- `asset` 作为风险资产基数及 `balance` 降级和 `fund_asset_fallback` 诊断。
- 股票代码、手数、资金、持仓、T+1 和仓位校验。
- `intent_id` 稳定性、`external_id` 完整往返、跨会话唯一约束和重复意图拒绝。
- `strategy_type` 白名单、结构化 `algo_param` 序列化和 continue 开关固定值。
- 拆单监控账户/篮子响应解析和比率字段展示。
- 一分钟价格聚合为 10 分钟 OHLCV、未完成桶过滤和午休边界。
- 多标的共同完成时间、同一桶只评估一次和后台基准标的不丢失。
- 日线 MA250 历史与 10 分钟触发/定价上下文保持独立。
- 日志和异常 URL 凭证脱敏。
- 策略 `generate_intents` 契约校验。

### 16.2 集成测试

CI 使用本地 fake FT Client，不连接真实客户端。fake server 必须支持：

- v1 登录和 token 失效。
- 登录账户级 `login_status`、资金字段语义、持仓和账户状态。
- 上传母单成功、失败和响应超时。
- `external_id` 在上传、母单查询和 WebSocket 中完整往返，以及重复提交对账。
- 部分成交、全成、撤单和错单。
- 拆单监控账户/篮子指标。
- WebSocket Ping/Pong、断线和重连。
- 查询补偿和重启恢复。

### 16.3 API 与前端测试

- 连接检查不泄露凭证。
- 预检失败时禁止启动。
- 观察、人工确认和自动模拟三种流程。
- 暂停、恢复、停止和紧急停止。
- 资金、持仓、信号、订单和成交展示。
- `unknown` 和 `needs_attention` 告警。

### 16.4 真实客户端验收

仅使用已确认的非凸模拟账户：

1. 启动非凸客户端并登录模拟环境。
2. 核对客户端版本不低于 3.11.4，并保存版本和确认来源。
3. autostrategy 登录检查通过，登录响应账户 `login_status=true`，账户状态和下单引擎状态正常。
4. 核对 `asset`、`balance`、`available` 与客户端界面一致，并验证风险资产基数降级行为。
5. 确认首发允许的 `strategy_type` 和每个 `algo_param` 的名称、格式与效果。
6. 选取一只沪深 A 股和一只 ETF，分别完成代码映射查询和小额母单验证。
7. 验证 `external_id` 在上传响应、账户母单查询和 WebSocket Mudan 推送中完整往返，并确认长度和唯一性范围。
8. 调用拆单监控接口，核对账户/篮子的完成率、暴露、撤单率和错废单率展示。
9. 动态网格策略先运行 `observe`，确认意图正确。
10. 切换 `manual`，分别完成一笔买入和一笔卖出。
11. 切换 `auto`，验证自动提交、部分成交/成交、持仓更新。
12. 人为断开 WebSocket，验证轮询补偿和恢复。
13. 在提交响应未知场景重启 autostrategy，验证按 `external_id` 恢复且不重复下单。
14. 停止会话，验证未完成母单撤销且不再产生新订单。

任何真实客户端验收步骤都不得使用非白名单账户。

---

## 17. 总体验收标准

### 17.1 第 1 期：连接与账户

- [ ] 可检查客户端连接、账户登录和下单引擎状态。
- [ ] 可验证客户端不低于 3.11.4；版本不明确时客户端模拟盘会话被阻断。
- [ ] 只能选择白名单模拟账户。
- [ ] 可显示资金和持仓，且不泄露凭证。

### 17.2 第 2 期：订单意图与人工控制

- [ ] 新生成策略具备 `run_paper(config)` 和 `generate_intents(context)`，并显式声明顶层 `symbols` 执行股票池。
- [ ] 观察模式不产生客户端订单。
- [ ] 人工确认后可精确提交单个母单。
- [ ] 所有拒绝都有可解释原因。

### 17.3 第 3 期：自动执行与可靠性

- [ ] 自动模式通过风控后提交母单。
- [ ] WebSocket 和轮询能同步母单、子单和成交。
- [ ] `external_id` 可用于跨会话幂等、提交未知对账和重启恢复。
- [ ] 可获取并展示账户/篮子拆单监控指标，且不将其误作资金或订单事实。
- [ ] 网络重试、响应丢失和应用重启不会重复下单。
- [ ] 暂停、恢复、停止、撤单和紧急停止符合设计。

### 17.4 第 4 期：当前策略端到端

- [ ] 当前策略声明最大执行股票池，并在客户确认的沪深 A 股/ETF 白名单上完成迁移。
- [ ] 使用所有策略行情标的共有的最新已完成 10 分钟 K 线触发评估，同一 K 线最多产生一次评估。
- [ ] 当前策略继续使用截至上一交易日的日线计算 MA250，并使用最新完成 10 分钟 ETF K 线定价，不混淆两个周期。
- [ ] 仓位和资金以非凸客户端为准。
- [ ] 至少一只沪深 A 股和一只 ETF 完成信号、母单、子单、成交、持仓和复盘闭环。

### 17.5 发布闸门

- [ ] 全部单元、集成、API 和前端自动化测试通过。
- [ ] fake FT Client 端到端测试通过。
- [ ] 非凸真实模拟客户端验收清单通过。
- [ ] 未发现密码、token 或登录 URL 泄露。
- [ ] 非白名单账户启动测试被硬阻断。
- [ ] README 和 UI 明确区分本地回放与客户端模拟盘。

---

## 18. 外部联调前提

实施可以在 fake FT Client 上完成，但真实客户端验收依赖以下资料：

1. 一个专用非凸模拟账户 ID、账户昵称和 broker ID。
2. 客户端版本不低于 3.11.4，且确认与 API v0.0.23 的接口和字段行为一致。
3. 代表性沪深 A 股和 ETF 在上传母单接口中的准确 `stock_code` 格式及映射规则。
4. 首发获准使用的算法名称，以及每种 `strategy_type` 的 `algo_param` 参数 schema、取值范围和默认行为。
5. `external_id` 的最大长度、允许字符、唯一性范围，以及上传响应、查询和 WebSocket 中的完整往返行为。
6. 模拟账户算法母单的部分成交、涨跌停、超时和结束时间行为。
7. WebSocket Trade 消息的完整实际字段样例。
8. 算法子单各状态码在当前客户端版本中的实际流转样例，特别是 `Finish`、`FtError` 和部分撤单。
9. 股票账户 `balance` 与 `asset` 在当前券商版本中的界面对应和异常值语义。
10. 是否能由 API 返回可靠的模拟/实盘账户环境标识。

这些资料缺失不影响适配器、fake server、策略契约和 UI 的开发，但缺失任一项时不得宣称真实客户端端到端验收完成。

---

## 19. 对现有系统的兼容要求

- 现有 `/paper-run` API 和本地 replay artifacts 保持可用。
- 现有 `PaperAccount` 只用于本地回放，不参与客户端模拟账户记账。
- 现有策略没有新入口时，回测和本地回放不受影响。
- 客户端模拟盘使用独立会话目录，不覆盖现有 `paper_run_result.json`。
- Web 工作台已有策略设计、代码生成和回测流程不改变审批顺序。
- 客户端模拟盘不改变 `STRATEGY_DESIGN.md` 中的核心策略逻辑，只改变执行与账户状态来源。

---

## 20. 需求确认点

本设计采用以下已经收敛的产品决策：

1. 一次性交付第 1-4 期，但内部仍按四个可测试里程碑实施。
2. 首发支持沪深 A 股和 ETF 的 10 分钟模拟交易，指数可作为行情基准；每个会话必须固化客户确认的最大执行标的白名单和代码映射。
3. 只支持白名单模拟账户，不提供实盘开关。
4. 账户与成交以非凸客户端为唯一事实来源。
5. 以 API v0.0.23 和客户端 3.11.4 为最低接入基线。
6. 自动执行固定使用算法母单 `algorithm_parent` 路由，首发默认 `TWAP`；普通委托 `direct_order` 保留但禁用。
7. `external_id=intent_id` 是客户端侧幂等、提交未知对账和重启恢复的第一关联键。
8. 新生成策略同时具备本地回放 `run_paper(config)` 与纯计算 `generate_intents(context)` 契约。
9. 默认在所有策略行情标的形成共同的已完成 10 分钟 K 线后评估一次；中长期日线状态只读取截至上一交易日的数据，避免未来数据。
10. HTTP 提交、WebSocket 推送、轮询补偿、拆单监控和持久化恢复同时交付。
11. 真实客户端资料未确认前，可以完成开发，但不能完成最终发布验收。

用户确认本设计后，下一步生成覆盖四个里程碑的详细 TDD 实施计划；确认前不进入代码实现。
