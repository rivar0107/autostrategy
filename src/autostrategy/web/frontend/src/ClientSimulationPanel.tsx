import { useEffect, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Col,
  Descriptions,
  Form,
  Input,
  InputNumber,
  Radio,
  Row,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
  message,
} from 'antd'

import { ApiError, api } from './api/client'
import type {
  ClientSimulationPreflight,
  ClientSimulationRequest,
  ClientSimulationSession,
  FtAccount,
  FtConnectionInput,
  FtExecutionMode,
} from './types'

const { Text, Title } = Typography

interface ClientSimulationPanelProps {
  slug: string
}

interface ConnectionFields {
  baseUrl: string
  ftAccount: string
  password: string
  passwordTransform: 'plain' | 'md5_32_lower'
  clientVersion: string
  simulationAccounts: string
  allowedSymbols: string
  symbolMappingJson: string
  externalIdMaxLength: number
  externalIdConfirmed: boolean
}

interface ExecutionFields {
  tradeAccount: string
  executionMode: FtExecutionMode
  algorithm: string
  algorithmParams: string
  startTime: string
  endTime: string
  maxOrderPct: number
  maxSymbolPct: number
  maxTotalPct: number
  acknowledgeSimulation: boolean
}

function displayError(error: unknown): string {
  if (error instanceof ApiError) return `${error.code}: ${error.message}`
  if (error instanceof Error) return error.message
  return '操作失败'
}

function parseAlgorithmParams(value: string): Record<string, string | number | boolean> {
  const text = value.trim()
  if (!text) return {}
  const parsed = JSON.parse(text)
  if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') {
    throw new Error('算法参数必须是 JSON 对象。')
  }
  for (const [key, item] of Object.entries(parsed)) {
    if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(key)) {
      throw new Error(`算法参数名 ${key} 不合法。`)
    }
    if (!['string', 'number', 'boolean'].includes(typeof item)) {
      throw new Error(`算法参数 ${key} 只允许字符串、数字或布尔值。`)
    }
  }
  return parsed as Record<string, string | number | boolean>
}

export function parseAllowedSymbols(value: string): string[] {
  const symbols = Array.from(new Set(
    value
      .split(/[\s,，]+/)
      .map(item => item.trim().toUpperCase())
      .filter(Boolean),
  ))
  if (!symbols.length) throw new Error('请至少填写一个策略可能下单的标的。')
  const invalid = symbols.filter(symbol => !/^\d{6}\.(SH|SZ)$/.test(symbol))
  if (invalid.length) {
    throw new Error(`证券代码格式不正确：${invalid.join('、')}。请使用 600519.SH 格式。`)
  }
  return symbols
}

export function parseSymbolMapping(
  value: string,
  allowedSymbols: string[],
): Record<string, string> {
  let parsed: unknown
  try {
    parsed = JSON.parse(value.trim())
  } catch {
    throw new Error('客户端代码映射必须是有效的 JSON 对象。')
  }
  if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') {
    throw new Error('客户端代码映射必须是 JSON 对象。')
  }
  const mapping = Object.fromEntries(
    Object.entries(parsed).map(([key, item]) => [key.trim().toUpperCase(), String(item).trim()]),
  )
  if (Object.values(mapping).some(value => !value)) {
    throw new Error('客户端代码映射值不能为空。')
  }
  const mappingKeys = Object.keys(mapping).sort()
  const expectedKeys = [...allowedSymbols].sort()
  if (
    mappingKeys.length !== expectedKeys.length
    || mappingKeys.some((key, index) => key !== expectedKeys[index])
  ) {
    throw new Error('客户端代码映射的键必须与本次策略允许标的完全一致。')
  }
  return mapping
}

function percent(value: number | undefined): string {
  return typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : 'N/A'
}

export function currentStatus(
  sessionStatus: boolean | undefined,
  preflightStatus: boolean | undefined,
): boolean | undefined {
  return sessionStatus !== undefined ? sessionStatus : preflightStatus
}

export default function ClientSimulationPanel({ slug }: ClientSimulationPanelProps) {
  const [connectionForm] = Form.useForm<ConnectionFields>()
  const [executionForm] = Form.useForm<ExecutionFields>()
  const [messageApi, contextHolder] = message.useMessage()
  const [connectionReady, setConnectionReady] = useState(false)
  const [accounts, setAccounts] = useState<FtAccount[]>([])
  const [preflight, setPreflight] = useState<ClientSimulationPreflight | null>(null)
  const [session, setSession] = useState<ClientSimulationSession | null>(null)
  const [loading, setLoading] = useState<string | null>(null)

  const executionMode = Form.useWatch('executionMode', executionForm) || 'observe'

  useEffect(() => {
    if (!session || !['running', 'paused', 'stopping', 'needs_attention'].includes(session.status)) {
      return
    }
    const timer = window.setInterval(async () => {
      try {
        setSession(await api.clientSimulationSession(slug, session.session_id))
      } catch {
        // Keep the last safe snapshot visible; explicit operations surface errors.
      }
    }, 3000)
    return () => window.clearInterval(timer)
  }, [session?.session_id, session?.status, slug])

  const connectionPayload = (values: ConnectionFields): FtConnectionInput => {
    const allowedSymbols = parseAllowedSymbols(values.allowedSymbols)
    return {
      base_url: values.baseUrl,
      ft_account: values.ftAccount,
      password: values.password,
      password_transform: values.passwordTransform,
      confirmed_client_version: values.clientVersion,
      allowed_simulation_accounts: values.simulationAccounts
        .split(',')
        .map(item => item.trim())
        .filter(Boolean),
      allowed_symbols: allowedSymbols,
      symbol_mapping: parseSymbolMapping(values.symbolMappingJson, allowedSymbols),
      allowed_algorithms: ['TWAP'],
      external_id_max_length: values.externalIdMaxLength,
      external_id_scope_confirmed: values.externalIdConfirmed,
    }
  }

  const simulationPayload = (values: ExecutionFields): ClientSimulationRequest => ({
    trade_account: values.tradeAccount,
    execution_mode: values.executionMode,
    acknowledge_simulation: values.acknowledgeSimulation,
    execution_route: 'algorithm_parent',
    execution_window_start: values.startTime,
    execution_window_end: values.endTime,
    algorithm: {
      strategy_type: values.algorithm,
      params: parseAlgorithmParams(values.algorithmParams),
      reach_limit_continue: false,
      over_time_continue: false,
    },
    risk: {
      max_order_pct: values.maxOrderPct,
      max_symbol_position_pct: values.maxSymbolPct,
      max_total_position_pct: values.maxTotalPct,
    },
  })

  const checkConnection = async () => {
    setLoading('connection')
    setConnectionReady(false)
    setPreflight(null)
    try {
      const values = await connectionForm.validateFields()
      const result = await api.checkFtClientConnection(connectionPayload(values))
      const nextAccounts = await api.ftClientAccounts()
      setAccounts(nextAccounts)
      setConnectionReady(result.ready)
      if (nextAccounts.length) {
        executionForm.setFieldValue('tradeAccount', nextAccounts[0].trade_account)
      }
      result.ready
        ? messageApi.success('非凸客户端连接成功，凭证仅保存在本机进程内存中。')
        : messageApi.error('客户端连接检查未通过。')
    } catch (error) {
      messageApi.error(displayError(error))
    } finally {
      setLoading(null)
    }
  }

  const runPreflight = async () => {
    setLoading('preflight')
    try {
      const values = await executionForm.validateFields()
      const result = await api.preflightClientSimulation(slug, simulationPayload(values))
      setPreflight(result)
      result.ready ? messageApi.success('预执行检查通过。') : messageApi.warning('存在硬阻断项。')
    } catch (error) {
      messageApi.error(displayError(error))
    } finally {
      setLoading(null)
    }
  }

  const startSession = async () => {
    setLoading('start')
    try {
      const values = await executionForm.validateFields()
      const created = await api.createClientSimulationSession(slug, simulationPayload(values))
      setSession(created)
      messageApi.success('非凸客户端模拟盘会话已启动。')
    } catch (error) {
      messageApi.error(displayError(error))
    } finally {
      setLoading(null)
    }
  }

  const updateSession = async (action: 'pause' | 'resume' | 'stop') => {
    if (!session) return
    setLoading(action)
    try {
      const next = action === 'pause'
        ? await api.pauseClientSimulationSession(slug, session.session_id)
        : action === 'resume'
          ? await api.resumeClientSimulationSession(slug, session.session_id)
          : await api.stopClientSimulationSession(slug, session.session_id)
      setSession(next)
    } catch (error) {
      messageApi.error(displayError(error))
    } finally {
      setLoading(null)
    }
  }

  const approveIntent = async (intentId: string) => {
    if (!session) return
    setLoading(`approve:${intentId}`)
    try {
      setSession(await api.approveClientSimulationIntent(slug, session.session_id, intentId))
    } catch (error) {
      messageApi.error(displayError(error))
    } finally {
      setLoading(null)
    }
  }

  const monitoring = session?.monitoring?.trade_accounts.find(
    item => item.trade_account === session.trade_account,
  ) || preflight?.monitoring?.trade_accounts[0]
  const accountLoggedIn = currentStatus(
    session?.account_login_status,
    preflight?.health?.login_status,
  )
  const orderEngineReady = currentStatus(
    session?.order_engine_status,
    preflight?.health?.order_engine_status,
  )

  return (
    <div className="client-simulation-panel">
      {contextHolder}
      <Alert
        type="warning"
        showIcon
        title="非凸客户端模拟账户"
        description="该模式会向本机非凸客户端提交真实模拟委托，不是本地历史回放。仅在沪深市场形成共同的已完成 10 分钟 K 线后评估策略，账户与委托状态约每 3 秒同步。"
        className="workbench-alert"
      />

      <Card className="workbench-card" title="1. 客户端连接（客户填写）">
        <Form
          form={connectionForm}
          layout="vertical"
          initialValues={{
            baseUrl: 'http://127.0.0.1:11356',
            passwordTransform: 'plain',
            clientVersion: '3.11.4',
            allowedSymbols: '',
            symbolMappingJson: '{}',
            externalIdMaxLength: 64,
            externalIdConfirmed: false,
          }}
        >
          <Row gutter={16}>
            <Col xs={24} md={12}>
              <Form.Item label="客户端地址" name="baseUrl" rules={[{ required: true }]}>
                <Input placeholder="http://127.0.0.1:11356" />
              </Form.Item>
            </Col>
            <Col xs={24} md={12}>
              <Form.Item label="客户端版本" name="clientVersion" rules={[{ required: true }]}>
                <Input placeholder="最低 3.11.4" />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item label="非凸账号" name="ftAccount" rules={[{ required: true }]}>
                <Input autoComplete="username" />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item label="密码" name="password" rules={[{ required: true }]}>
                <Input.Password autoComplete="current-password" />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item label="密码处理" name="passwordTransform" rules={[{ required: true }]}>
                <Select options={[
                  { value: 'plain', label: '原文（客户端本地处理）' },
                  { value: 'md5_32_lower', label: 'MD5 32 位小写' },
                ]} />
              </Form.Item>
            </Col>
            <Col xs={24} md={12}>
              <Form.Item
                label="模拟交易账户白名单"
                name="simulationAccounts"
                rules={[{ required: true }]}
                extra="多个交易账户 ID 使用英文逗号分隔；只填写已确认的模拟账户。"
              >
                <Input placeholder="SIM_ACCOUNT_ID" />
              </Form.Item>
            </Col>
            <Col xs={24} md={12}>
              <Form.Item
                label="本次策略允许标的"
                name="allowedSymbols"
                rules={[
                  { required: true },
                  {
                    validator: (_, value) => {
                      try {
                        parseAllowedSymbols(value || '')
                        return Promise.resolve()
                      } catch (error) {
                        return Promise.reject(error)
                      }
                    },
                  },
                ]}
                extra="填写该策略可能提交订单的全部沪深 A 股或 ETF；支持逗号或换行分隔。指数只能作行情基准。"
              >
                <Input.TextArea rows={3} placeholder={'例如：600519.SH\n510500.SH'} />
              </Form.Item>
            </Col>
            <Col xs={24} md={12}>
              <Form.Item
                label="客户端代码映射"
                name="symbolMappingJson"
                dependencies={['allowedSymbols']}
                rules={[
                  { required: true },
                  {
                    validator: (_, value) => {
                      try {
                        const allowed = parseAllowedSymbols(
                          connectionForm.getFieldValue('allowedSymbols') || '',
                        )
                        parseSymbolMapping(value || '', allowed)
                        return Promise.resolve()
                      } catch (error) {
                        return Promise.reject(error)
                      }
                    },
                  },
                ]}
                extra="JSON 的键须与允许标的完全一致；值填写非凸客户端实际接收的证券代码。"
              >
                <Input.TextArea
                  rows={3}
                  placeholder={'{"600519.SH":"600519.SH","510500.SH":"510500.SH"}'}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item
                label="external_id 最大长度"
                name="externalIdMaxLength"
                rules={[{ required: true }]}
                extra="需填写真实客户端联调确认的最大长度；当前平台幂等 ID 需要至少 64 字符。"
              >
                <InputNumber min={1} precision={0} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item
            name="externalIdConfirmed"
            valuePropName="checked"
            rules={[{
              validator: (_, checked) => checked
                ? Promise.resolve()
                : Promise.reject(new Error('请先完成 external_id 联调确认。')),
            }]}
          >
            <Checkbox>已确认客户端支持 64 字符 external_id，并可在母单查询中完整返回</Checkbox>
          </Form.Item>
          <Space>
            <Button type="primary" onClick={checkConnection} loading={loading === 'connection'}>
              检查连接并读取账户
            </Button>
            <Tag color={connectionReady ? 'success' : 'default'}>
              {connectionReady ? '连接已验证' : '尚未验证'}
            </Tag>
          </Space>
          <Descriptions size="small" column={3} className="mt-16">
            <Descriptions.Item label="账户登录状态">
              <Tag color={accountLoggedIn ? 'success' : 'default'}>
                {accountLoggedIn === undefined ? '待预检' : accountLoggedIn ? '已登录' : '未登录'}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="交易引擎状态">
              <Tag color={orderEngineReady ? 'success' : 'default'}>
                {orderEngineReady === undefined ? '待预检' : orderEngineReady ? '可用' : '不可用'}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="最近 10 分钟 K 线">
              {session?.last_evaluated_bar_at || '会话启动后显示'}
            </Descriptions.Item>
          </Descriptions>
          <div className="credential-memory-note">
            密码仅在本机内存中使用，不写入配置、日志或会话文件
          </div>
        </Form>
      </Card>

      <Card className="workbench-card" title="2. 执行与风控（客户填写）">
        <Form
          form={executionForm}
          layout="vertical"
          initialValues={{
            executionMode: 'observe',
            algorithm: 'TWAP',
            algorithmParams: '{}',
            startTime: '093500',
            endTime: '145000',
            maxOrderPct: 5,
            maxSymbolPct: 20,
            maxTotalPct: 80,
            acknowledgeSimulation: false,
          }}
        >
          <Row gutter={16}>
            <Col xs={24} md={8}>
              <Form.Item label="模拟交易账户" name="tradeAccount" rules={[{ required: true }]}>
                <Select
                  disabled={!connectionReady}
                  placeholder="先检查客户端连接"
                  options={accounts.map(account => ({
                    value: account.trade_account,
                    label: `${account.trade_account} · ${account.nickname || account.broker_name}`,
                  }))}
                />
              </Form.Item>
            </Col>
            <Col xs={24} md={16}>
              <Form.Item label="执行模式" name="executionMode">
                <Radio.Group aria-label="执行模式">
                  <Radio value="observe">观察（不下单）</Radio>
                  <Radio value="manual">人工确认</Radio>
                  <Radio value="auto">自动模拟</Radio>
                </Radio.Group>
              </Form.Item>
            </Col>
            <Col xs={24} md={8}>
              <Form.Item label="算法类型" name="algorithm" rules={[{ required: true }]}>
                <Select options={[{ value: 'TWAP', label: 'TWAP' }]} />
              </Form.Item>
            </Col>
            <Col xs={24} md={16}>
              <Form.Item
                label="算法参数"
                name="algorithmParams"
                extra={'结构化 JSON，例如 {"delay_end_time":10}；不会把 signal_price 自动映射为 limit_price。'}
              >
                <Input.TextArea rows={2} />
              </Form.Item>
            </Col>
            <Col xs={12} md={6}>
              <Form.Item label="开始时间" name="startTime" rules={[{ required: true, pattern: /^\d{6}$/ }]}>
                <Input placeholder="093500" />
              </Form.Item>
            </Col>
            <Col xs={12} md={6}>
              <Form.Item label="结束时间" name="endTime" rules={[{ required: true, pattern: /^\d{6}$/ }]}>
                <Input placeholder="145000" />
              </Form.Item>
            </Col>
            <Col xs={8} md={4}>
              <Form.Item label="单笔资产上限" name="maxOrderPct" rules={[{ required: true }]}>
                <InputNumber min={0.1} max={5} suffix="%" style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col xs={8} md={4}>
              <Form.Item label="单标的仓位上限" name="maxSymbolPct" rules={[{ required: true }]}>
                <InputNumber min={1} max={20} suffix="%" style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col xs={8} md={4}>
              <Form.Item label="总仓位上限" name="maxTotalPct" rules={[{ required: true }]}>
                <InputNumber min={1} max={80} suffix="%" style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item
            name="acknowledgeSimulation"
            valuePropName="checked"
            rules={executionMode === 'observe' ? [] : [{
              validator: (_, checked) => checked
                ? Promise.resolve()
                : Promise.reject(new Error('人工或自动模式必须确认模拟账户。')),
            }]}
          >
            <Checkbox>我确认所选交易账户是模拟账户，并接受当前标的与风险上限</Checkbox>
          </Form.Item>
          <Space wrap>
            <Button
              onClick={runPreflight}
              disabled={!connectionReady}
              loading={loading === 'preflight'}
            >
              运行预执行检查
            </Button>
            <Button
              type="primary"
              onClick={startSession}
              disabled={!connectionReady || !preflight?.ready}
              loading={loading === 'start'}
            >
              启动非凸模拟盘
            </Button>
          </Space>
        </Form>
      </Card>

      {preflight && (
        <Card className="workbench-card" title="预执行检查">
          <Space direction="vertical" style={{ width: '100%' }}>
            {preflight.checks.map(check => (
              <Alert
                key={`${check.code}-${check.passed}`}
                type={check.passed ? 'success' : 'error'}
                showIcon
                title={check.code}
                description={check.message}
              />
            ))}
          </Space>
        </Card>
      )}

      {session && (
        <>
          <Card className="workbench-card">
            <Space style={{ width: '100%', justifyContent: 'space-between' }}>
              <div>
                <Title level={5} style={{ margin: 0 }}>会话 {session.session_id}</Title>
                <Text type="secondary">
                  客户端 {session.client_version} · 账户 {session.trade_account} · {session.execution_mode}
                </Text>
              </div>
              <Space>
                <Tag color={session.status === 'running' ? 'processing' : 'default'}>{session.status}</Tag>
                {session.status === 'running' && (
                  <Button onClick={() => updateSession('pause')} loading={loading === 'pause'}>暂停</Button>
                )}
                {session.status === 'paused' && (
                  <Button onClick={() => updateSession('resume')} loading={loading === 'resume'}>恢复</Button>
                )}
                {!['stopped', 'completed', 'failed'].includes(session.status) && (
                  <Button danger onClick={() => updateSession('stop')} loading={loading === 'stop'}>停止并撤单</Button>
                )}
              </Space>
            </Space>
            <Row gutter={[16, 16]} className="mt-16">
              <Col span={6}><Statistic title="可用资金" value={session.funds?.available || 0} precision={2} /></Col>
              <Col span={6}><Statistic title="风险资产基数" value={session.funds?.risk_equity || 0} precision={2} /></Col>
              <Col span={6}><Statistic title="总完成率" value={percent(monitoring?.total_rate)} /></Col>
              <Col span={6}><Statistic title="执行暴露" value={percent(monitoring?.exposure)} /></Col>
              <Col span={6}><Statistic title="买入完成率" value={percent(monitoring?.buy_rate)} /></Col>
              <Col span={6}><Statistic title="卖出完成率" value={percent(monitoring?.sale_rate)} /></Col>
              <Col span={6}><Statistic title="撤单率" value={percent(monitoring?.cancel_rate)} /></Col>
              <Col span={6}><Statistic title="错废单率" value={percent(monitoring?.error_rate)} /></Col>
            </Row>
          </Card>

          <Card className="workbench-card" title="订单意图">
            <Table
              size="small"
              rowKey="intent_id"
              pagination={false}
              dataSource={session.intents}
              columns={[
                { title: '标的', dataIndex: 'symbol' },
                { title: '方向', dataIndex: 'side' },
                { title: '数量', dataIndex: 'quantity' },
                { title: '信号价', dataIndex: 'signal_price' },
                { title: '状态', dataIndex: 'status', render: value => <Tag>{value}</Tag> },
                { title: '原因', dataIndex: 'reason' },
                {
                  title: '操作',
                  render: (_, intent) => session.execution_mode === 'manual' && intent.status === 'validated'
                    ? (
                      <Button
                        size="small"
                        type="primary"
                        loading={loading === `approve:${intent.intent_id}`}
                        onClick={() => approveIntent(intent.intent_id)}
                      >
                        确认提交
                      </Button>
                    )
                    : null,
                },
              ]}
            />
          </Card>

          <Card className="workbench-card" title="算法母单">
            <Table
              size="small"
              rowKey="parent_order_id"
              pagination={false}
              dataSource={session.orders}
              columns={[
                { title: '母单 ID', dataIndex: 'parent_order_id' },
                { title: 'external_id', dataIndex: 'external_id', ellipsis: true },
                { title: '标的', dataIndex: 'stock_code' },
                { title: '委托量', dataIndex: 'order_volume' },
                { title: '成交量', dataIndex: 'trade_volume' },
                { title: '状态', dataIndex: 'normalized_status', render: value => <Tag>{value}</Tag> },
              ]}
            />
          </Card>

          <Card className="workbench-card" title="账户持仓">
            <Descriptions bordered size="small" column={2}>
              {session.positions.map(position => (
                <Descriptions.Item key={position.stock_code} label={position.stock_code}>
                  总持仓 {position.total_volume} · 可用 {position.available_volume} · 在途 {position.in_transit_volume}
                </Descriptions.Item>
              ))}
            </Descriptions>
          </Card>
        </>
      )}
    </div>
  )
}
