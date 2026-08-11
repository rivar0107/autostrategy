import React, { useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Empty,
  Form,
  Input,
  Layout,
  Modal,
  Popconfirm,
  Progress,
  Select,
  Space,
  Spin,
  Steps,
  Table,
  Tag,
  Typography,
  message,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import {
  ApiError,
  api,
} from './api/client'
import StrategyWorkbench from './StrategyWorkbench'
import type {
  ConfigResponse,
  DesignJobResponse,
  LlmConfigUpdate,
  Strategy,
} from './types'
import './App.css'

const { Header, Content } = Layout
const { Title, Text } = Typography

const STATUS_COLOR: Record<string, string> = {
  draft: 'default',
  designed: 'processing',
  coded: 'warning',
  backtested: 'success',
  paper_running: 'cyan',
  optimized: 'purple',
  active: 'green',
  archived: 'default',
}

const ARTIFACT_LABELS: Record<string, string> = {
  design: '设计文档',
  strategy_code: '策略代码',
  config: '配置文件',
  readme: '说明文档',
  requirements: '依赖清单',
  fetch_data: '数据获取脚本',
  backtest_result: '回测结果',
  paper_run_result: '模拟运行结果',
  paper_run_events: '模拟运行事件',
  paper_run_log: '模拟运行日志',
}

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return `${error.code}: ${error.message}`
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'Unknown error'
}

function isJobGone(error: unknown): boolean {
  return error instanceof ApiError && (error.code === 'job_not_found' || error.status === 404)
}

function formatMetric(value: unknown, suffix = ''): string {
  if (typeof value === 'number') {
    if (suffix === '%') {
      return `${value.toFixed(2)}%`
    }
    return `${value.toFixed(2)}${suffix}`
  }
  if (value === null || value === undefined) {
    return 'N/A'
  }
  return String(value)
}

function setupHint(config: ConfigResponse | null): string {
  return config?.llm_setup_hint || `请在启动 autostrategy 的终端中设置 ${config?.llm_api_key_env || 'AUTOSTRATEGY_LLM_API_KEY'}。`
}

function shellExample(apiKeyEnv: string): string {
  return `export ${apiKeyEnv}="your-api-key"\nautostrategy serve`
}

function llmConfigFromError(error: ApiError, current: ConfigResponse | null): ConfigResponse {
  const details = error.details
  return {
    version: current?.version || '',
    default_market: current?.default_market || 'A股',
    llm_provider: typeof details.provider === 'string' ? details.provider : current?.llm_provider || 'openai',
    llm_model: current?.llm_model || 'gpt-4o-mini',
    llm_base_url: current?.llm_base_url || null,
    llm_api_key_env: typeof details.api_key_env === 'string' ? details.api_key_env : current?.llm_api_key_env || 'AUTOSTRATEGY_LLM_API_KEY',
    llm_ready: false,
    llm_missing_api_key: true,
    llm_setup_hint: typeof details.setup_hint === 'string' ? details.setup_hint : current?.llm_setup_hint || null,
    llm_checked_env_vars: Array.isArray(details.checked_env_vars) ? details.checked_env_vars.filter((value) => typeof value === 'string') : current?.llm_checked_env_vars || [],
  }
}

function getRouteSlug(): string | null {
  const hash = window.location.hash
  const match = hash.match(/^#\/strategy\/(.+)$/)
  return match ? decodeURIComponent(match[1]) : null
}

const DESIGN_STEPS = [
  { title: '提交任务', description: '创建策略工作区' },
  { title: '生成设计', description: '调用 LLM 生成策略设计文档' },
  { title: '保存结果', description: '写入本地文件并更新状态' },
]

function DesignProgress({ job, onCancel }: { job: DesignJobResponse | null; onCancel: () => void }) {
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    if (!job || job.status === 'succeeded' || job.status === 'failed') return
    const timer = setInterval(() => {
      setElapsed((value) => value + 1)
    }, 1000)
    return () => clearInterval(timer)
  }, [job])

  if (!job) return <Spin />

  const currentStep = job.status === 'queued' ? 0 : job.status === 'running' ? 1 : 2
  const percent = job.status === 'succeeded' ? 100 : job.status === 'failed' ? 100 : Math.min(90, 10 + elapsed * 2)

  return (
    <Space direction="vertical" className="full-width" size="large">
      <Steps
        direction="vertical"
        current={currentStep}
        status={job.status === 'failed' ? 'error' : job.status === 'succeeded' ? 'finish' : 'process'}
        items={DESIGN_STEPS.map((step, index) => ({
          title: step.title,
          description: index === currentStep && job.status === 'running' ? `${step.description}…` : step.description,
        }))}
      />
      <Progress percent={percent} status={job.status === 'failed' ? 'exception' : 'active'} />
      <Space className="full-width design-progress-footer">
        <Text type="secondary">已耗时 {elapsed} 秒</Text>
        {job.status !== 'succeeded' && job.status !== 'failed' && (
          <Button onClick={onCancel}>取消等待</Button>
        )}
      </Space>
      {job.status === 'failed' && job.error && <Alert type="error" title={job.error} showIcon />}
    </Space>
  )
}

function App() {
  const [strategies, setStrategies] = useState<Strategy[]>([])
  const [templates, setTemplates] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [createOpen, setCreateOpen] = useState(false)
  const [designJob, setDesignJob] = useState<DesignJobResponse | null>(null)
  const [designProgressOpen, setDesignProgressOpen] = useState(false)
  const designPollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const [config, setConfig] = useState<ConfigResponse | null>(null)

  useEffect(() => {
    return () => {
      if (designPollRef.current) {
        clearInterval(designPollRef.current)
      }
    }
  }, [])
  const [llmSetupOpen, setLlmSetupOpen] = useState(false)
  const [setupAutoShown, setSetupAutoShown] = useState(false)
  const [setupNotice, setSetupNotice] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [messageApi, contextHolder] = message.useMessage()
  const [createForm] = Form.useForm()
  const [llmForm] = Form.useForm<LlmConfigUpdate>()
  const [routeSlug, setRouteSlug] = useState<string | null>(getRouteSlug())

  const loadConfig = async (autoOpen = false) => {
    try {
      const nextConfig = await api.config()
      setConfig(nextConfig)
      if (autoOpen && !nextConfig.llm_ready && !setupAutoShown) {
        setSetupNotice(nextConfig.llm_setup_hint || null)
        setLlmSetupOpen(true)
        setSetupAutoShown(true)
      }
      return nextConfig
    } catch (err) {
      setError(errorMessage(err))
      return null
    }
  }

  const loadInitial = async () => {
    setLoading(true)
    setError(null)
    loadConfig(true)
    try {
      const [strategyList, templateList] = await Promise.all([api.strategies(), api.templates()])
      setStrategies(strategyList)
      setTemplates(templateList)
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadInitial()
  }, [])

  useEffect(() => {
    const onHashChange = () => setRouteSlug(getRouteSlug())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])



  useEffect(() => {
    if (!config) return
    llmForm.setFieldsValue({
      provider: config.llm_provider,
      model: config.llm_model,
      base_url: config.llm_base_url || undefined,
      api_key_env: config.llm_api_key_env,
    })
  }, [config, llmForm])

  const openLlmSetup = () => {
    setSetupNotice(config?.llm_setup_hint || null)
    setLlmSetupOpen(true)
  }

  const handleLlmConfigurationError = (err: unknown): boolean => {
    if (!(err instanceof ApiError) || err.code !== 'llm_configuration_required') {
      return false
    }
    const nextConfig = llmConfigFromError(err, config)
    setConfig(nextConfig)
    setSetupNotice(setupHint(nextConfig))
    setLlmSetupOpen(true)
    return true
  }

  const updateLlmConfig = async (values: LlmConfigUpdate) => {
    setActionLoading('llm-config')
    try {
      const nextConfig = await api.updateLlmConfig({
        provider: values.provider,
        model: values.model,
        base_url: values.base_url || null,
        api_key_env: values.api_key_env,
      })
      setConfig(nextConfig)
      setSetupNotice(nextConfig.llm_setup_hint || null)
      messageApi.success('LLM 设置已保存')
      if (nextConfig.llm_ready) {
        setLlmSetupOpen(false)
      }
    } catch (err) {
      messageApi.error(errorMessage(err))
    } finally {
      setActionLoading(null)
    }
  }


  const openDetail = (strategy: Strategy) => {
    window.open(`#/strategy/${encodeURIComponent(strategy.slug)}`, '_blank')
  }


  const clearDesignPoll = () => {
    if (designPollRef.current) {
      clearInterval(designPollRef.current)
      designPollRef.current = null
    }
  }

  const closeDesignProgress = () => {
    clearDesignPoll()
    setDesignProgressOpen(false)
    setDesignJob(null)
  }

  const startDesignPolling = (jobId: string) => {
    clearDesignPoll()
    designPollRef.current = setInterval(async () => {
      try {
        const job = await api.designJob(jobId)
        setDesignJob(job)
        if (job.status === 'succeeded') {
          clearDesignPoll()
          messageApi.success('策略已创建，设计文档已生成')
          await loadInitial()
          closeDesignProgress()
          setCreateOpen(false)
          createForm.resetFields()
          if (job.strategy) {
            openDetail(job.strategy)
          }
        } else if (job.status === 'failed') {
          clearDesignPoll()
          if (job.error_code === 'llm_not_configured') {
            const details = job.error ? { message: job.error } : {}
            const apiError = new ApiError(job.error || 'LLM 未配置', 'llm_not_configured', details, 400)
            if (!handleLlmConfigurationError(apiError)) {
              messageApi.error(job.error || '创建失败')
            }
          } else {
            messageApi.error(job.error || '创建失败')
          }
          closeDesignProgress()
        }
      } catch (err) {
        if (isJobGone(err)) {
          clearDesignPoll()
          messageApi.error('任务状态丢失，请刷新列表查看结果')
          closeDesignProgress()
        }
      }
    }, 800)
  }

  const createStrategy = async (values: { name: string; prompt?: string; market: string; template?: string }) => {
    const prompt = values.prompt?.trim()
    setActionLoading('create')
    try {
      if (!prompt) {
        const strategy = await api.createStrategy(values)
        messageApi.success('策略已创建')
        setCreateOpen(false)
        createForm.resetFields()
        await loadInitial()
        openDetail(strategy)
        return
      }

      const job = await api.createDesign({ ...values, prompt })
      if (job.status === 'succeeded') {
        messageApi.success('策略已创建，设计文档已生成')
        setCreateOpen(false)
        createForm.resetFields()
        await loadInitial()
        if (job.strategy) {
          openDetail(job.strategy)
        }
        return
      }
      if (job.status === 'failed') {
        throw new ApiError(job.error || '创建失败', job.error_code || 'api_error', {}, 400)
      }
      setDesignJob(job)
      setDesignProgressOpen(true)
      startDesignPolling(job.job_id)
    } catch (err) {
      if (!handleLlmConfigurationError(err)) {
        messageApi.error(errorMessage(err))
      }
    } finally {
      setActionLoading(null)
    }
  }





  const deleteStrategy = async (slug: string) => {
    setActionLoading(`delete:${slug}`)
    try {
      await api.deleteStrategy(slug)
      messageApi.success('策略已删除')
      await loadInitial()
    } catch (err) {
      messageApi.error(errorMessage(err))
    } finally {
      setActionLoading(null)
    }
  }

  const columns: ColumnsType<Strategy> = useMemo(() => [
    { title: '名称', dataIndex: 'name', key: 'name' },
    { title: 'Slug', dataIndex: 'slug', key: 'slug' },
    { title: '市场', dataIndex: 'market', key: 'market' },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => <Tag color={STATUS_COLOR[status] || 'default'}>{status}</Tag>,
    },
    {
      title: '模板',
      dataIndex: 'template',
      key: 'template',
      render: (template?: string | null) => template || '无',
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button type="link" onClick={() => openDetail(record)}>查看详情</Button>
          <Popconfirm title="删除策略？" onConfirm={() => deleteStrategy(record.slug)}>
            <Button danger type="link" loading={actionLoading === `delete:${record.slug}`}>删除</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ], [actionLoading])


  const watchedApiKeyEnv = Form.useWatch('api_key_env', llmForm) || config?.llm_api_key_env || 'AUTOSTRATEGY_LLM_API_KEY'

  if (routeSlug) {
    return <StrategyWorkbench slug={routeSlug} onBack={() => { window.location.hash = '' }} />
  }

  return (
    <Layout className="app-shell">
      {contextHolder}
      <Header className="app-header">
        <Space orientation="vertical" size={0}>
          <Title level={3} className="app-title">autostrategy</Title>
          <Text type="secondary">您的本地策略agent工作台</Text>
        </Space>
        <Space>
          {config && <Tag color={config.llm_ready ? 'green' : 'orange'}>{config.llm_ready ? 'LLM 已就绪' : 'LLM 未配置'}</Tag>}
          <Button onClick={openLlmSetup}>LLM 设置</Button>
          <Button onClick={loadInitial}>刷新</Button>
          <Button type="primary" onClick={() => setCreateOpen(true)}>创建策略</Button>
        </Space>
      </Header>

      <Content className="app-content">
        {error && <Alert type="error" title="加载失败" description={error} showIcon className="mb-16" />}
        <Card title="策略列表" extra={<Tag>{strategies.length} 个策略</Tag>}>
          <Spin spinning={loading}>
            <Table
              rowKey="slug"
              columns={columns}
              dataSource={strategies}
              locale={{ emptyText: <Empty description="暂无策略" /> }}
              pagination={false}
            />
          </Spin>
        </Card>
      </Content>

      <Modal
        title="创建策略"
        open={createOpen}
        onCancel={() => {
          if (!designProgressOpen) {
            setCreateOpen(false)
          }
        }}
        footer={null}
        destroyOnHidden
        mask={{ closable: !designProgressOpen }}
        keyboard={!designProgressOpen}
      >
        <Form layout="vertical" form={createForm} onFinish={createStrategy} initialValues={{ market: 'A股' }}>
          <Form.Item name="name" label="策略名称" rules={[{ required: true, message: '请输入策略名称' }]}>
            <Input placeholder="例如 dual-ma-demo" />
          </Form.Item>
          <Form.Item name="prompt" label="策略想法（可选，填写后自动生成设计文档）">
            <Input.TextArea rows={5} placeholder="帮我做一个 A 股双均线策略" />
          </Form.Item>
          <Form.Item name="market" label="市场" rules={[{ required: true }]}>
            <Select options={['A股', '港股', '美股'].map((value) => ({ value, label: value }))} />
          </Form.Item>
          <Form.Item name="template" label="模板">
            <Select allowClear options={templates.map((value) => ({ value, label: value }))} />
          </Form.Item>
          <Button type="primary" htmlType="submit" loading={actionLoading === 'create'} block>创建</Button>
        </Form>
      </Modal>

      <Modal
        title="正在创建策略"
        open={designProgressOpen}
        footer={null}
        closable={false}
        mask={{ closable: false }}
        keyboard={false}
        width={520}
      >
        <DesignProgress job={designJob} onCancel={closeDesignProgress} />
      </Modal>

      <Modal title="LLM 设置" open={llmSetupOpen} onCancel={() => setLlmSetupOpen(false)} footer={null} destroyOnHidden>
        <Space orientation="vertical" className="full-width" size="middle">
          <Alert
            type={config?.llm_ready ? 'success' : 'warning'}
            title={config?.llm_ready ? 'LLM 已就绪' : '需要配置本地 LLM API key'}
            description={setupNotice || setupHint(config)}
            showIcon
          />
          <Alert
            type="info"
            title="autostrategy 不保存 API key"
            description="这里只保存 provider、model、base_url 和环境变量名。真正的 API key 只从启动服务的本机 shell 环境变量读取。修改环境变量后，通常需要重启 autostrategy serve。"
            showIcon
          />
          <Form layout="vertical" form={llmForm} onFinish={updateLlmConfig}>
            <Form.Item name="provider" label="模型服务商（Provider）" rules={[{ required: true, message: '请输入模型服务商' }]}>
              <Select options={['openai', 'deepseek', 'kimi', 'qwen', 'zai', 'minimax', 'gemini', 'local'].map((value) => ({ value, label: value }))} />
            </Form.Item>
            <Form.Item name="model" label="模型名称（Model）" rules={[{ required: true, message: '请输入模型名称' }]}>
              <Input placeholder="例如 gpt-4o-mini 或 deepseek-chat" />
            </Form.Item>
            <Form.Item name="base_url" label="接口地址（Base URL）">
              <Input placeholder="可选，例如 https://api.openai.com/v1" />
            </Form.Item>
            <Form.Item name="api_key_env" label="API key 环境变量" rules={[{ required: true, message: '请输入环境变量名' }]}>
              <Input placeholder="AUTOSTRATEGY_LLM_API_KEY" />
            </Form.Item>
            <Alert
              type="info"
              title="在启动服务的终端中执行"
              description={<pre className="code-preview">{shellExample(watchedApiKeyEnv)}</pre>}
              showIcon
            />
            <Button type="primary" htmlType="submit" loading={actionLoading === 'llm-config'} block className="mt-16">保存 LLM 设置</Button>
          </Form>
        </Space>
      </Modal>
    </Layout>
  )
}

export default App
