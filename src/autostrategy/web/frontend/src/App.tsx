import React, { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Drawer,
  Empty,
  Form,
  Input,
  Layout,
  Modal,
  Popconfirm,
  Progress,
  Row,
  Select,
  Space,
  Spin,
  Statistic,
  Table,
  Tabs,
  Tag,
  Typography,
  message,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import {
  ApiError,
  api,
} from './api/client'
import type {
  ArtifactContent,
  ArtifactMeta,
  BacktestJobResponse,
  BacktestResponse,
  ConfigResponse,
  LlmConfigUpdate,
  PaperRunResponse,
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

function formatMetric(value: unknown, suffix = ''): string {
  if (typeof value === 'number') {
    return `${value.toFixed(2)}${suffix}`
  }
  if (value === null || value === undefined) {
    return 'N/A'
  }
  return String(value)
}

function backtestJobMessage(job: BacktestJobResponse): string {
  if (job.status === 'queued') return '回测任务已提交，正在排队。'
  if (job.status === 'running') return '回测正在独立子进程中运行。'
  if (job.status === 'succeeded') return `回测完成，评分 ${formatMetric(job.score)}`
  if (job.status === 'timed_out') return job.error || '回测任务超时。'
  if (job.status === 'stopped') return '回测任务已停止。'
  return job.error || '回测任务失败。'
}

function paperRunJobMessage(job: BacktestJobResponse): string {
  if (job.status === 'queued') return '模拟运行任务已提交，正在排队。'
  if (job.status === 'running') return '模拟运行正在独立子进程中 replay。'
  if (job.status === 'succeeded') return '模拟运行完成。'
  if (job.status === 'stopped') return '模拟运行已停止。'
  if (job.status === 'timed_out') return job.error || '模拟运行任务超时。'
  return job.error || '模拟运行任务失败。'
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

function App() {
  const [strategies, setStrategies] = useState<Strategy[]>([])
  const [templates, setTemplates] = useState<string[]>([])
  const [selected, setSelected] = useState<Strategy | null>(null)
  const [artifacts, setArtifacts] = useState<ArtifactMeta[]>([])
  const [artifactContent, setArtifactContent] = useState<Record<string, ArtifactContent>>({})
  const [backtestResult, setBacktestResult] = useState<BacktestResponse | null>(null)
  const [backtestJob, setBacktestJob] = useState<BacktestJobResponse | null>(null)
  const [paperRunResult, setPaperRunResult] = useState<PaperRunResponse | null>(null)
  const [paperRunJob, setPaperRunJob] = useState<BacktestJobResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [createOpen, setCreateOpen] = useState(false)
  const [designOpen, setDesignOpen] = useState(false)
  const [detailOpen, setDetailOpen] = useState(false)
  const [config, setConfig] = useState<ConfigResponse | null>(null)
  const [llmSetupOpen, setLlmSetupOpen] = useState(false)
  const [setupAutoShown, setSetupAutoShown] = useState(false)
  const [setupNotice, setSetupNotice] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [messageApi, contextHolder] = message.useMessage()
  const [createForm] = Form.useForm()
  const [designForm] = Form.useForm()
  const [llmForm] = Form.useForm<LlmConfigUpdate>()

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
    if (!selected || !backtestJob || !['queued', 'running'].includes(backtestJob.status)) return
    const timer = window.setInterval(async () => {
      try {
        const nextJob = await api.backtestJob(selected.slug, backtestJob.job_id)
        setBacktestJob(nextJob)
        if (nextJob.status === 'succeeded') {
          messageApi.success(backtestJobMessage(nextJob))
          const result = await api.backtestResult(selected.slug)
          setBacktestResult(result)
          await loadInitial()
          await refreshSelected(selected.slug)
        }
        if (nextJob.status === 'failed' || nextJob.status === 'timed_out') {
          messageApi.error(backtestJobMessage(nextJob))
        }
      } catch (err) {
        messageApi.error(errorMessage(err))
      }
    }, 1000)
    return () => window.clearInterval(timer)
  }, [backtestJob, selected, messageApi])

  useEffect(() => {
    if (!selected || !paperRunJob || !['queued', 'running'].includes(paperRunJob.status)) return
    const timer = window.setInterval(async () => {
      try {
        const nextJob = await api.paperRunJob(selected.slug, paperRunJob.job_id)
        setPaperRunJob(nextJob)
        if (nextJob.status === 'running') {
          try {
            setPaperRunResult(await api.paperRunResult(selected.slug))
          } catch {
            setPaperRunResult(null)
          }
        }
        if (nextJob.status === 'succeeded' || nextJob.status === 'stopped') {
          messageApi.success(paperRunJobMessage(nextJob))
          const result = await api.paperRunResult(selected.slug)
          setPaperRunResult(result)
          await loadInitial()
          await refreshSelected(selected.slug)
        }
        if (nextJob.status === 'failed' || nextJob.status === 'timed_out') {
          messageApi.error(paperRunJobMessage(nextJob))
          try {
            setPaperRunResult(await api.paperRunResult(selected.slug))
          } catch {
            setPaperRunResult(null)
          }
        }
      } catch (err) {
        messageApi.error(errorMessage(err))
      }
    }, 1000)
    return () => window.clearInterval(timer)
  }, [paperRunJob, selected, messageApi])

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

  const refreshSelected = async (slug: string) => {
    const [detail, artifactList] = await Promise.all([api.strategyDetail(slug), api.artifacts(slug)])
    setSelected(detail.strategy)
    setArtifacts(artifactList.artifacts)
    const firstPreviewable = artifactList.artifacts.find((artifact) => artifact.exists)
    if (firstPreviewable) {
      try {
        const content = await api.artifact(slug, firstPreviewable.artifact_key)
        setArtifactContent({ [firstPreviewable.artifact_key]: content })
      } catch {
        setArtifactContent({})
      }
    } else {
      setArtifactContent({})
    }
    try {
      setBacktestResult(await api.backtestResult(slug))
    } catch {
      setBacktestResult(null)
    }
    try {
      setPaperRunResult(await api.paperRunResult(slug))
    } catch {
      setPaperRunResult(null)
    }
  }

  const openDetail = async (strategy: Strategy) => {
    setDetailOpen(true)
    setSelected(strategy)
    setActionLoading('detail')
    try {
      await refreshSelected(strategy.slug)
    } catch (err) {
      messageApi.error(errorMessage(err))
    } finally {
      setActionLoading(null)
    }
  }

  const loadArtifact = async (key: string) => {
    if (!selected || artifactContent[key]) {
      return
    }
    setActionLoading(`artifact:${key}`)
    try {
      const content = await api.artifact(selected.slug, key)
      setArtifactContent((current) => ({ ...current, [key]: content }))
    } catch (err) {
      messageApi.error(errorMessage(err))
    } finally {
      setActionLoading(null)
    }
  }

  const createStrategy = async (values: { name: string; market: string; template?: string }) => {
    setActionLoading('create')
    try {
      const strategy = await api.createStrategy(values)
      messageApi.success('策略已创建')
      setCreateOpen(false)
      createForm.resetFields()
      await loadInitial()
      await openDetail(strategy)
    } catch (err) {
      messageApi.error(errorMessage(err))
    } finally {
      setActionLoading(null)
    }
  }

  const createDesign = async (values: { name: string; prompt: string; market: string; template?: string }) => {
    setActionLoading('design')
    try {
      const result = await api.createDesign(values)
      messageApi.success('设计文档已生成')
      setDesignOpen(false)
      designForm.resetFields()
      await loadInitial()
      await openDetail(result.strategy)
    } catch (err) {
      if (!handleLlmConfigurationError(err)) {
        messageApi.error(errorMessage(err))
      }
    } finally {
      setActionLoading(null)
    }
  }

  const runCodegen = async (force = false) => {
    if (!selected) return
    setActionLoading('codegen')
    try {
      const result = await api.codegen(selected.slug, force)
      messageApi.success(`已生成 ${result.generated_files.length} 个文件`)
      await loadInitial()
      await refreshSelected(selected.slug)
    } catch (err) {
      if (!handleLlmConfigurationError(err)) {
        messageApi.error(errorMessage(err))
      }
    } finally {
      setActionLoading(null)
    }
  }

  const runBacktest = async () => {
    if (!selected) return
    setActionLoading('backtest')
    try {
      const job = await api.backtest(selected.slug)
      setBacktestJob(job)
      messageApi.success('回测任务已提交')
    } catch (err) {
      messageApi.error(errorMessage(err))
    } finally {
      setActionLoading(null)
    }
  }

  const startPaperRun = async () => {
    if (!selected) return
    setActionLoading('paper-run')
    try {
      const job = await api.startPaperRun(selected.slug)
      setPaperRunJob(job)
      messageApi.success('模拟运行任务已提交')
    } catch (err) {
      messageApi.error(errorMessage(err))
    } finally {
      setActionLoading(null)
    }
  }

  const stopPaperRun = async () => {
    if (!selected || !paperRunJob) return
    setActionLoading('paper-stop')
    try {
      const job = await api.stopPaperRunJob(selected.slug, paperRunJob.job_id)
      setPaperRunJob(job)
      messageApi.success('已请求停止模拟运行')
    } catch (err) {
      messageApi.error(errorMessage(err))
    } finally {
      setActionLoading(null)
    }
  }

  const deleteStrategy = async (slug: string) => {
    setActionLoading(`delete:${slug}`)
    try {
      await api.deleteStrategy(slug)
      messageApi.success('策略已删除')
      if (selected?.slug === slug) {
        setDetailOpen(false)
        setSelected(null)
      }
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
  ], [actionLoading, selected])

  const artifactTabs = artifacts.map((artifact) => ({
    key: artifact.artifact_key,
    label: (
      <Space size={4}>
        <span>{ARTIFACT_LABELS[artifact.artifact_key] || artifact.artifact_key}</span>
        <Tag color={artifact.exists ? 'green' : 'default'}>{artifact.exists ? '已生成' : '缺失'}</Tag>
      </Space>
    ),
    children: <ArtifactPanel artifact={artifact} content={artifactContent[artifact.artifact_key]} loading={actionLoading === `artifact:${artifact.artifact_key}`} />,
  }))

  const watchedApiKeyEnv = Form.useWatch('api_key_env', llmForm) || config?.llm_api_key_env || 'AUTOSTRATEGY_LLM_API_KEY'

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
          <Button onClick={() => setCreateOpen(true)}>创建策略</Button>
          <Button type="primary" onClick={() => setDesignOpen(true)}>自然语言生成设计</Button>
        </Space>
      </Header>

      <Content className="app-content">
        {error && <Alert type="error" message="加载失败" description={error} showIcon className="mb-16" />}
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

      <Modal title="创建空策略" open={createOpen} onCancel={() => setCreateOpen(false)} footer={null} destroyOnHidden>
        <Form layout="vertical" form={createForm} onFinish={createStrategy} initialValues={{ market: 'A股' }}>
          <Form.Item name="name" label="策略名称" rules={[{ required: true, message: '请输入策略名称' }]}>
            <Input placeholder="例如 dual-ma-demo" />
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

      <Modal title="自然语言生成策略设计" open={designOpen} onCancel={() => setDesignOpen(false)} footer={null} destroyOnHidden>
        <Form layout="vertical" form={designForm} onFinish={createDesign} initialValues={{ market: 'A股' }}>
          <Form.Item name="name" label="策略名称" rules={[{ required: true, message: '请输入策略名称' }]}>
            <Input placeholder="例如 phase4b-dual-ma" />
          </Form.Item>
          <Form.Item name="prompt" label="策略想法" rules={[{ required: true, message: '请输入自然语言策略想法' }]}>
            <Input.TextArea rows={5} placeholder="帮我做一个 A 股双均线策略" />
          </Form.Item>
          <Form.Item name="market" label="市场" rules={[{ required: true }]}>
            <Select options={['A股', '港股', '美股'].map((value) => ({ value, label: value }))} />
          </Form.Item>
          <Form.Item name="template" label="参考模板">
            <Select allowClear options={templates.map((value) => ({ value, label: value }))} />
          </Form.Item>
          <Button type="primary" htmlType="submit" loading={actionLoading === 'design'} block>生成设计</Button>
        </Form>
      </Modal>

      <Modal title="LLM 设置" open={llmSetupOpen} onCancel={() => setLlmSetupOpen(false)} footer={null} destroyOnHidden>
        <Space orientation="vertical" className="full-width" size="middle">
          <Alert
            type={config?.llm_ready ? 'success' : 'warning'}
            message={config?.llm_ready ? 'LLM 已就绪' : '需要配置本地 LLM API key'}
            description={setupNotice || setupHint(config)}
            showIcon
          />
          <Alert
            type="info"
            message="autostrategy 不保存 API key"
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
              message="在启动服务的终端中执行"
              description={<pre className="code-preview">{shellExample(watchedApiKeyEnv)}</pre>}
              showIcon
            />
            <Button type="primary" htmlType="submit" loading={actionLoading === 'llm-config'} block className="mt-16">保存 LLM 设置</Button>
          </Form>
        </Space>
      </Modal>

      <Drawer title="策略工作台" size="large" open={detailOpen} onClose={() => setDetailOpen(false)}>
        {!selected ? <Spin /> : (
          <Space orientation="vertical" size="large" className="full-width">
            <Card>
              <Descriptions title={selected.name} bordered column={2}>
                <Descriptions.Item label="Slug">{selected.slug}</Descriptions.Item>
                <Descriptions.Item label="市场">{selected.market}</Descriptions.Item>
                <Descriptions.Item label="状态"><Tag color={STATUS_COLOR[selected.status]}>{selected.status}</Tag></Descriptions.Item>
                <Descriptions.Item label="模板">{selected.template || '无'}</Descriptions.Item>
              </Descriptions>
              <Space className="mt-16">
                <Button onClick={() => refreshSelected(selected.slug)} loading={actionLoading === 'detail'}>刷新详情</Button>
                <Button onClick={() => runCodegen(false)} loading={actionLoading === 'codegen'}>生成代码</Button>
                <Button type="primary" onClick={runBacktest} loading={actionLoading === 'backtest'}>运行回测</Button>
                <Button onClick={startPaperRun} loading={actionLoading === 'paper-run'}>启动模拟运行</Button>
                {paperRunJob && ['queued', 'running'].includes(paperRunJob.status) && (
                  <Button danger onClick={stopPaperRun} loading={actionLoading === 'paper-stop'}>停止模拟运行</Button>
                )}
              </Space>
            </Card>
            <BacktestJobPanel job={backtestJob} />
            <PaperRunJobPanel job={paperRunJob} />
            <BacktestSummary result={backtestResult} />
            <PaperRunSummary result={paperRunResult} />
            <Card title="产物文件">
              <Tabs items={artifactTabs} onChange={loadArtifact} />
            </Card>
          </Space>
        )}
      </Drawer>
    </Layout>
  )
}

function ArtifactPanel({ artifact, content, loading }: { artifact: ArtifactMeta; content?: ArtifactContent; loading: boolean }) {
  if (!artifact.exists) {
    return <Empty description={`${artifact.relative_path} 不存在`} />
  }
  if (loading) {
    return <Spin />
  }
  if (!content) {
    return <Alert type="info" message="点击 Tab 后加载预览内容" showIcon />
  }
  return (
    <Space orientation="vertical" className="full-width">
      <Text type="secondary">{content.relative_path} · {content.size} bytes</Text>
      <pre className="code-preview">{content.content}</pre>
    </Space>
  )
}

function BacktestJobPanel({ job }: { job: BacktestJobResponse | null }) {
  if (!job) return null
  const isActive = job.status === 'queued' || job.status === 'running'
  const alertType = job.status === 'succeeded' ? 'success' : job.status === 'failed' || job.status === 'timed_out' ? 'error' : 'info'
  return (
    <Alert
      type={alertType}
      message={`回测任务：${job.status}`}
      description={backtestJobMessage(job)}
      showIcon
      icon={isActive ? <Spin size="small" /> : undefined}
    />
  )
}

function PaperRunJobPanel({ job }: { job: BacktestJobResponse | null }) {
  if (!job) return null
  const isActive = job.status === 'queued' || job.status === 'running'
  const alertType = job.status === 'succeeded' || job.status === 'stopped' ? 'success' : job.status === 'failed' || job.status === 'timed_out' ? 'error' : 'info'
  return (
    <Alert
      type={alertType}
      message={`模拟运行任务：${job.status}`}
      description={paperRunJobMessage(job)}
      showIcon
      icon={isActive ? <Spin size="small" /> : undefined}
    />
  )
}

function BacktestSummary({ result }: { result: BacktestResponse | null }) {
  if (!result) {
    return <Alert type="info" message="暂无回测结果" description="生成代码并运行回测后，这里会展示分数与关键指标。" showIcon />
  }
  const backtest = result.result.backtest || {}
  return (
    <Card title="回测摘要">
      <Row gutter={[16, 16]}>
        <Col span={6}><Statistic title="评分" value={result.score} /></Col>
        <Col span={6}><Statistic title="年化收益" value={formatMetric(backtest.annual_return, '%')} /></Col>
        <Col span={6}><Statistic title="最大回撤" value={formatMetric(backtest.max_drawdown, '%')} /></Col>
        <Col span={6}><Statistic title="夏普比率" value={formatMetric(backtest.sharpe)} /></Col>
      </Row>
      <Progress percent={Math.min(Math.max(result.score, 0), 100)} className="mt-16" />
    </Card>
  )
}

function PaperRunSummary({ result }: { result: PaperRunResponse | null }) {
  if (!result) {
    return <Alert type="info" message="暂无模拟运行结果" description="启动模拟运行后，这里会展示 replay 进度、最新决策和虚拟表现。" showIcon />
  }
  const payload = result.result || {}
  const summary = payload.summary || {}
  const replay = payload.replay || {}
  const latestDecision = payload.latest_decision
  return (
    <Card title={<Space><span>模拟运行摘要</span><Tag color="cyan">Paper Run</Tag><Tag>{payload.run_status || 'unknown'}</Tag></Space>}>
      <Row gutter={[16, 16]}>
        <Col span={6}><Statistic title="模拟收益" value={formatMetric(summary.paper_return, '%')} /></Col>
        <Col span={6}><Statistic title="最大回撤" value={formatMetric(summary.paper_max_drawdown, '%')} /></Col>
        <Col span={6}><Statistic title="交易次数" value={formatMetric(summary.trade_count)} /></Col>
        <Col span={6}><Statistic title="最终资产" value={formatMetric(summary.final_value)} /></Col>
      </Row>
      <Progress percent={Math.round((replay.progress || 0) * 100)} className="mt-16" />
      <Descriptions bordered column={2} className="mt-16" size="small">
        <Descriptions.Item label="当前时间">{replay.current_at || 'N/A'}</Descriptions.Item>
        <Descriptions.Item label="已处理 bars">{formatMetric(replay.bars_processed)}</Descriptions.Item>
        <Descriptions.Item label="最近动作">{latestDecision?.action || 'N/A'}</Descriptions.Item>
        <Descriptions.Item label="原因">{latestDecision?.reason || 'N/A'}</Descriptions.Item>
      </Descriptions>
    </Card>
  )
}

export default App
