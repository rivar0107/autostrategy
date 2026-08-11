import React, { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Descriptions,
  Empty,
  Layout,
  message,
  Progress,
  Row,
  Segmented,
  Space,
  Spin,
  Statistic,
  Table,
  Collapse,
  Steps,
  Tag,
  Typography,
} from 'antd'
import {
  ArrowLeftOutlined,
  ArrowRightOutlined,
  FolderOpenOutlined,
  CodeOutlined,
  FileTextOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  StopOutlined,
} from '@ant-design/icons'
import { ApiError, api } from './api/client'
import type {
  ArtifactContent,
  ArtifactMeta,
  BacktestJobResponse,
  BacktestResponse,
  PaperRunResponse,
  ResearchQuality,
  Strategy,
} from './types'
import ResearchPanel from './ResearchPanel'
import ClientSimulationPanel from './ClientSimulationPanel'
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

const STATUS_LABEL: Record<string, string> = {
  draft: '草稿',
  designed: '已设计',
  coded: '已生成代码',
  backtested: '已回测',
  paper_running: '模拟运行中',
  optimized: '已优化',
  active: '实盘运行',
  archived: '已归档',
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

const ARTIFACT_ICONS: Record<string, string> = {}

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return `${error.code}: ${error.message}`
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'Unknown error'
}

function codegenErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.code === 'llm_configuration_required') {
      return '生成策略代码失败：LLM 尚未配置，请先完成 LLM 设置后重试。'
    }
    if (error.code === 'validation_error') {
      return '生成策略代码失败：策略内容校验未通过，请检查设计文档后重试。'
    }
    return `生成策略代码失败：${error.message || '服务暂时不可用，请稍后重试。'}`
  }
  if (error instanceof Error && error.message) {
    return `生成策略代码失败：${error.message}`
  }
  return '生成策略代码失败：发生未知错误，请稍后重试。'
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

function ResearchQualityAlert({ quality }: { quality?: ResearchQuality }) {
  if (!quality?.warnings?.length) return null
  return (
    <Alert
      type="warning"
      showIcon
      title="研究质量提示"
      description={(
        <ul style={{ margin: 0, paddingLeft: 20 }}>
          {quality.warnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      )}
      className="workbench-alert mt-16"
    />
  )
}

function formatPercentValue(value: unknown): string {
  if (typeof value === 'number') {
    return `${value.toFixed(2)}%`
  }
  return 'N/A'
}

interface EquityPoint {
  date: string
  equity: number
}

function EquityCurveChart({ points }: { points: EquityPoint[] }) {
  const [hover, setHover] = useState<number | null>(null)
  if (!points.length) return null
  const width = 720
  const height = 260
  const pad = { top: 16, right: 20, bottom: 28, left: 76 }
  const values = points.map((point) => point.equity)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min || 1
  const xFor = (index: number) => pad.left + (index / Math.max(points.length - 1, 1)) * (width - pad.left - pad.right)
  const yFor = (value: number) => pad.top + (1 - (value - min) / span) * (height - pad.top - pad.bottom)
  const path = points.map((point, index) => `${index === 0 ? 'M' : 'L'}${xFor(index).toFixed(1)},${yFor(point.equity).toFixed(1)}`).join(' ')
  const areaPath = `${path} L${xFor(points.length - 1).toFixed(1)},${(height - pad.bottom).toFixed(1)} L${pad.left},${(height - pad.bottom).toFixed(1)} Z`
  const ticks = Array.from({ length: 5 }, (_, i) => min + (span * i) / 4)
  const hovered = hover !== null ? points[hover] : null
  const up = points[points.length - 1].equity >= points[0].equity
  const lineColor = up ? '#3f8600' : '#cf1322'
  return (
    <div className="equity-chart">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(event) => {
          const rect = (event.currentTarget as SVGSVGElement).getBoundingClientRect()
          const x = ((event.clientX - rect.left) / rect.width) * width
          const ratio = (x - pad.left) / (width - pad.left - pad.right)
          const index = Math.round(ratio * (points.length - 1))
          setHover(Math.max(0, Math.min(points.length - 1, index)))
        }}
      >
        {ticks.map((tick) => (
          <g key={tick}>
            <line x1={pad.left} x2={width - pad.right} y1={yFor(tick)} y2={yFor(tick)} stroke="#f0f0f0" />
            <text x={pad.left - 8} y={yFor(tick) + 4} textAnchor="end" fontSize="11" fill="#8c8c8c">
              {(tick / 10000).toFixed(1)}万
            </text>
          </g>
        ))}
        <path d={areaPath} fill={lineColor} opacity={0.08} />
        <path d={path} fill="none" stroke={lineColor} strokeWidth={2} />
        {hovered && hover !== null && (
          <g>
            <line x1={xFor(hover)} x2={xFor(hover)} y1={pad.top} y2={height - pad.bottom} stroke="#999" strokeDasharray="4 3" />
            <circle cx={xFor(hover)} cy={yFor(hovered.equity)} r={4} fill={lineColor} />
          </g>
        )}
        <text x={pad.left} y={height - 8} fontSize="11" fill="#8c8c8c">{points[0].date}</text>
        <text x={width - pad.right} y={height - 8} textAnchor="end" fontSize="11" fill="#8c8c8c">{points[points.length - 1].date}</text>
      </svg>
      <div className="equity-chart-tooltip">
        {hovered ? `${hovered.date}　资产 ${hovered.equity.toLocaleString('zh-CN', { maximumFractionDigits: 0 })} 元` : '移动鼠标查看每日资产'}
      </div>
    </div>
  )
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

function getWorkflowStep(status: string): number {
  const steps: Record<string, number> = {
    draft: 0,
    designed: 1,
    coded: 2,
    backtested: 3,
    paper_running: 4,
    optimized: 5,
    active: 5,
    archived: 5,
  }
  return steps[status] ?? 0
}

interface StrategyWorkbenchProps {
  slug: string
  onBack: () => void
}

interface FileNode {
  key: string
  name: string
  exists: boolean
  artifactKey: string
}

interface FolderNode {
  name: string
  children: (FileNode | FolderNode)[]
}

function isFileNode(node: FileNode | FolderNode): node is FileNode {
  return 'key' in node
}

export default function StrategyWorkbench({ slug, onBack }: StrategyWorkbenchProps) {
  const [strategy, setStrategy] = useState<Strategy | null>(null)
  const [artifacts, setArtifacts] = useState<ArtifactMeta[]>([])
  const [artifactContent, setArtifactContent] = useState<Record<string, ArtifactContent>>({})
  const [openFileTabs, setOpenFileTabs] = useState<string[]>([])
  const [activeTab, setActiveTab] = useState('overview')
  const [activeFile, setActiveFile] = useState<string | null>(null)
  const [backtestResult, setBacktestResult] = useState<BacktestResponse | null>(null)
  const [backtestJob, setBacktestJob] = useState<BacktestJobResponse | null>(null)
  const [paperRunResult, setPaperRunResult] = useState<PaperRunResponse | null>(null)
  const [paperRunJob, setPaperRunJob] = useState<BacktestJobResponse | null>(null)
  const [simulationMode, setSimulationMode] = useState<'local' | 'ft-client'>('local')
  const [loading, setLoading] = useState(false)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [messageApi, contextHolder] = message.useMessage()

  const refreshStrategy = async () => {
    setLoading(true)
    try {
      const [detail, artifactList] = await Promise.all([api.strategyDetail(slug), api.artifacts(slug)])
      setStrategy(detail.strategy)
      setArtifacts(artifactList.artifacts)
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
    } catch (err) {
      messageApi.error(errorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refreshStrategy()
  }, [slug])

  useEffect(() => {
    if (!strategy || !backtestJob || !['queued', 'running'].includes(backtestJob.status)) return
    const timer = window.setInterval(async () => {
      try {
        const nextJob = await api.backtestJob(strategy.slug, backtestJob.job_id)
        setBacktestJob(nextJob)
        if (nextJob.status === 'succeeded') {
          messageApi.success(backtestJobMessage(nextJob))
          const result = await api.backtestResult(strategy.slug)
          setBacktestResult(result)
          await refreshStrategy()
        }
        if (nextJob.status === 'failed' || nextJob.status === 'timed_out') {
          messageApi.error(backtestJobMessage(nextJob))
        }
      } catch (err) {
        if (isJobGone(err)) {
          setBacktestJob(null)
          return
        }
        messageApi.error(errorMessage(err))
      }
    }, 1000)
    return () => window.clearInterval(timer)
  }, [backtestJob, strategy, messageApi])

  useEffect(() => {
    if (!strategy || !paperRunJob || !['queued', 'running'].includes(paperRunJob.status)) return
    const timer = window.setInterval(async () => {
      try {
        const nextJob = await api.paperRunJob(strategy.slug, paperRunJob.job_id)
        setPaperRunJob(nextJob)
        if (nextJob.status === 'running') {
          try {
            setPaperRunResult(await api.paperRunResult(strategy.slug))
          } catch {
            setPaperRunResult(null)
          }
        }
        if (nextJob.status === 'succeeded' || nextJob.status === 'stopped') {
          messageApi.success(paperRunJobMessage(nextJob))
          const result = await api.paperRunResult(strategy.slug)
          setPaperRunResult(result)
          await refreshStrategy()
        }
        if (nextJob.status === 'failed' || nextJob.status === 'timed_out') {
          messageApi.error(paperRunJobMessage(nextJob))
          try {
            setPaperRunResult(await api.paperRunResult(strategy.slug))
          } catch {
            setPaperRunResult(null)
          }
        }
      } catch (err) {
        if (isJobGone(err)) {
          setPaperRunJob(null)
          return
        }
        messageApi.error(errorMessage(err))
      }
    }, 1000)
    return () => window.clearInterval(timer)
  }, [paperRunJob, strategy, messageApi])

  const loadArtifact = async (key: string) => {
    if (artifactContent[key]) return
    setActionLoading(`artifact:${key}`)
    try {
      const content = await api.artifact(slug, key)
      setArtifactContent((current) => ({ ...current, [key]: content }))
    } catch (err) {
      messageApi.error(errorMessage(err))
    } finally {
      setActionLoading(null)
    }
  }

  const runCodegen = async (force = false) => {
    setActionLoading('codegen')
    try {
      const result = await api.codegen(slug, force)
      setStrategy(result.strategy)
      messageApi.success(`已生成 ${result.generated_files.length} 个文件`)
      await refreshStrategy()
    } catch (err) {
      messageApi.error(codegenErrorMessage(err))
    } finally {
      setActionLoading(null)
    }
  }

  const runBacktest = async () => {
    setActionLoading('backtest')
    try {
      const job = await api.backtest(slug)
      setBacktestJob(job)
      messageApi.success('回测任务已提交')
    } catch (err) {
      messageApi.error(errorMessage(err))
    } finally {
      setActionLoading(null)
    }
  }

  const revealInFinder = async () => {
    setActionLoading('reveal')
    try {
      await api.revealStrategy(slug)
      messageApi.success('已打开本地文件夹')
    } catch (err) {
      messageApi.error(errorMessage(err))
    } finally {
      setActionLoading(null)
    }
  }

  const startPaperRun = async () => {
    setActionLoading('paper-run')
    try {
      const job = await api.startPaperRun(slug)
      setPaperRunJob(job)
      messageApi.success('模拟运行任务已提交')
    } catch (err) {
      messageApi.error(errorMessage(err))
    } finally {
      setActionLoading(null)
    }
  }

  const stopPaperRun = async () => {
    if (!paperRunJob) return
    setActionLoading('paper-stop')
    try {
      const job = await api.stopPaperRunJob(slug, paperRunJob.job_id)
      setPaperRunJob(job)
      messageApi.success('已请求停止模拟运行')
    } catch (err) {
      if (isJobGone(err)) {
        setPaperRunJob(null)
        return
      }
      messageApi.error(errorMessage(err))
    } finally {
      setActionLoading(null)
    }
  }

  const openFile = (key: string) => {
    setActiveFile(key)
    setActiveTab(key)
    if (!openFileTabs.includes(key)) {
      setOpenFileTabs([...openFileTabs, key])
    }
    loadArtifact(key)
  }

  const closeFileTab = (key: string, e: React.MouseEvent) => {
    e.stopPropagation()
    const newTabs = openFileTabs.filter(t => t !== key)
    setOpenFileTabs(newTabs)
    if (activeTab === key) {
      setActiveTab('overview')
      setActiveFile(null)
    }
  }

  const switchTab = (tabId: string) => {
    setActiveTab(tabId)
    if (openFileTabs.includes(tabId)) {
      setActiveFile(tabId)
    } else {
      setActiveFile(null)
    }
  }

  // Build file tree from artifacts
  const fileTree = useMemo(() => {
    const root: FolderNode = { name: slug, children: [] }
    const backtestFolder: FolderNode = { name: 'backtest', children: [] }
    const paperFolder: FolderNode = { name: 'paper_run', children: [] }

    artifacts.forEach(artifact => {
      const node: FileNode = {
        key: artifact.artifact_key,
        name: artifact.relative_path.split('/').pop() || artifact.artifact_key,
        exists: artifact.exists,
        artifactKey: artifact.artifact_key,
      }
      if (artifact.artifact_key.startsWith('backtest_')) {
        backtestFolder.children.push(node)
      } else if (artifact.artifact_key.startsWith('paper_run_')) {
        paperFolder.children.push(node)
      } else {
        root.children.push(node)
      }
    })

    if (backtestFolder.children.length > 0) root.children.push(backtestFolder)
    if (paperFolder.children.length > 0) root.children.push(paperFolder)
    return root
  }, [artifacts, slug])

  const currentStep = strategy ? getWorkflowStep(strategy.status) : 0
  const isPaperRunning = paperRunJob && ['queued', 'running'].includes(paperRunJob.status)

  interface NextAction {
    title: string
    description: string
    buttonText: string
    loading: boolean
    onClick: () => void
  }
  const nextAction: NextAction | null = useMemo(() => {
    if (!strategy) return null
    switch (strategy.status) {
      case 'draft':
        return null
      case 'designed':
        return {
          title: '下一步：生成策略代码',
          description: '设计文档已完成。让 AI 根据设计文档生成可运行的策略代码，才能继续回测。',
          buttonText: '生成策略代码',
          loading: actionLoading === 'codegen',
          onClick: () => runCodegen(false),
        }
      case 'coded':
        return {
          title: '下一步：运行回测',
          description: '策略代码已生成。用历史行情数据回测，验证策略是否有效。',
          buttonText: '运行回测',
          loading: actionLoading === 'backtest',
          onClick: runBacktest,
        }
      case 'backtested':
        return {
          title: '下一步：启动模拟运行',
          description: '回测已完成。用实时数据做模拟运行，观察策略在真实行情下的表现。',
          buttonText: '启动模拟运行',
          loading: actionLoading === 'paper-run',
          onClick: startPaperRun,
        }
      default:
        return null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [strategy, actionLoading])

  const renderFileTreeNode = (node: FileNode | FolderNode, depth = 0): React.ReactNode => {
    if (isFileNode(node)) {
      return (
        <div
          key={node.key}
          className={`tree-file ${activeFile === node.key ? 'active' : ''}`}
          style={{ paddingLeft: `${depth * 16 + 18}px` }}
          onClick={() => openFile(node.key)}
        >
          <span className="name">{node.name}</span>
          <span className={`dot ${node.exists ? 'ok' : 'miss'}`} />
        </div>
      )
    }
    return (
      <div key={node.name} className="tree-folder">
        <div
          className="tree-folder-header"
          style={{ paddingLeft: `${depth * 16 + 6}px` }}
          onClick={(e) => {
            const folder = e.currentTarget.parentElement
            if (folder) folder.classList.toggle('collapsed')
          }}
        >
          <span className="arrow">▼</span>
          <span>{node.name}</span>
        </div>
        <div className="tree-children">
          {node.children.map(child => renderFileTreeNode(child, depth + 1))}
        </div>
      </div>
    )
  }

  const fixedTabs = [
    { id: 'overview', label: '概览' },
    { id: 'backtest', label: '回测结果' },
    { id: 'research', label: '研究流程' },
    { id: 'paper', label: '模拟运行', live: isPaperRunning },
    { id: 'logs', label: '运行日志' },
  ]

  const renderTabContent = () => {
    if (activeFile) {
      const artifact = artifacts.find(a => a.artifact_key === activeFile)
      const content = artifactContent[activeFile]
      const isLoading = actionLoading === `artifact:${activeFile}`

      if (!artifact) return <Empty description="文件不存在" />
      if (!artifact.exists) {
        return (
          <Card className="workbench-card">
            <Empty description={`${artifact.relative_path} 尚未生成`} />
          </Card>
        )
      }
      if (isLoading) {
        return <Card className="workbench-card"><Spin /></Card>
      }
      if (!content) {
        return <Card className="workbench-card"><Empty description="加载中..." /></Card>
      }
      return (
        <Card className="workbench-card">
          <div className="card-title">
            <span className="card-title-icon blue">📄</span>
            {content.relative_path}
            <Text type="secondary" style={{ marginLeft: 'auto', fontSize: 12 }}>
              {content.size} bytes
            </Text>
          </div>
          <pre className="code-preview workbench-code">{content.content}</pre>
        </Card>
      )
    }

    switch (activeTab) {
      case 'overview':
        return (
          <div className="workbench-tab-content">
            <Card className="workbench-card">
              <div className="card-title">
                <span className="card-title-icon blue">⚡</span>
                工作流进度
              </div>
              <Steps
                current={currentStep}
                size="small"
                items={[
                  { title: '创建策略' },
                  { title: '设计文档' },
                  { title: '生成代码' },
                  { title: '回测验证' },
                  { title: '模拟运行' },
                  { title: '实盘部署' },
                ]}
              />
            </Card>

            {nextAction && (
              <Card className="workbench-card workbench-next-card">
                <div className="workbench-next-body">
                  <div className="workbench-next-text">
                    <div className="workbench-next-title">{nextAction.title}</div>
                    <Text type="secondary">{nextAction.description}</Text>
                  </div>
                  <Button
                    type="primary"
                    size="large"
                    icon={<ArrowRightOutlined />}
                    iconPlacement="end"
                    loading={nextAction.loading}
                    onClick={nextAction.onClick}
                  >
                    {nextAction.buttonText}
                  </Button>
                </div>
              </Card>
            )}

            <Card className="workbench-card">
              <div className="card-title">
                <span className="card-title-icon blue">⚡</span>
                快速操作
              </div>
              <Space wrap>
                <Button type="primary" icon={<PlayCircleOutlined />} onClick={runBacktest} loading={actionLoading === 'backtest'}>
                  运行回测
                </Button>
                {isPaperRunning ? (
                  <Button danger icon={<StopOutlined />} onClick={stopPaperRun} loading={actionLoading === 'paper-stop'}>
                    停止模拟
                  </Button>
                ) : (
                  <Button icon={<PlayCircleOutlined />} onClick={startPaperRun} loading={actionLoading === 'paper-run'}>
                    启动模拟运行
                  </Button>
                )}
                <Button icon={<ReloadOutlined />} onClick={() => runCodegen(true)} loading={actionLoading === 'codegen'}>
                  重新生成代码
                </Button>
              </Space>
            </Card>

            {backtestJob && (
              <Alert
                type={backtestJob.status === 'succeeded' ? 'success' : backtestJob.status === 'failed' || backtestJob.status === 'timed_out' ? 'error' : 'info'}
                title={`回测任务：${backtestJob.status}`}
                description={backtestJobMessage(backtestJob)}
                showIcon
                icon={backtestJob.status === 'queued' || backtestJob.status === 'running' ? <Spin size="small" /> : undefined}
                className="workbench-alert"
              />
            )}
            {paperRunJob && (
              <Alert
                type={paperRunJob.status === 'succeeded' || paperRunJob.status === 'stopped' ? 'success' : paperRunJob.status === 'failed' || paperRunJob.status === 'timed_out' ? 'error' : 'info'}
                title={`模拟运行任务：${paperRunJob.status}`}
                description={paperRunJobMessage(paperRunJob)}
                showIcon
                icon={paperRunJob.status === 'queued' || paperRunJob.status === 'running' ? <Spin size="small" /> : undefined}
                className="workbench-alert"
              />
            )}

            <Card className="workbench-card">
              <div className="card-title">
                <span className="card-title-icon green">📈</span>
                核心指标
              </div>
              {backtestResult ? (
                <>
                  <Row gutter={[16, 16]}>
                    <Col span={6}><Statistic title="评分" value={backtestResult.score} /></Col>
                    <Col span={6}><Statistic title="年化收益" value={formatMetric(backtestResult.result?.backtest?.annual_return, '%')} /></Col>
                    <Col span={6}><Statistic title="最大回撤" value={formatMetric(backtestResult.result?.backtest?.max_drawdown, '%')} /></Col>
                    <Col span={6}><Statistic title="夏普比率" value={formatMetric(backtestResult.result?.backtest?.sharpe)} /></Col>
                  </Row>
                  <Progress percent={Math.min(Math.max(backtestResult.score, 0), 100)} className="mt-16" />
                  <ResearchQualityAlert quality={backtestResult.result.research_quality} />
                </>
              ) : (
                <Empty description="暂无回测数据" image={Empty.PRESENTED_IMAGE_SIMPLE} />
              )}
            </Card>

            <Card className="workbench-card">
              <div className="card-title">
                <span className="card-title-icon orange">📋</span>
                策略摘要
              </div>
              <p className="strategy-summary">
                {strategy?.description || '暂无策略描述。'}
              </p>
            </Card>
          </div>
        )

      case 'backtest':
        return (
          <div className="workbench-tab-content">
            {backtestResult ? (
              <>
                <Card className="workbench-card">
                  <div className="card-title">
                    <span className="card-title-icon green">📊</span>
                    回测摘要
                  </div>
                  <Row gutter={[16, 16]}>
                    <Col span={6}><Statistic title="评分" value={backtestResult.score} /></Col>
                    <Col span={6}><Statistic title="年化收益" value={formatMetric(backtestResult.result?.backtest?.annual_return, '%')} /></Col>
                    <Col span={6}><Statistic title="最大回撤" value={formatMetric(backtestResult.result?.backtest?.max_drawdown, '%')} /></Col>
                    <Col span={6}><Statistic title="夏普比率" value={formatMetric(backtestResult.result?.backtest?.sharpe)} /></Col>
                  </Row>
                  <Progress percent={Math.min(Math.max(backtestResult.score, 0), 100)} className="mt-16" />
                  <ResearchQualityAlert quality={backtestResult.result.research_quality} />
                </Card>
                {Array.isArray(backtestResult.result?.backtest?.equity_curve) && backtestResult.result.backtest.equity_curve.length > 0 && (
                  <Card className="workbench-card">
                    <div className="card-title">
                      <span className="card-title-icon green">📈</span>
                      资产走势
                    </div>
                    <EquityCurveChart points={backtestResult.result.backtest.equity_curve} />
                  </Card>
                )}
                {Array.isArray(backtestResult.result?.backtest?.trades) && backtestResult.result.backtest.trades.length > 0 && (
                  <Card className="workbench-card">
                    <div className="card-title">
                      <span className="card-title-icon orange">🧾</span>
                      交易记录（{backtestResult.result.backtest.trades.length} 笔）
                    </div>
                    <Table
                      size="small"
                      rowKey={(row: any) => `${row.date}-${row.symbol}-${row.action}-${row.price}`}
                      pagination={false}
                      dataSource={backtestResult.result.backtest.trades}
                      columns={[
                        { title: '日期', dataIndex: 'date' },
                        { title: '标的', dataIndex: 'symbol' },
                        {
                          title: '操作',
                          dataIndex: 'action',
                          render: (action: string) => (
                            <Tag color={action === 'buy' ? 'green' : action === 'sell' ? 'red' : 'default'}>
                              {action === 'buy' ? '买入' : action === 'sell' ? '卖出' : action}
                            </Tag>
                          ),
                        },
                        { title: '价格', dataIndex: 'price', render: (v: number) => formatMetric(v) },
                        { title: '数量', dataIndex: 'quantity', render: (v: number) => formatMetric(v) },
                        { title: '金额', dataIndex: 'cost', render: (v: number) => formatMetric(v) },
                      ]}
                    />
                  </Card>
                )}
                <Collapse
                  className="workbench-card"
                  items={[{
                    key: 'raw',
                    label: '查看原始数据（JSON）',
                    children: <pre className="code-preview">{JSON.stringify(backtestResult.result, null, 2)}</pre>,
                  }]}
                />
              </>
            ) : (
              <Card className="workbench-card">
                <Empty description="暂无回测结果">
                  <Button type="primary" onClick={runBacktest} loading={actionLoading === 'backtest'}>
                    立即运行回测
                  </Button>
                </Empty>
              </Card>
            )}
          </div>
        )

      case 'research':
        return (
          <ResearchPanel
            slug={slug}
            strategy={strategy}
            onStrategyChanged={refreshStrategy}
          />
        )

      case 'paper':
        return (
          <div className="workbench-tab-content">
            <Card className="workbench-card">
              <div className="card-title">
                <span className="card-title-icon blue">⚙</span>
                选择模拟运行方式
              </div>
              <Segmented
                block
                value={simulationMode}
                onChange={(value) => setSimulationMode(value as 'local' | 'ft-client')}
                options={[
                  { label: '本地历史回放', value: 'local' },
                  { label: '非凸客户端模拟盘', value: 'ft-client' },
                ]}
              />
            </Card>
            {simulationMode === 'ft-client' ? (
              <ClientSimulationPanel slug={slug} />
            ) : (
              <>
            {paperRunResult ? (
              <>
                <Card className="workbench-card">
                  <div className="card-title">
                    <span className="card-title-icon green">●</span>
                    模拟运行状态
                    <Tag color={isPaperRunning ? 'cyan' : 'default'} style={{ marginLeft: 'auto' }}>
                      {paperRunResult.result?.run_status || 'unknown'}
                    </Tag>
                  </div>
                  <Row gutter={[16, 16]}>
                    <Col span={6}><Statistic title="模拟收益" value={formatPercentValue(paperRunResult.result?.summary?.paper_return)} /></Col>
                    <Col span={6}><Statistic title="最大回撤" value={formatPercentValue(paperRunResult.result?.summary?.paper_max_drawdown)} /></Col>
                    <Col span={6}><Statistic title="交易次数" value={formatMetric(paperRunResult.result?.summary?.trade_count)} /></Col>
                    <Col span={6}><Statistic title="最终资产" value={formatMetric(paperRunResult.result?.summary?.final_value)} /></Col>
                  </Row>
                  <Row gutter={[16, 16]} className="mt-16">
                    <Col span={6}><Statistic title="现金" value={formatMetric(paperRunResult.result?.paper?.cash)} /></Col>
                    <Col span={6}><Statistic title="持仓市值" value={formatMetric(paperRunResult.result?.paper?.equity)} /></Col>
                    <Col span={6}><Statistic title="持仓数量" value={formatMetric(paperRunResult.result?.paper?.position_count)} /></Col>
                    <Col span={6}><Statistic title="未实现盈亏" value={formatMetric(paperRunResult.result?.paper?.unrealized_pnl)} /></Col>
                  </Row>
                  <Progress percent={Math.round((paperRunResult.result?.replay?.progress || 0) * 100)} className="mt-16" />
                  {paperRunResult.result?.replay && (
                    <Descriptions bordered size="small" column={2} className="mt-16">
                      <Descriptions.Item label="当前时间">{paperRunResult.result.replay.current_at || 'N/A'}</Descriptions.Item>
                      <Descriptions.Item label="已处理 bars">{formatMetric(paperRunResult.result.replay.bars_processed)}</Descriptions.Item>
                      <Descriptions.Item label="最近动作">{paperRunResult.result.latest_decision?.action || 'N/A'}</Descriptions.Item>
                      <Descriptions.Item label="原因">{paperRunResult.result.latest_decision?.reason || 'N/A'}</Descriptions.Item>
                    </Descriptions>
                  )}
                </Card>
                {paperRunResult.result?.paper?.positions?.length > 0 && (
                  <Card className="workbench-card">
                    <div className="card-title">当前持仓</div>
                    <Descriptions bordered size="small" column={2}>
                      {paperRunResult.result.paper.positions.map((pos: any) => (
                        <Descriptions.Item key={pos.symbol} label={pos.symbol}>
                          {`${pos.quantity} 股 @ ${formatMetric(pos.avg_price)}（市值 ${formatMetric(pos.market_value)}）`}
                        </Descriptions.Item>
                      ))}
                    </Descriptions>
                  </Card>
                )}
              </>
            ) : (
              <Card className="workbench-card">
                <Empty description="暂无模拟运行数据">
                  <Button type="primary" onClick={startPaperRun} loading={actionLoading === 'paper-run'}>
                    启动模拟运行
                  </Button>
                </Empty>
              </Card>
            )}
              </>
            )}
          </div>
        )

      case 'logs':
        return (
          <div className="workbench-tab-content">
            <Card className="workbench-card">
              <div className="card-title">
                <span className="card-title-icon orange">📝</span>
                运行日志
              </div>
              <div className="log-panel">
                <div><span className="log-time">[系统]</span> <span className="log-info">[INFO]</span> 策略 {slug} 已加载</div>
                <div><span className="log-time">[系统]</span> <span className="log-info">[INFO]</span> 配置文件验证通过</div>
                {backtestJob && (
                  <div><span className="log-time">[回测]</span> <span className="log-info">[INFO]</span> {backtestJobMessage(backtestJob)}</div>
                )}
                {paperRunJob && (
                  <div><span className="log-time">[模拟]</span> <span className="log-info">[INFO]</span> {paperRunJobMessage(paperRunJob)}</div>
                )}
                {!backtestJob && !paperRunJob && (
                  <div><span className="log-time">[系统]</span> <span className="log-warn">[WARN]</span> 暂无运行任务</div>
                )}
              </div>
            </Card>
          </div>
        )

      default:
        return null
    }
  }

  if (loading && !strategy) {
    return (
      <Layout className="workbench-shell">
        <Header className="workbench-header">
          <Space>
            <Button icon={<ArrowLeftOutlined />} onClick={onBack}>返回列表</Button>
            <Title level={4} style={{ margin: 0, color: '#fff' }}>加载中...</Title>
          </Space>
        </Header>
        <Content className="workbench-loading"><Spin size="large" /></Content>
      </Layout>
    )
  }

  return (
    <Layout className="workbench-shell">
      {contextHolder}
      <Header className="workbench-header">
        <div className="workbench-header-left">
          <Button type="text" icon={<ArrowLeftOutlined />} onClick={onBack} style={{ color: '#fff' }}>
            返回列表
          </Button>
          <span className="workbench-logo">autostrategy</span>
          <span className="workbench-divider">|</span>
          <span className="workbench-strategy-title">策略工作台 · {strategy?.name || slug}</span>
        </div>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={refreshStrategy} loading={loading}>刷新</Button>
        </Space>
      </Header>

      <div className="workbench-info-bar">
        <div className="info-item">
          <span className="info-label">状态</span>
          <Tag color={STATUS_COLOR[strategy?.status || 'draft']}>{STATUS_LABEL[strategy?.status || 'draft'] || strategy?.status}</Tag>
        </div>
        <div className="info-item">
          <span className="info-label">市场</span>
          <span className="info-value">{strategy?.market || 'N/A'}</span>
        </div>
        <div className="info-item">
          <span className="info-label">模板</span>
          <span className="info-value">{strategy?.template || '无'}</span>
        </div>
        <div className="info-item" style={{ marginLeft: 'auto' }}>
          <span className="info-label">策略ID</span>
          <span className="info-value" style={{ fontFamily: 'monospace' }}>{slug}</span>
        </div>
        <div className="info-actions">
          <Button size="small" type="primary" icon={<PlayCircleOutlined />} onClick={runBacktest} loading={actionLoading === 'backtest'}>
            运行回测
          </Button>
          {isPaperRunning ? (
            <Button size="small" danger icon={<StopOutlined />} onClick={stopPaperRun} loading={actionLoading === 'paper-stop'}>
              停止模拟
            </Button>
          ) : (
            <Button size="small" icon={<PlayCircleOutlined />} onClick={startPaperRun} loading={actionLoading === 'paper-run'}>
              启动模拟
            </Button>
          )}
          <Button size="small" icon={<ReloadOutlined />} onClick={() => runCodegen(true)} loading={actionLoading === 'codegen'}>
            重新生成
          </Button>
        </div>
      </div>

      <div className="workbench-main">
        <aside className="workbench-sidebar">
          <div className="sidebar-header">
            <span>策略文件</span>
            <Space size={0}>
              <Button
                type="text"
                size="small"
                title="打开本地文件夹"
                icon={<FolderOpenOutlined />}
                onClick={revealInFinder}
                loading={actionLoading === 'reveal'}
              />
              <Button type="text" size="small" title="刷新" icon={<ReloadOutlined />} onClick={refreshStrategy} />
            </Space>
          </div>
          <div className="file-tree">
            {fileTree.children.map(child => renderFileTreeNode(child))}
          </div>
        </aside>

        <div className="workbench-center">
          <div className="tab-bar">
            {fixedTabs.map(tab => (
              <div
                key={tab.id}
                className={`tab pinned ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => switchTab(tab.id)}
              >
                {tab.label}
                {tab.live && <span className="live-dot" />}
              </div>
            ))}
            {openFileTabs.map(tabId => {
              const artifact = artifacts.find(a => a.artifact_key === tabId)
              return (
                <div
                  key={tabId}
                  className={`tab ${activeTab === tabId ? 'active' : ''}`}
                  onClick={() => switchTab(tabId)}
                >
                  {artifact?.relative_path.split('/').pop() || tabId}
                  <span className="close" onClick={(e) => closeFileTab(tabId, e)}>×</span>
                </div>
              )
            })}
          </div>
          <div className="tab-content">
            {renderTabContent()}
          </div>
        </div>
      </div>
    </Layout>
  )
}
