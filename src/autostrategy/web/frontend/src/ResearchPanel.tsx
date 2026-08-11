import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Col,
  Collapse,
  Descriptions,
  Empty,
  Input,
  Modal,
  Row,
  Select,
  Space,
  Spin,
  Steps,
  Table,
  Tag,
  Typography,
  message,
} from 'antd'
import { ReloadOutlined } from '@ant-design/icons'
import { ApiError, api } from './api/client'
import type {
  DatasetManifest,
  DatasetManifestCreate,
  ExperimentSession,
  ExperimentStatus,
  Strategy,
  StrategyVersion,
} from './types'

const { Text } = Typography

const STATUS_LABEL: Record<ExperimentStatus, string> = {
  created: '已创建',
  baseline_completed: '基线完成',
  diagnosed: '诊断完成',
  optimized: '候选已选定',
  oos_validated: '样本外已验证',
  awaiting_decision: '等待决策',
  accepted: '已接受',
  rejected: '已拒绝',
  failed: '失败',
}

const STATUS_STEP: Record<ExperimentStatus, number> = {
  created: 0,
  baseline_completed: 1,
  diagnosed: 2,
  optimized: 3,
  oos_validated: 4,
  awaiting_decision: 5,
  accepted: 6,
  rejected: 6,
  failed: 0,
}

const emptyDraft: DatasetManifestCreate = {
  version_id: '',
  train: { start: '', end: '' },
  validation: { start: '', end: '' },
  test: { start: '', end: '' },
  benchmark: '',
  data_source: 'strategy_fetch',
  frequency: 'daily',
  adjustment: 'forward',
}

type Decision =
  | { action: 'accept' | 'reject'; sessionId: string }
  | { action: 'rollback'; versionId: string }

function errorText(error: unknown): string {
  if (error instanceof ApiError) return error.message
  if (error instanceof Error) return error.message
  return '研究操作失败'
}

function shortId(value?: string | null): string {
  return value ? `${value.slice(0, 10)}…` : 'N/A'
}

function rangeText(range?: { start: string; end: string }): string {
  return range ? `${range.start} → ${range.end}` : 'N/A'
}

function replaceSession(
  sessions: ExperimentSession[],
  updated: ExperimentSession,
): ExperimentSession[] {
  return [updated, ...sessions.filter((item) => item.session_id !== updated.session_id)]
}

interface ResearchPanelProps {
  slug: string
  strategy: Strategy | null
  onStrategyChanged: () => Promise<void>
}

export default function ResearchPanel({ slug, strategy, onStrategyChanged }: ResearchPanelProps) {
  const [versions, setVersions] = useState<StrategyVersion[]>([])
  const [manifests, setManifests] = useState<DatasetManifest[]>([])
  const [sessions, setSessions] = useState<ExperimentSession[]>([])
  const [selectedSessionId, setSelectedSessionId] = useState<string>()
  const [selectedManifestId, setSelectedManifestId] = useState<string>()
  const [draft, setDraft] = useState<DatasetManifestCreate>(emptyDraft)
  const [loading, setLoading] = useState(true)
  const [action, setAction] = useState<string>()
  const [decision, setDecision] = useState<Decision>()
  const [reason, setReason] = useState('')
  const [messageApi, contextHolder] = message.useMessage()

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const [nextVersions, nextManifests, nextSessions] = await Promise.all([
        api.strategyVersions(slug),
        api.datasetManifests(slug),
        api.experiments(slug),
      ])
      setVersions(nextVersions)
      setManifests(nextManifests)
      setSessions(nextSessions)
      setSelectedSessionId((current) => (
        current && nextSessions.some((item) => item.session_id === current)
          ? current
          : nextSessions[0]?.session_id
      ))
      setSelectedManifestId((current) => (
        current && nextManifests.some((item) => item.manifest_id === current)
          ? current
          : nextManifests[0]?.manifest_id
      ))
      setDraft((current) => ({
        ...current,
        version_id: current.version_id || nextVersions.find((item) => item.version_id === strategy?.active_version_id)?.version_id || nextVersions[nextVersions.length - 1]?.version_id || '',
      }))
    } catch (error) {
      messageApi.error(errorText(error))
    } finally {
      setLoading(false)
    }
  }, [messageApi, slug, strategy?.active_version_id])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const session = useMemo(
    () => sessions.find((item) => item.session_id === selectedSessionId) || sessions[0],
    [selectedSessionId, sessions],
  )
  const manifest = manifests.find((item) => item.manifest_id === session?.manifest_id)
  const rollbackAncestorIds = useMemo(() => {
    const byId = new Map(versions.map((item) => [item.version_id, item]))
    const ancestors = new Set<string>()
    let cursor = strategy?.active_version_id
      ? byId.get(strategy.active_version_id)
      : undefined
    while (cursor?.parent_version_id) {
      ancestors.add(cursor.parent_version_id)
      cursor = byId.get(cursor.parent_version_id)
    }
    return ancestors
  }, [strategy?.active_version_id, versions])

  const updateSession = (updated: ExperimentSession) => {
    setSessions((current) => replaceSession(current, updated))
    setSelectedSessionId(updated.session_id)
  }

  const runStep = async (step: 'baseline' | 'diagnose' | 'optimize' | 'oos') => {
    if (!session) return
    setAction(step)
    try {
      const updated = step === 'baseline'
        ? await api.runExperimentBaseline(slug, session.session_id)
        : step === 'diagnose'
          ? await api.diagnoseExperiment(slug, session.session_id)
          : step === 'optimize'
            ? await api.optimizeExperiment(slug, session.session_id)
            : await api.validateExperimentOos(slug, session.session_id)
      updateSession(updated)
      messageApi.success('研究阶段已完成')
    } catch (error) {
      messageApi.error(errorText(error))
      await refresh()
    } finally {
      setAction(undefined)
    }
  }

  const createManifest = async () => {
    const ranges = [draft.train, draft.validation, draft.test]
    if (!draft.version_id || !draft.benchmark.trim() || ranges.some((range) => !range.start || !range.end)) {
      messageApi.error('请完整填写版本、基准和三个数据区间')
      return
    }
    setAction('manifest')
    try {
      const created = await api.captureDatasetManifest(slug, draft)
      setManifests((current) => [created, ...current])
      setSelectedManifestId(created.manifest_id)
      messageApi.success('数据快照已冻结')
    } catch (error) {
      messageApi.error(errorText(error))
    } finally {
      setAction(undefined)
    }
  }

  const createExperiment = async () => {
    const selected = manifests.find((item) => item.manifest_id === selectedManifestId)
    if (!selected) {
      messageApi.error('请先选择或创建数据快照')
      return
    }
    setAction('session')
    try {
      const created = await api.createExperiment(slug, selected.version_id, selected.manifest_id)
      updateSession(created)
      messageApi.success('实验会话已创建')
    } catch (error) {
      messageApi.error(errorText(error))
    } finally {
      setAction(undefined)
    }
  }

  const submitDecision = async () => {
    if (!decision || !reason.trim()) return
    setAction(`decision:${decision.action}`)
    try {
      if (decision.action === 'rollback') {
        await api.rollbackStrategyVersion(slug, decision.versionId, reason.trim())
        await onStrategyChanged()
        await refresh()
      } else {
        const updated = await api.decideExperiment(
          slug,
          decision.sessionId,
          decision.action,
          reason.trim(),
        )
        updateSession(updated)
        if (decision.action === 'accept') await onStrategyChanged()
      }
      messageApi.success(decision.action === 'rollback' ? '版本已回滚' : '研究决策已保存')
      setDecision(undefined)
      setReason('')
    } catch (error) {
      messageApi.error(errorText(error))
    } finally {
      setAction(undefined)
    }
  }

  const nextActions: Partial<Record<ExperimentStatus, { label: string; step: 'baseline' | 'diagnose' | 'optimize' | 'oos' }>> = {
    created: { label: '运行基线', step: 'baseline' as const },
    baseline_completed: { label: '生成诊断', step: 'diagnose' as const },
    diagnosed: { label: '自动生成并评估候选', step: 'optimize' as const },
    optimized: { label: '揭晓一次样本外结果', step: 'oos' as const },
  }
  const nextAction = session ? nextActions[session.status] : undefined

  const decisionTitle = decision?.action === 'accept'
    ? '确认接受候选版本'
    : decision?.action === 'reject'
      ? '确认拒绝候选版本'
      : '确认回滚策略版本'
  const decisionButton = decision?.action === 'accept'
    ? '确认接受'
    : decision?.action === 'reject'
      ? '确认拒绝'
      : '确认回滚'

  if (loading) return <Card className="workbench-card"><Spin /></Card>

  return (
    <div className="workbench-tab-content">
      {contextHolder}
      <Card className="workbench-card">
        <div className="card-title">
          <span className="card-title-icon blue">🧪</span>
          可复现研究闭环
          <Button size="small" icon={<ReloadOutlined />} onClick={refresh} style={{ marginLeft: 'auto' }}>
            刷新研究状态
          </Button>
        </div>
        <Alert
          type="info"
          showIcon
          title="候选优化不会修改当前正式策略；测试集只允许揭晓一次。"
        />
        {session && (
          <Steps
            className="mt-16"
            size="small"
            current={STATUS_STEP[session.status]}
            status={session.status === 'failed' ? 'error' : 'process'}
            items={['创建', '基线', '诊断', '优化', '样本外', '决策', '完成'].map((title) => ({ title }))}
          />
        )}
      </Card>

      <Card className="workbench-card" title="实验输入（不可变）">
        {session && manifest ? (
          <Descriptions bordered size="small" column={2}>
            <Descriptions.Item label="会话状态"><Tag>{STATUS_LABEL[session.status]}</Tag></Descriptions.Item>
            <Descriptions.Item label="实验会话"><Text code>{shortId(session.session_id)}</Text></Descriptions.Item>
            <Descriptions.Item label="基础版本"><Text code>{shortId(session.base_version_id)}</Text></Descriptions.Item>
            <Descriptions.Item label="数据摘要"><Text code>{shortId(manifest.data_digest)}</Text></Descriptions.Item>
            <Descriptions.Item label="训练集">{rangeText(manifest.train)}</Descriptions.Item>
            <Descriptions.Item label="验证集">{rangeText(manifest.validation)}</Descriptions.Item>
            <Descriptions.Item label="测试集">{rangeText(manifest.test)}</Descriptions.Item>
            <Descriptions.Item label="基准 / 成本">{manifest.benchmark} / {manifest.commission} / {manifest.slippage}</Descriptions.Item>
          </Descriptions>
        ) : (
          <Empty description="尚未创建实验会话" />
        )}
        <Space wrap className="mt-16">
          <Select
            style={{ minWidth: 260 }}
            placeholder="选择数据快照"
            value={selectedManifestId}
            onChange={setSelectedManifestId}
            options={manifests.map((item) => ({
              value: item.manifest_id,
              label: `${item.benchmark} · ${shortId(item.data_digest)}`,
            }))}
          />
          <Button type="primary" onClick={createExperiment} loading={action === 'session'} disabled={!selectedManifestId}>
            创建新实验
          </Button>
          {sessions.length > 1 && (
            <Select
              style={{ minWidth: 260 }}
              value={session?.session_id}
              onChange={setSelectedSessionId}
              options={sessions.map((item) => ({ value: item.session_id, label: `${STATUS_LABEL[item.status]} · ${shortId(item.session_id)}` }))}
            />
          )}
        </Space>
      </Card>

      {session && (
        <>
          {session.error && <Alert className="workbench-card" type="error" showIcon title="实验执行失败" description={session.error} />}
          {nextAction && (
            <Card className="workbench-card workbench-next-card">
              <div className="workbench-next-body">
                <div>
                  <div className="workbench-next-title">下一步：{nextAction.label}</div>
                  <Text type="secondary">系统会持久化本阶段结果，完成后才能进入下一阶段。</Text>
                </div>
                <Button type="primary" onClick={() => runStep(nextAction.step)} loading={action === nextAction.step}>
                  {nextAction.label}
                </Button>
              </div>
            </Card>
          )}

          {session.diagnostics.length > 0 && (
            <Card className="workbench-card" title={`结构化诊断（${session.diagnostics.length}）`}>
              <Table
                size="small"
                pagination={false}
                rowKey="code"
                dataSource={session.diagnostics}
                columns={[
                  { title: '级别', dataIndex: 'severity', render: (value: string) => <Tag color={value === 'critical' ? 'red' : value === 'warning' ? 'orange' : 'blue'}>{value}</Tag> },
                  { title: '诊断', dataIndex: 'hypothesis' },
                  { title: '建议', dataIndex: 'suggested_actions', render: (items: string[]) => items.join('；') || '—' },
                ]}
              />
            </Card>
          )}

          {session.candidates.length > 0 && (
            <Card className="workbench-card" title={`优化候选（${session.candidates.length}）`}>
              <Table
                size="small"
                pagination={false}
                rowKey="candidate_id"
                dataSource={session.candidates}
                columns={[
                  { title: '候选', dataIndex: 'name' },
                  { title: '状态', dataIndex: 'status', render: (value: string) => <Tag color={value === 'selected' ? 'green' : 'default'}>{value}</Tag> },
                  { title: '验证分', dataIndex: 'validation_score', render: (value?: number) => value ?? '—' },
                  { title: '提升', dataIndex: 'improvement', render: (value?: number) => value == null ? '—' : `${value > 0 ? '+' : ''}${value}` },
                  { title: '参数变化', dataIndex: 'config_overrides', render: (value: Record<string, unknown>) => <Text code>{JSON.stringify(value)}</Text> },
                ]}
              />
            </Card>
          )}

          {session.oos_revealed && (
            <Alert
              className="workbench-card"
              type={session.oos_passed ? 'success' : session.status === 'failed' ? 'error' : 'warning'}
              showIcon
              title={session.oos_passed ? '样本外验证通过' : '样本外验证未通过'}
              description="测试集已经揭晓，本实验不能再次运行样本外验证。"
            />
          )}

          {session.status === 'awaiting_decision' && (
            <Card className="workbench-card" title="人工决策">
              <Space>
                <Button type="primary" onClick={() => setDecision({ action: 'accept', sessionId: session.session_id })}>接受候选</Button>
                <Button danger onClick={() => setDecision({ action: 'reject', sessionId: session.session_id })}>拒绝候选</Button>
              </Space>
            </Card>
          )}
          {(session.status === 'accepted' || session.status === 'rejected') && (
            <Alert
              className="workbench-card"
              type={session.status === 'accepted' ? 'success' : 'info'}
              showIcon
              title={STATUS_LABEL[session.status]}
              description={session.decision_reason || '已保存决策'}
            />
          )}
        </>
      )}

      <Card className="workbench-card" title="版本历史与回滚">
        <Table
          size="small"
          pagination={false}
          rowKey="version_id"
          dataSource={[...versions].reverse()}
          columns={[
            { title: '版本', dataIndex: 'version', render: (value: number) => `v${value}` },
            { title: '状态', dataIndex: 'state', render: (value: string) => <Tag>{value}</Tag> },
            { title: '变更', dataIndex: 'change_summary' },
            { title: '摘要', dataIndex: 'content_digest', render: (value: string) => <Text code>{shortId(value)}</Text> },
            {
              title: '操作',
              render: (_, version: StrategyVersion) => version.state === 'accepted' && rollbackAncestorIds.has(version.version_id)
                ? <Button size="small" onClick={() => setDecision({ action: 'rollback', versionId: version.version_id })}>回滚到此版本</Button>
                : <Text type="secondary">{version.version_id === strategy?.active_version_id ? '当前版本' : '—'}</Text>,
            },
          ]}
        />
      </Card>

      <Collapse
        className="workbench-card"
        items={[{
          key: 'manifest',
          label: '冻结新的 DatasetManifest',
          children: (
            <>
              <Row gutter={[12, 12]}>
                <Col span={12}>
                  <Select
                    aria-label="策略版本"
                    className="full-width"
                    value={draft.version_id || undefined}
                    placeholder="选择策略版本"
                    onChange={(value) => setDraft((current) => ({ ...current, version_id: value }))}
                    options={versions.filter((item) => item.state === 'accepted').map((item) => ({ value: item.version_id, label: `v${item.version} · ${item.change_summary}` }))}
                  />
                </Col>
                <Col span={12}><Input aria-label="回测基准" value={draft.benchmark} placeholder="基准，例如 000905.SH" onChange={(event) => setDraft((current) => ({ ...current, benchmark: event.target.value }))} /></Col>
                {(['train', 'validation', 'test'] as const).map((split) => (
                  <Col span={8} key={split}>
                    <Text strong>{split === 'train' ? '训练集' : split === 'validation' ? '验证集' : '测试集'}</Text>
                    <Space.Compact className="full-width mt-16">
                      <Input type="date" aria-label={`${split}-start`} value={draft[split].start} onChange={(event) => setDraft((current) => ({ ...current, [split]: { ...current[split], start: event.target.value } }))} />
                      <Input type="date" aria-label={`${split}-end`} value={draft[split].end} onChange={(event) => setDraft((current) => ({ ...current, [split]: { ...current[split], end: event.target.value } }))} />
                    </Space.Compact>
                  </Col>
                ))}
              </Row>
              <Button className="mt-16" type="primary" onClick={createManifest} loading={action === 'manifest'}>抓取并冻结数据</Button>
            </>
          ),
        }]}
      />

      <Modal
        open={Boolean(decision)}
        title={decisionTitle}
        okText={decisionButton}
        cancelText="取消"
        confirmLoading={action?.startsWith('decision:')}
        okButtonProps={{ disabled: !reason.trim(), danger: decision?.action !== 'accept' }}
        onOk={submitDecision}
        onCancel={() => { setDecision(undefined); setReason('') }}
      >
        <Alert
          type={decision?.action === 'accept' ? 'warning' : 'info'}
          showIcon
          title={decision?.action === 'accept' ? '接受后将切换正式策略版本。' : '该操作会被写入版本审计记录。'}
          className="mb-16"
        />
        <Input.TextArea
          autoFocus
          rows={3}
          value={reason}
          placeholder="请输入决策原因"
          onChange={(event) => setReason(event.target.value)}
        />
      </Modal>
    </div>
  )
}
