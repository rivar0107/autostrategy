import type {
  ArtifactContent,
  ArtifactList,
  AppInfo,
  BacktestJobResponse,
  BacktestResponse,
  ConfigResponse,
  ClientSimulationPreflight,
  ClientSimulationRequest,
  ClientSimulationSession,
  DatasetManifest,
  DatasetManifestCreate,
  DesignJobResponse,
  ExperimentSession,
  FtAccount,
  FtConnectionInput,
  LlmConfigUpdate,
  PaperRunResponse,
  Strategy,
  StrategyDetail,
  StrategyVersion,
} from '../types'

const API_PREFIX = '/api/v1'

export class ApiError extends Error {
  code: string
  status: number
  details: Record<string, unknown>

  constructor(message: string, code = 'api_error', details: Record<string, unknown> = {}, status = 0) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
    this.details = details
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_PREFIX}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {}),
    },
    ...init,
  })
  const text = await response.text()
  let data: any = null
  if (text) {
    try {
      data = JSON.parse(text)
    } catch {
      data = null
    }
  }
  if (!response.ok) {
    const error = data?.error || {}
    throw new ApiError(
      error.message || text || `HTTP ${response.status}`,
      error.code || 'api_error',
      error.details || {},
      response.status,
    )
  }
  return data as T
}

export const api = {
  info: () => request<AppInfo>('/info'),
  config: () => request<ConfigResponse>('/config'),
  updateLlmConfig: (payload: LlmConfigUpdate) =>
    request<ConfigResponse>('/config/llm', { method: 'PUT', body: JSON.stringify(payload) }),
  templates: () => request<string[]>('/templates'),
  strategies: () => request<Strategy[]>('/strategies'),
  createStrategy: (payload: { name: string; market: string; template?: string | null }) =>
    request<Strategy>('/strategies', { method: 'POST', body: JSON.stringify(payload) }),
  deleteStrategy: (slug: string) => request<void>(`/strategies/${slug}`, { method: 'DELETE' }),
  revealStrategy: (slug: string) => request<void>(`/strategies/${slug}/reveal`, { method: 'POST' }),
  strategyDetail: (slug: string) => request<StrategyDetail>(`/strategies/${slug}`),
  createDesign: (payload: { name: string; prompt: string; market: string; template?: string | null }) =>
    request<DesignJobResponse>('/designs', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  designJob: (jobId: string) => request<DesignJobResponse>(`/design-jobs/${jobId}`),
  codegen: (slug: string, force = false) =>
    request<{ strategy: Strategy; generated_files: string[] }>(`/strategies/${slug}/codegen`, {
      method: 'POST',
      body: JSON.stringify({ force }),
    }),
  backtest: (slug: string) =>
    request<BacktestJobResponse>(`/strategies/${slug}/backtest`, { method: 'POST' }),
  backtestJob: (slug: string, jobId: string) =>
    request<BacktestJobResponse>(`/strategies/${slug}/backtest-jobs/${jobId}`),
  backtestResult: (slug: string) => request<BacktestResponse>(`/strategies/${slug}/backtest-result`),
  startPaperRun: (slug: string) =>
    request<BacktestJobResponse>(`/strategies/${slug}/paper-run`, { method: 'POST' }),
  paperRunJob: (slug: string, jobId: string) =>
    request<BacktestJobResponse>(`/strategies/${slug}/paper-run-jobs/${jobId}`),
  stopPaperRunJob: (slug: string, jobId: string) =>
    request<BacktestJobResponse>(`/strategies/${slug}/paper-run-jobs/${jobId}/stop`, { method: 'POST' }),
  paperRunResult: (slug: string) => request<PaperRunResponse>(`/strategies/${slug}/paper-run-result`),
  checkFtClientConnection: (payload: FtConnectionInput) =>
    request<ClientSimulationPreflight>('/broker-connections/ft-client/check', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  ftClientAccounts: () => request<FtAccount[]>('/broker-connections/ft-client/accounts'),
  preflightClientSimulation: (slug: string, payload: ClientSimulationRequest) =>
    request<ClientSimulationPreflight>(`/strategies/${slug}/client-simulation/preflight`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  createClientSimulationSession: (slug: string, payload: ClientSimulationRequest) =>
    request<ClientSimulationSession>(`/strategies/${slug}/client-simulation/sessions`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  clientSimulationSessions: (slug: string) =>
    request<ClientSimulationSession[]>(`/strategies/${slug}/client-simulation/sessions`),
  clientSimulationSession: (slug: string, sessionId: string) =>
    request<ClientSimulationSession>(`/strategies/${slug}/client-simulation/sessions/${sessionId}`),
  pauseClientSimulationSession: (slug: string, sessionId: string) =>
    request<ClientSimulationSession>(`/strategies/${slug}/client-simulation/sessions/${sessionId}/pause`, {
      method: 'POST',
    }),
  resumeClientSimulationSession: (slug: string, sessionId: string) =>
    request<ClientSimulationSession>(`/strategies/${slug}/client-simulation/sessions/${sessionId}/resume`, {
      method: 'POST',
    }),
  stopClientSimulationSession: (slug: string, sessionId: string) =>
    request<ClientSimulationSession>(`/strategies/${slug}/client-simulation/sessions/${sessionId}/stop`, {
      method: 'POST',
    }),
  approveClientSimulationIntent: (slug: string, sessionId: string, intentId: string) =>
    request<ClientSimulationSession>(
      `/strategies/${slug}/client-simulation/sessions/${sessionId}/intents/${intentId}/approve`,
      { method: 'POST' },
    ),
  rejectClientSimulationIntent: (slug: string, sessionId: string, intentId: string, reason: string) =>
    request<ClientSimulationSession>(
      `/strategies/${slug}/client-simulation/sessions/${sessionId}/intents/${intentId}/reject`,
      { method: 'POST', body: JSON.stringify({ reason }) },
    ),
  artifacts: (slug: string) => request<ArtifactList>(`/strategies/${slug}/artifacts`),
  artifact: (slug: string, key: string) =>
    request<ArtifactContent>(`/strategies/${slug}/artifacts/${key}`),
  strategyVersions: (slug: string) =>
    request<StrategyVersion[]>(`/strategies/${slug}/versions`),
  rollbackStrategyVersion: (slug: string, versionId: string, reason: string) =>
    request<StrategyVersion>(`/strategies/${slug}/versions/${versionId}/rollback`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    }),
  datasetManifests: (slug: string) =>
    request<DatasetManifest[]>(`/strategies/${slug}/dataset-manifests`),
  captureDatasetManifest: (slug: string, payload: DatasetManifestCreate) =>
    request<DatasetManifest>(`/strategies/${slug}/dataset-manifests`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  experiments: (slug: string) =>
    request<ExperimentSession[]>(`/strategies/${slug}/experiments`),
  createExperiment: (slug: string, baseVersionId: string, manifestId: string) =>
    request<ExperimentSession>(`/strategies/${slug}/experiments`, {
      method: 'POST',
      body: JSON.stringify({ base_version_id: baseVersionId, manifest_id: manifestId }),
    }),
  runExperimentBaseline: (slug: string, sessionId: string) =>
    request<ExperimentSession>(`/strategies/${slug}/experiments/${sessionId}/baseline`, {
      method: 'POST',
    }),
  diagnoseExperiment: (slug: string, sessionId: string) =>
    request<ExperimentSession>(`/strategies/${slug}/experiments/${sessionId}/diagnose`, {
      method: 'POST',
    }),
  optimizeExperiment: (slug: string, sessionId: string) =>
    request<ExperimentSession>(`/strategies/${slug}/experiments/${sessionId}/optimize`, {
      method: 'POST',
      body: JSON.stringify({}),
    }),
  validateExperimentOos: (slug: string, sessionId: string) =>
    request<ExperimentSession>(`/strategies/${slug}/experiments/${sessionId}/validate-oos`, {
      method: 'POST',
      body: JSON.stringify({}),
    }),
  decideExperiment: (slug: string, sessionId: string, decision: 'accept' | 'reject', reason: string) =>
    request<ExperimentSession>(`/strategies/${slug}/experiments/${sessionId}/${decision}`, {
      method: 'POST',
      body: JSON.stringify({ reason }),
    }),
}
