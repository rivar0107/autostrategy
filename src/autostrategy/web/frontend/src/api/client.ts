import type {
  ArtifactContent,
  ArtifactList,
  AppInfo,
  BacktestJobResponse,
  BacktestResponse,
  ConfigResponse,
  LlmConfigUpdate,
  PaperRunResponse,
  Strategy,
  StrategyDetail,
} from '../types'

const API_PREFIX = '/api/v1'

export class ApiError extends Error {
  code: string
  details: Record<string, unknown>

  constructor(message: string, code = 'api_error', details: Record<string, unknown> = {}) {
    super(message)
    this.name = 'ApiError'
    this.code = code
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
    throw new ApiError(error.message || text || `HTTP ${response.status}`, error.code || 'api_error', error.details || {})
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
  strategyDetail: (slug: string) => request<StrategyDetail>(`/strategies/${slug}`),
  createDesign: (payload: { name: string; prompt: string; market: string; template?: string | null }) =>
    request<{ strategy: Strategy; design_path: string }>('/designs', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
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
  artifacts: (slug: string) => request<ArtifactList>(`/strategies/${slug}/artifacts`),
  artifact: (slug: string, key: string) =>
    request<ArtifactContent>(`/strategies/${slug}/artifacts/${key}`),
}
