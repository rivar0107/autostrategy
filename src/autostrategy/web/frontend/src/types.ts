export type StrategyStatus =
  | 'draft'
  | 'designed'
  | 'coded'
  | 'backtested'
  | 'paper_running'
  | 'optimized'
  | 'active'
  | 'archived'

export interface Strategy {
  name: string
  slug: string
  description: string
  market: string
  status: StrategyStatus
  template?: string | null
  tags: string[]
}

export interface StrategyDetail {
  strategy: Strategy
  paths: Record<string, string>
}

export interface ArtifactMeta {
  slug: string
  artifact_key: string
  relative_path: string
  path: string
  exists: boolean
  size: number
  modified_at?: string | null
  content_type: string
}

export interface ArtifactList {
  slug: string
  artifacts: ArtifactMeta[]
}

export interface ArtifactContent extends ArtifactMeta {
  content: string
  json?: unknown
}

export interface BacktestResponse {
  strategy: Strategy
  result_path: string
  score: number
  result: Record<string, any>
}

export type BacktestJobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'timed_out' | 'stopped'

export interface BacktestJobResponse {
  job_id: string
  slug: string
  status: BacktestJobStatus
  created_at: string
  started_at?: string | null
  finished_at?: string | null
  result_path?: string | null
  score?: number | null
  error?: string | null
  stop_requested?: boolean
}

export interface PaperRunResponse {
  strategy: Strategy
  result_path: string
  result: Record<string, any>
}

export interface AppInfo {
  version: string
  workspace_root: string
  templates: string[]
  llm_provider: string
  llm_model: string
}

export interface ConfigResponse {
  version: string
  default_market: string
  llm_provider: string
  llm_model: string
  llm_base_url?: string | null
  llm_api_key_env: string
  llm_ready: boolean
  llm_missing_api_key: boolean
  llm_setup_hint?: string | null
  llm_checked_env_vars: string[]
}

export interface LlmConfigUpdate {
  provider: string
  model: string
  base_url?: string | null
  api_key_env: string
}
