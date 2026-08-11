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
  version?: number
  content_digest?: string
  current_version_id?: string | null
  active_version_id?: string | null
}

export type StrategyVersionState = 'candidate' | 'accepted' | 'rejected'

export interface StrategyVersion {
  version_id: string
  strategy_slug: string
  version: number
  parent_version_id?: string | null
  content_digest: string
  artifact_path: string
  change_summary: string
  state: StrategyVersionState
  created_at: string
}

export interface DateRange {
  start: string
  end: string
}

export interface DatasetManifest {
  manifest_id: string
  strategy_slug: string
  version_id: string
  data_source: string
  symbols: string[]
  frequency: string
  adjustment: string
  benchmark: string
  commission: number
  slippage: number
  train: DateRange
  validation: DateRange
  test: DateRange
  snapshot_path: string
  snapshot_files: Record<string, string>
  output_type: 'dataframe' | 'mapping'
  data_digest: string
  locked: true
  created_at: string
}

export interface DiagnosticFinding {
  code: string
  category: 'data' | 'signal' | 'risk' | 'execution' | 'overfit' | 'leakage' | 'robustness'
  severity: 'info' | 'warning' | 'critical'
  evidence: Record<string, unknown>
  hypothesis: string
  suggested_actions: string[]
  auto_fixable: boolean
}

export interface ExperimentCandidate {
  candidate_id: string
  name: string
  hypothesis: string
  config_overrides: Record<string, unknown>
  status: 'proposed' | 'evaluated' | 'failed' | 'selected'
  train_run_id?: string | null
  validation_run_id?: string | null
  train_score?: number | null
  validation_score?: number | null
  improvement?: number | null
  eligible: boolean
  version_id?: string | null
  error?: string | null
}

export type ExperimentStatus =
  | 'created'
  | 'baseline_completed'
  | 'diagnosed'
  | 'optimized'
  | 'oos_validated'
  | 'awaiting_decision'
  | 'accepted'
  | 'rejected'
  | 'failed'

export interface ExperimentSession {
  session_id: string
  strategy_slug: string
  base_version_id: string
  manifest_id: string
  status: ExperimentStatus
  baseline_train_run_id?: string | null
  baseline_validation_run_id?: string | null
  diagnostics: DiagnosticFinding[]
  candidates: ExperimentCandidate[]
  selected_candidate_id?: string | null
  selected_version_id?: string | null
  oos_base_run_id?: string | null
  oos_candidate_run_id?: string | null
  oos_revealed: boolean
  oos_passed?: boolean | null
  decision?: 'accepted' | 'rejected' | null
  decision_reason?: string | null
  accepted_version_id?: string | null
  error?: string | null
  created_at: string
  updated_at: string
}

export interface DatasetManifestCreate {
  version_id: string
  train: DateRange
  validation: DateRange
  test: DateRange
  benchmark: string
  data_source?: string
  frequency?: string
  adjustment?: string
  commission?: number
  slippage?: number
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

export interface ResearchQuality {
  trade_sample: 'insufficient' | 'limited' | 'adequate'
  has_equity_curve: boolean
  has_trade_records: boolean
  has_benchmark: boolean
  has_out_of_sample: boolean
  warnings: string[]
}

export interface BacktestWorkflowResult extends Record<string, any> {
  backtest: Record<string, any>
  research_quality?: ResearchQuality
}

export interface BacktestResponse {
  strategy: Strategy
  result_path: string
  score: number
  result: BacktestWorkflowResult
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

export type DesignJobStatus = 'queued' | 'running' | 'succeeded' | 'failed'

export interface DesignJobResponse {
  job_id: string
  name: string
  status: DesignJobStatus
  created_at: string
  started_at?: string | null
  finished_at?: string | null
  strategy?: Strategy | null
  design_path?: string | null
  error?: string | null
  error_code?: string | null
}

export interface PaperRunResponse {
  strategy: Strategy
  result_path: string
  result: Record<string, any>
}

export type FtExecutionMode = 'observe' | 'manual' | 'auto'

export interface FtConnectionInput {
  base_url: string
  ft_account: string
  password: string
  password_transform: 'plain' | 'md5_32_lower'
  confirmed_client_version: string
  allowed_simulation_accounts: string[]
  allowed_symbols: string[]
  symbol_mapping: Record<string, string>
  allowed_algorithms: string[]
  external_id_max_length: number
  external_id_scope_confirmed: boolean
}

export interface FtAccount {
  ft_account: string
  ft_account_name: string
  broker_id: string
  broker_name: string
  trade_account: string
  nickname: string
  login_status: boolean
}

export interface FtAlgorithmConfig {
  strategy_type: string
  params: Record<string, string | number | boolean>
  reach_limit_continue: false
  over_time_continue: false
}

export interface ClientSimulationRequest {
  trade_account: string
  execution_mode: FtExecutionMode
  acknowledge_simulation: boolean
  execution_route: 'algorithm_parent'
  execution_window_start: string
  execution_window_end: string
  algorithm: FtAlgorithmConfig
  risk: {
    max_order_pct: number
    max_symbol_position_pct: number
    max_total_position_pct: number
  }
}

export interface PreflightCheck {
  code: string
  passed: boolean
  message: string
  details: Record<string, unknown>
}

export interface FtFundSnapshot {
  trade_account: string
  balance: number
  asset: number
  available: number
  frozen: number
  profit: number
  risk_equity: number
  diagnostics: string[]
  collected_at: string
}

export interface FtPositionSnapshot {
  trade_account: string
  stock_code: string
  total_volume: number
  available_volume: number
  locked_volume: number
  in_transit_volume: number
}

export interface FtMonitoringMetric {
  account_id: string
  trade_account: string
  basket_id?: string | null
  basket_name?: string | null
  plan_buy: number
  plan_sale: number
  trade_buy: number
  trade_sale: number
  buy_rate: number
  sale_rate: number
  exposure: number
  cancel_rate: number
  total_rate: number
  error_rate: number
}

export interface FtMonitoringSnapshot {
  ft_account: string
  trade_accounts: FtMonitoringMetric[]
  baskets: FtMonitoringMetric[]
  diagnostics: string[]
  collected_at: string
}

export interface ClientSimulationPreflight {
  ready: boolean
  checks: PreflightCheck[]
  account?: FtAccount | null
  health?: {
    trade_account: string
    login_status: boolean
    order_engine_status: boolean
  } | null
  funds?: FtFundSnapshot | null
  positions: FtPositionSnapshot[]
  monitoring?: FtMonitoringSnapshot | null
}

export interface ClientSimulationIntent {
  intent_id: string
  intent_key: string
  symbol: string
  broker_symbol: string
  side: 'buy' | 'sell'
  quantity: number
  signal_price?: number | null
  reason: string
  status: string
}

export interface ClientSimulationOrder {
  parent_order_id: string
  external_id: string
  stock_code: string
  order_volume: number
  trade_volume: number
  raw_status: number
  raw_status_msg: string
  normalized_status: string
}

export interface ClientSimulationSession {
  session_id: string
  strategy_slug: string
  strategy_version: number
  status: string
  execution_mode: FtExecutionMode
  execution_route: 'algorithm_parent'
  ft_account: string
  trade_account: string
  broker_id: string
  account_nickname: string
  account_login_status: boolean
  order_engine_status: boolean
  execution_window_start: string
  execution_window_end: string
  last_evaluated_bar_at?: string | null
  client_version: string
  minimum_client_version: string
  preflight: PreflightCheck[]
  funds?: FtFundSnapshot | null
  positions: FtPositionSnapshot[]
  monitoring?: FtMonitoringSnapshot | null
  intents: ClientSimulationIntent[]
  orders: ClientSimulationOrder[]
  latest_error?: string | null
  created_at: string
  updated_at: string
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
