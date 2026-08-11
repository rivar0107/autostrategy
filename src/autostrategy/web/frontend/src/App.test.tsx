import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'
import StrategyWorkbench from './StrategyWorkbench'
import type { ConfigResponse } from './types'

const readyConfig: ConfigResponse = {
  version: '0.1.0',
  default_market: 'A股',
  llm_provider: 'openai',
  llm_model: 'gpt-4o-mini',
  llm_base_url: null,
  llm_api_key_env: 'AUTOSTRATEGY_LLM_API_KEY',
  llm_ready: true,
  llm_missing_api_key: false,
  llm_setup_hint: null,
  llm_checked_env_vars: ['AUTOSTRATEGY_LLM_API_KEY', 'OPENAI_API_KEY'],
}

const missingConfig: ConfigResponse = {
  ...readyConfig,
  llm_ready: false,
  llm_missing_api_key: true,
  llm_setup_hint: 'Set AUTOSTRATEGY_LLM_API_KEY in the local shell before starting autostrategy.',
}

function jsonResponse(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), { status })
}

function mockFetch(configResponse: ConfigResponse = readyConfig) {
  return vi.fn(async (url: string, init?: RequestInit) => {
    if (url.endsWith('/config/llm')) {
      return jsonResponse({ ...configResponse, ...(JSON.parse(String(init?.body || '{}'))) })
    }
    if (url.endsWith('/config')) {
      return jsonResponse(configResponse)
    }
    if (url.endsWith('/strategies')) {
      return jsonResponse([])
    }
    if (url.endsWith('/backtest')) {
      return jsonResponse({
        job_id: 'job-1',
        slug: 'demo',
        status: 'queued',
        created_at: '2026-07-05T00:00:00Z',
      }, 202)
    }
    if (url.includes('/backtest-jobs/')) {
      return jsonResponse({
        job_id: 'job-1',
        slug: 'demo',
        status: 'succeeded',
        created_at: '2026-07-05T00:00:00Z',
        started_at: '2026-07-05T00:00:01Z',
        finished_at: '2026-07-05T00:00:02Z',
        score: 80,
      })
    }
    if (url.endsWith('/templates')) {
      return jsonResponse(['dual-ma'])
    }
    return jsonResponse({ status: 'ok' })
  })
}

beforeEach(() => {
  vi.stubGlobal('fetch', mockFetch())
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

test('renders Ant Design dashboard shell', async () => {
  render(<App />)

  expect(screen.getByText('您的本地策略agent工作台')).toBeInTheDocument()
  expect(await screen.findByText('策略列表')).toBeInTheDocument()
  expect(screen.getByText('创建策略')).toBeInTheDocument()
})

test('auto-opens LLM setup without blocking strategy rendering', async () => {
  vi.stubGlobal('fetch', mockFetch(missingConfig))

  render(<App />)

  expect(await screen.findByText('策略列表')).toBeInTheDocument()
  expect(await screen.findByText('需要配置本地 LLM API key')).toBeInTheDocument()
  expect(screen.getByText('autostrategy 不保存 API key')).toBeInTheDocument()
})

test('saves non-secret LLM configuration', async () => {
  const fetchMock = mockFetch(missingConfig)
  vi.stubGlobal('fetch', fetchMock)
  const user = userEvent.setup()

  render(<App />)

  await user.click(await screen.findByText('LLM 设置'))
  await user.clear(screen.getByLabelText('API key 环境变量'))
  await user.type(screen.getByLabelText('API key 环境变量'), 'DEEPSEEK_API_KEY')
  await user.click(screen.getByText('保存 LLM 设置'))

  await waitFor(() => {
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/config/llm', expect.objectContaining({
      method: 'PUT',
      body: expect.stringContaining('DEEPSEEK_API_KEY'),
    }))
  })
  expect(String(fetchMock.mock.calls.find((call) => String(call[0]).endsWith('/config/llm'))?.[1]?.body)).not.toContain('api_key"')
})

test('runtime LLM configuration error reopens setup modal', async () => {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    if (url.endsWith('/config')) {
      return jsonResponse(readyConfig)
    }
    if (url.endsWith('/strategies')) {
      return jsonResponse([])
    }
    if (url.endsWith('/templates')) {
      return jsonResponse(['dual-ma'])
    }
    if (url.endsWith('/designs')) {
      return jsonResponse({
        error: {
          code: 'llm_configuration_required',
          message: 'LLM API key is not configured.',
          details: {
            provider: 'openai',
            api_key_env: 'AUTOSTRATEGY_LLM_API_KEY',
            setup_hint: 'Set AUTOSTRATEGY_LLM_API_KEY in the local shell before starting autostrategy.',
            llm_ready: false,
            checked_env_vars: ['AUTOSTRATEGY_LLM_API_KEY'],
          },
        },
      }, 428)
    }
    return jsonResponse({ status: 'ok' })
  }))
  const user = userEvent.setup()

  render(<App />)

  await user.click(await screen.findByText('创建策略'))
  await user.type(screen.getByLabelText('策略名称'), 'demo')
  await user.type(screen.getByLabelText(/策略想法/), '帮我做一个策略')
  await user.click(screen.getByRole('button', { name: '创 建' }))

  expect(await screen.findByText('需要配置本地 LLM API key')).toBeInTheDocument()
})

test('shows submitted backtest job status', async () => {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    if (url.endsWith('/config')) {
      return jsonResponse(readyConfig)
    }
    if (url.endsWith('/strategies/demo/artifacts')) {
      return jsonResponse({ slug: 'demo', artifacts: [] })
    }
    if (url.endsWith('/strategies/demo/backtest-result')) {
      return jsonResponse({
        strategy: { name: 'demo', slug: 'demo', description: '', market: 'A股', status: 'backtested', tags: [] },
        result_path: '/tmp/backtest_result.json',
        score: 80,
        result: { backtest: { total_trades: 10 } },
      })
    }
    if (url.endsWith('/strategies/demo/backtest')) {
      return jsonResponse({
        job_id: 'job-1',
        slug: 'demo',
        status: 'running',
        created_at: '2026-07-05T00:00:00Z',
      }, 202)
    }
    if (url.endsWith('/strategies/demo/backtest-jobs/job-1')) {
      return jsonResponse({
        job_id: 'job-1',
        slug: 'demo',
        status: 'succeeded',
        created_at: '2026-07-05T00:00:00Z',
        score: 80,
      })
    }
    if (url.endsWith('/strategies/demo')) {
      return jsonResponse({
        strategy: { name: 'demo', slug: 'demo', description: '', market: 'A股', status: 'coded', tags: [] },
        paths: {},
      })
    }
    if (url.endsWith('/strategies')) {
      return jsonResponse([{ name: 'demo', slug: 'demo', description: '', market: 'A股', status: 'coded', tags: [] }])
    }
    if (url.endsWith('/templates')) {
      return jsonResponse(['dual-ma'])
    }
    return jsonResponse({ status: 'ok' })
  }))
  const user = userEvent.setup()

  render(<StrategyWorkbench slug="demo" onBack={() => {}} />)

  const buttons = await screen.findAllByText('运行回测')
  await user.click(buttons[0])
  expect(await screen.findByText('回测任务：running')).toBeInTheDocument()
})

test('displays backtest percentage metrics without multiplying them again', async () => {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    if (url.endsWith('/strategies/demo/artifacts')) {
      return jsonResponse({ slug: 'demo', artifacts: [] })
    }
    if (url.endsWith('/strategies/demo/backtest-result')) {
      return jsonResponse({
        strategy: { name: 'demo', slug: 'demo', description: '', market: 'A股', status: 'backtested', tags: [] },
        result_path: '/tmp/backtest_result.json',
        score: 80,
        result: {
          backtest: {
            annual_return: 12.0,
            max_drawdown: 8.0,
            sharpe: 1.5,
            total_trades: 10,
          },
          research_quality: {
            trade_sample: 'limited',
            has_equity_curve: false,
            has_trade_records: false,
            has_benchmark: false,
            has_out_of_sample: false,
            warnings: [
              '交易样本少于 30 笔，结论仅可作为初步参考。',
              '缺少真实基准序列或基准收益，不能判断超额收益。',
            ],
          },
        },
      })
    }
    if (url.endsWith('/strategies/demo/paper-run-result')) {
      return jsonResponse({ error: { code: 'paper_run_error', message: 'missing', details: {} } }, 404)
    }
    if (url.endsWith('/strategies/demo')) {
      return jsonResponse({
        strategy: { name: 'demo', slug: 'demo', description: '', market: 'A股', status: 'backtested', tags: [] },
        paths: {},
      })
    }
    return jsonResponse({ status: 'ok' })
  }))

  render(<StrategyWorkbench slug="demo" onBack={() => {}} />)

  expect((await screen.findAllByText('12.00%')).length).toBeGreaterThan(0)
  expect(screen.queryByText('1200.00%')).not.toBeInTheDocument()
  expect((await screen.findAllByText('8.00%')).length).toBeGreaterThan(0)
  expect(await screen.findByText('研究质量提示')).toBeInTheDocument()
  expect(screen.getByText('交易样本少于 30 笔，结论仅可作为初步参考。')).toBeInTheDocument()
  expect(screen.getByText('缺少真实基准序列或基准收益，不能判断超额收益。')).toBeInTheDocument()
})

test('designed strategy generates code and immediately advances to backtest', async () => {
  let detailCalls = 0
  let resolveCodegen!: (response: Response) => void
  let resolveRefreshDetail!: (response: Response) => void
  const codegenResponse = new Promise<Response>((resolve) => {
    resolveCodegen = resolve
  })
  const refreshDetailResponse = new Promise<Response>((resolve) => {
    resolveRefreshDetail = resolve
  })
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (url.endsWith('/strategies/demo/artifacts')) {
      return jsonResponse({ slug: 'demo', artifacts: [] })
    }
    if (url.endsWith('/strategies/demo/backtest-result')) {
      return jsonResponse({ error: { code: 'backtest_error', message: 'missing', details: {} } }, 404)
    }
    if (url.endsWith('/strategies/demo/paper-run-result')) {
      return jsonResponse({ error: { code: 'paper_run_error', message: 'missing', details: {} } }, 404)
    }
    if (url.endsWith('/strategies/demo/codegen')) {
      expect(init).toEqual(expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ force: false }),
      }))
      return codegenResponse
    }
    if (url.endsWith('/strategies/demo')) {
      detailCalls += 1
      if (detailCalls > 1) return refreshDetailResponse
      return jsonResponse({
        strategy: { name: 'demo', slug: 'demo', description: '', market: 'A股', status: 'designed', tags: [] },
        paths: {},
      })
    }
    return jsonResponse({ status: 'ok' })
  })
  vi.stubGlobal('fetch', fetchMock)
  const user = userEvent.setup()

  render(<StrategyWorkbench slug="demo" onBack={() => {}} />)

  const codegenTitle = await screen.findByText('下一步：生成策略代码')
  const codegenCard = codegenTitle.closest('.ant-card')
  expect(codegenCard).not.toBeNull()
  const codegenButton = within(codegenCard as HTMLElement).getByRole('button', {
    name: /生成策略代码$/,
  })
  expect(codegenButton).toHaveClass('ant-btn-primary')

  const clickPromise = user.click(codegenButton)
  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
    '/api/v1/strategies/demo/codegen',
    expect.objectContaining({ method: 'POST' }),
  ))
  expect(codegenButton).toHaveClass('ant-btn-loading')

  await act(async () => {
    resolveCodegen(jsonResponse({
      strategy: { name: 'demo', slug: 'demo', description: '', market: 'A股', status: 'coded', tags: [] },
      generated_files: ['strategy.py', 'config.yaml'],
    }))
    await Promise.resolve()
  })

  let backtestTitle: HTMLElement
  try {
    backtestTitle = await screen.findByText('下一步：运行回测', {}, { timeout: 100 })
  } finally {
    await act(async () => {
      resolveRefreshDetail(jsonResponse({
        strategy: { name: 'demo', slug: 'demo', description: '', market: 'A股', status: 'coded', tags: [] },
        paths: {},
      }))
      await clickPromise
    })
  }
  const backtestCard = backtestTitle.closest('.ant-card')
  expect(backtestCard).not.toBeNull()
  expect(within(backtestCard as HTMLElement).getByRole('button', {
    name: /运行回测$/,
  })).toHaveClass('ant-btn-primary')
})

test('code generation failure shows a clear Chinese error', async () => {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    if (url.endsWith('/strategies/demo/artifacts')) {
      return jsonResponse({ slug: 'demo', artifacts: [] })
    }
    if (url.endsWith('/strategies/demo/backtest-result')) {
      return jsonResponse({ error: { code: 'backtest_error', message: 'missing', details: {} } }, 404)
    }
    if (url.endsWith('/strategies/demo/paper-run-result')) {
      return jsonResponse({ error: { code: 'paper_run_error', message: 'missing', details: {} } }, 404)
    }
    if (url.endsWith('/strategies/demo/codegen')) {
      return jsonResponse({
        error: {
          code: 'validation_error',
          message: 'Generated design failed quality check.',
          details: {},
        },
      }, 422)
    }
    if (url.endsWith('/strategies/demo')) {
      return jsonResponse({
        strategy: { name: 'demo', slug: 'demo', description: '', market: 'A股', status: 'designed', tags: [] },
        paths: {},
      })
    }
    return jsonResponse({ status: 'ok' })
  }))
  const user = userEvent.setup()

  render(<StrategyWorkbench slug="demo" onBack={() => {}} />)

  const codegenTitle = await screen.findByText('下一步：生成策略代码')
  const codegenCard = codegenTitle.closest('.ant-card')
  expect(codegenCard).not.toBeNull()
  await user.click(within(codegenCard as HTMLElement).getByRole('button', {
    name: /生成策略代码$/,
  }))

  expect(await screen.findByText('生成策略代码失败：策略内容校验未通过，请检查设计文档后重试。')).toBeInTheDocument()
})

test('shows submitted paper run status', async () => {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    if (url.endsWith('/config')) {
      return jsonResponse(readyConfig)
    }
    if (url.endsWith('/strategies/demo/artifacts')) {
      return jsonResponse({ slug: 'demo', artifacts: [] })
    }
    if (url.endsWith('/strategies/demo/backtest-result')) {
      return jsonResponse({ error: { code: 'backtest_error', message: 'missing', details: {} } }, 400)
    }
    if (url.endsWith('/strategies/demo/paper-run-result')) {
      return jsonResponse({
        strategy: { name: 'demo', slug: 'demo', description: '', market: 'A股', status: 'paper_running', tags: [] },
        result_path: '/tmp/paper_run_result.json',
        result: {
          mode: 'paper_run',
          run_status: 'completed',
          replay: {
            current_at: '2024-01-02',
            bars_processed: 1,
            progress: 1,
            feed: {
              source: 'data/feed.csv',
              bar_count: 2456,
              symbol_count: 5,
              symbols: ['0700.HK', '563300.SH', '588000.SH', '9868_HK', 'TSLA'],
              start: '2024-01-02T00:00:00',
              end: '2025-12-31T00:00:00',
            },
          },
          summary: { paper_return: 1, paper_max_drawdown: 0.5, trade_count: 1, final_value: 1010000 },
          latest_decision: { action: 'buy', reason: 'signal' },
          review: {
            metrics: { total_return: 1, max_drawdown: 0.5, realized_pnl: 10000, turnover: 100000 },
            key_events: [{ type: 'buy', timestamp: '2024-01-02', symbol: 'A', price: 10, size: 100, reason: 'signal' }],
          },
        },
      })
    }
    if (url.endsWith('/strategies/demo/paper-run')) {
      return jsonResponse({
        job_id: 'paper-job-1',
        slug: 'demo',
        status: 'running',
        created_at: '2026-07-05T00:00:00Z',
      }, 202)
    }
    if (url.endsWith('/strategies/demo')) {
      return jsonResponse({
        strategy: { name: 'demo', slug: 'demo', description: '', market: 'A股', status: 'coded', tags: [] },
        paths: {},
      })
    }
    if (url.endsWith('/strategies')) {
      return jsonResponse([{ name: 'demo', slug: 'demo', description: '', market: 'A股', status: 'coded', tags: [] }])
    }
    if (url.endsWith('/templates')) {
      return jsonResponse(['dual-ma'])
    }
    return jsonResponse({ status: 'ok' })
  }))
  const user = userEvent.setup()

  render(<StrategyWorkbench slug="demo" onBack={() => {}} />)

  const tabs = await screen.findAllByText('模拟运行')
  await user.click(tabs[0])
  expect(await screen.findByText('模拟运行状态')).toBeInTheDocument()
  await user.click(screen.getByRole('button', { name: /启动模拟/ }))
  // Switch to overview tab to see the job alert
  await user.click(screen.getByText('概览'))
  expect(await screen.findByText('模拟运行任务：running')).toBeInTheDocument()
})

test('refreshes paper run result while job is running', async () => {
  let resultCalls = 0
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    if (url.endsWith('/config')) {
      return jsonResponse(readyConfig)
    }
    if (url.endsWith('/strategies/demo/artifacts')) {
      return jsonResponse({ slug: 'demo', artifacts: [] })
    }
    if (url.endsWith('/strategies/demo/backtest-result')) {
      return jsonResponse({ error: { code: 'backtest_error', message: 'missing', details: {} } }, 400)
    }
    if (url.endsWith('/strategies/demo/paper-run-result')) {
      resultCalls += 1
      return jsonResponse({
        strategy: { name: 'demo', slug: 'demo', description: '', market: 'A股', status: 'paper_running', tags: [] },
        result_path: '/tmp/paper_run_result.json',
        result: {
          mode: 'paper_run',
          run_status: 'running',
          replay: { current_at: '2024-01-03', bars_processed: 2, progress: 0.5 },
          summary: { paper_return: 0, paper_max_drawdown: 0, trade_count: 1, final_value: 1000000 },
          latest_decision: { action: 'hold', reason: 'waiting' },
        },
      })
    }
    if (url.endsWith('/strategies/demo/paper-run')) {
      return jsonResponse({ job_id: 'paper-job-1', slug: 'demo', status: 'running', created_at: '2026-07-05T00:00:00Z' }, 202)
    }
    if (url.includes('/strategies/demo/paper-run-jobs/')) {
      return jsonResponse({ job_id: 'paper-job-1', slug: 'demo', status: 'running', created_at: '2026-07-05T00:00:00Z' })
    }
    if (url.endsWith('/strategies/demo')) {
      return jsonResponse({ strategy: { name: 'demo', slug: 'demo', description: '', market: 'A股', status: 'coded', tags: [] }, paths: {} })
    }
    if (url.endsWith('/strategies')) {
      return jsonResponse([{ name: 'demo', slug: 'demo', description: '', market: 'A股', status: 'coded', tags: [] }])
    }
    if (url.endsWith('/templates')) {
      return jsonResponse(['dual-ma'])
    }
    return jsonResponse({ status: 'ok' })
  }))
  const user = userEvent.setup()

  render(<StrategyWorkbench slug="demo" onBack={() => {}} />)

  const tabs = await screen.findAllByText('模拟运行')
  await user.click(tabs[0])
  await screen.findByText('模拟运行状态')
  await user.click(screen.getByRole('button', { name: /启动模拟/ }))
  await waitFor(() => expect(resultCalls).toBeGreaterThan(1), { timeout: 1500 })

  expect(await screen.findByText('hold')).toBeInTheDocument()
  expect(screen.getByText('2024-01-03')).toBeInTheDocument()
})

test('ordinary non-LLM failures do not open setup modal', async () => {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    if (url.endsWith('/config')) {
      return jsonResponse(readyConfig)
    }
    if (url.endsWith('/templates')) {
      return jsonResponse(['dual-ma'])
    }
    if (url.endsWith('/strategies')) {
      return jsonResponse({ error: { code: 'api_error', message: 'boom', details: {} } }, 500)
    }
    return jsonResponse({ status: 'ok' })
  }))

  render(<App />)

  expect(await screen.findByText('加载失败')).toBeInTheDocument()
  expect(screen.queryByText('需要配置本地 LLM API key')).not.toBeInTheDocument()
})

test('research workbench shows immutable inputs and requires a reason before acceptance', async () => {
  const session = {
    session_id: 'session-1',
    strategy_slug: 'demo',
    base_version_id: 'version-base',
    manifest_id: 'manifest-1',
    status: 'awaiting_decision',
    diagnostics: [{
      code: 'sample.insufficient',
      category: 'robustness',
      severity: 'warning',
      evidence: { total_trades: 20 },
      hypothesis: '交易样本不足导致评分不稳定',
      suggested_actions: ['延长训练区间'],
      auto_fixable: false,
    }],
    candidates: [{
      candidate_id: 'candidate-1',
      name: '提高 alpha',
      hypothesis: '验证 alpha 参数',
      config_overrides: { alpha: 4 },
      status: 'selected',
      validation_score: 76,
      improvement: 8,
      eligible: true,
      version_id: 'version-candidate',
    }],
    selected_candidate_id: 'candidate-1',
    selected_version_id: 'version-candidate',
    oos_revealed: true,
    oos_passed: true,
    decision: null,
    created_at: '2026-08-10T00:00:00Z',
    updated_at: '2026-08-10T00:10:00Z',
  }
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    if (url.endsWith('/strategies/demo/artifacts')) {
      return jsonResponse({ slug: 'demo', artifacts: [] })
    }
    if (url.endsWith('/strategies/demo/backtest-result') || url.endsWith('/strategies/demo/paper-run-result')) {
      return jsonResponse({ error: { code: 'not_found', message: 'missing', details: {} } }, 404)
    }
    if (url.endsWith('/strategies/demo/versions')) {
      return jsonResponse([
        { version_id: 'version-base', strategy_slug: 'demo', version: 1, content_digest: 'a'.repeat(64), artifact_path: '/tmp/v1', change_summary: 'Baseline', state: 'accepted', created_at: '2026-08-10T00:00:00Z' },
        { version_id: 'version-candidate', strategy_slug: 'demo', version: 2, parent_version_id: 'version-base', content_digest: 'b'.repeat(64), artifact_path: '/tmp/v2', change_summary: 'Candidate', state: 'candidate', created_at: '2026-08-10T00:05:00Z' },
      ])
    }
    if (url.endsWith('/strategies/demo/dataset-manifests')) {
      return jsonResponse([{
        manifest_id: 'manifest-1', strategy_slug: 'demo', version_id: 'version-base',
        data_source: 'fixture', symbols: ['000905.SH'], frequency: 'daily', adjustment: 'forward',
        benchmark: '000905.SH', commission: 0.0003, slippage: 0.001,
        train: { start: '2018-01-01', end: '2021-12-31' },
        validation: { start: '2022-01-01', end: '2023-12-31' },
        test: { start: '2024-01-01', end: '2025-12-31' },
        snapshot_path: '/tmp/data', snapshot_files: { data: 'data.csv' }, output_type: 'dataframe',
        data_digest: 'c'.repeat(64), locked: true, created_at: '2026-08-10T00:00:00Z',
      }])
    }
    if (url.endsWith('/strategies/demo/experiments/session-1/accept')) {
      expect(init).toEqual(expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ reason: '样本外表现稳定' }),
      }))
      return jsonResponse({ ...session, status: 'accepted', decision: 'accepted', decision_reason: '样本外表现稳定', accepted_version_id: 'version-candidate' })
    }
    if (url.endsWith('/strategies/demo/experiments')) {
      return jsonResponse([session])
    }
    if (url.endsWith('/strategies/demo')) {
      return jsonResponse({
        strategy: { name: 'demo', slug: 'demo', description: '', market: 'A股', status: 'backtested', tags: [], current_version_id: 'version-base', active_version_id: 'version-base' },
        paths: {},
      })
    }
    return jsonResponse({ status: 'ok' })
  })
  vi.stubGlobal('fetch', fetchMock)
  const user = userEvent.setup()

  render(<StrategyWorkbench slug="demo" onBack={() => {}} />)

  await user.click(await screen.findByText('研究流程'))
  expect(await screen.findByText('交易样本不足导致评分不稳定')).toBeInTheDocument()
  expect(screen.getByText('2018-01-01 → 2021-12-31')).toBeInTheDocument()
  expect(screen.getByText('提高 alpha')).toBeInTheDocument()
  expect(screen.getByText('样本外验证通过')).toBeInTheDocument()

  await user.click(screen.getByRole('button', { name: '接受候选' }))
  expect(screen.getByText('确认接受候选版本')).toBeInTheDocument()
  const confirm = screen.getByRole('button', { name: '确认接受' })
  expect(confirm).toBeDisabled()
  await user.type(screen.getByPlaceholderText('请输入决策原因'), '样本外表现稳定')
  await user.click(confirm)

  expect((await screen.findAllByText('已接受')).length).toBeGreaterThan(0)
})
