import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'
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
  expect(screen.getByText('自然语言生成设计')).toBeInTheDocument()
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

  await user.click(await screen.findByText('自然语言生成设计'))
  await user.type(screen.getByLabelText('策略名称'), 'demo')
  await user.type(screen.getByLabelText('策略想法'), '帮我做一个策略')
  await user.click(screen.getByText('生成设计'))

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

  render(<App />)

  await user.click(await screen.findByText('查看详情'))
  await user.click(await screen.findByText('运行回测'))
  expect(await screen.findByText('回测任务：running')).toBeInTheDocument()
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
          replay: { current_at: '2024-01-02', bars_processed: 1, progress: 1 },
          summary: { paper_return: 1, paper_max_drawdown: 0.5, trade_count: 1, final_value: 1010000 },
          latest_decision: { action: 'buy', reason: 'signal' },
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

  render(<App />)

  await user.click(await screen.findByText('查看详情'))
  expect(await screen.findByText('模拟运行摘要')).toBeInTheDocument()
  await user.click(await screen.findByText('启动模拟运行'))
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

  render(<App />)

  await user.click(await screen.findByText('查看详情'))
  await user.click(await screen.findByText('启动模拟运行'))
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
