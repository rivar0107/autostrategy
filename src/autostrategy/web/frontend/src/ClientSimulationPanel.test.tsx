import { render, screen } from '@testing-library/react'
import { vi } from 'vitest'

import ClientSimulationPanel, {
  currentStatus,
  parseAllowedSymbols,
  parseSymbolMapping,
} from './ClientSimulationPanel'

vi.mock('./api/client', () => ({
  api: {
    checkFtClientConnection: vi.fn(),
    ftClientAccounts: vi.fn(),
    preflightClientSimulation: vi.fn(),
    createClientSimulationSession: vi.fn(),
    clientSimulationSession: vi.fn(),
    pauseClientSimulationSession: vi.fn(),
    resumeClientSimulationSession: vi.fn(),
    stopClientSimulationSession: vi.fn(),
    approveClientSimulationIntent: vi.fn(),
    rejectClientSimulationIntent: vi.fn(),
  },
  ApiError: class ApiError extends Error {},
}))

describe('ClientSimulationPanel', () => {
  it('prefers a current false session status over an older successful preflight', () => {
    expect(currentStatus(false, true)).toBe(false)
  })

  it('exposes every customer-entered FT connection and execution field', () => {
    render(<ClientSimulationPanel slug="grid-demo" />)

    expect(screen.getByText('非凸客户端模拟账户')).toBeInTheDocument()
    expect(screen.getByLabelText('客户端地址')).toBeInTheDocument()
    expect(screen.getByLabelText('非凸账号')).toBeInTheDocument()
    expect(screen.getByLabelText('密码')).toHaveAttribute('type', 'password')
    expect(screen.getByLabelText('密码处理')).toBeInTheDocument()
    expect(screen.getByLabelText('客户端版本')).toBeInTheDocument()
    expect(screen.getByLabelText('模拟交易账户白名单')).toBeInTheDocument()
    expect(screen.getByLabelText('本次策略允许标的')).toBeInTheDocument()
    expect(screen.getByLabelText('客户端代码映射')).toBeInTheDocument()
    expect(screen.queryByLabelText('588000.SH 客户端代码')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('563300.SH 客户端代码')).not.toBeInTheDocument()
    expect(screen.getByLabelText('external_id 最大长度')).toBeInTheDocument()
    expect(screen.getByRole('radiogroup', { name: '执行模式' })).toBeInTheDocument()
    expect(screen.getByLabelText('算法类型')).toBeInTheDocument()
    expect(screen.getByLabelText('算法参数')).toBeInTheDocument()
    expect(screen.getByLabelText('开始时间')).toBeInTheDocument()
    expect(screen.getByLabelText('结束时间')).toBeInTheDocument()
    expect(screen.getByLabelText('单笔资产上限')).toBeInTheDocument()
    expect(screen.getByLabelText('单标的仓位上限')).toBeInTheDocument()
    expect(screen.getByLabelText('总仓位上限')).toBeInTheDocument()
    expect(screen.getByText('账户登录状态')).toBeInTheDocument()
    expect(screen.getByText('交易引擎状态')).toBeInTheDocument()
    expect(screen.getByText('最近 10 分钟 K 线')).toBeInTheDocument()
    expect(screen.getByText(/仅在沪深市场形成共同的已完成 10 分钟 K 线后评估策略/)).toBeInTheDocument()
  })

  it('keeps the FT session start disabled before connection and preflight', () => {
    render(<ClientSimulationPanel slug="grid-demo" />)

    expect(screen.getByRole('button', { name: '启动非凸模拟盘' })).toBeDisabled()
    expect(screen.getByText('密码仅在本机内存中使用，不写入配置、日志或会话文件')).toBeInTheDocument()
  })

  it('normalizes the execution universe and requires an exact mapping', () => {
    const symbols = parseAllowedSymbols('600519.sh, 510500.SH\n600519.SH')

    expect(symbols).toEqual(['600519.SH', '510500.SH'])
    expect(parseSymbolMapping(
      '{"600519.SH":"600519.SH","510500.SH":"510500.SH"}',
      symbols,
    )).toEqual({ '600519.SH': '600519.SH', '510500.SH': '510500.SH' })
    expect(() => parseSymbolMapping('{"600519.SH":"600519.SH"}', symbols)).toThrow(
      '必须与本次策略允许标的完全一致',
    )
  })
})
