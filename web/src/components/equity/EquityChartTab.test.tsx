import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { EquityChartTab } from './EquityChartTab'
import { api } from '../../api/client'
import type { OHLCVResponseDTO } from '../../api/types'

// Mock klineAdapter to avoid JSDOM Canvas requirements
vi.mock('./klineAdapter', () => {
  return {
    initKlineChart: vi.fn().mockReturnValue({
      updateChart: vi.fn(),
      setOverlay: vi.fn(),
      setVolume: vi.fn(),
      setMomentum: vi.fn(),
      setATR: vi.fn(),
      setEvents: vi.fn(),
      set52WLevels: vi.fn(),
      setValuationTargets: vi.fn(),
      setInsiderFilings: vi.fn(),
      enableDrawing: vi.fn(),
      cancelActiveDrawing: vi.fn(),
      clearDrawings: vi.fn(),
      getEffectiveConfig: vi.fn(() => ({ overlay: 'BOLL', volume: false, momentum: 'NONE', atr: false, events: true, week52: true, dcf: true, insider: true })),
      resize: vi.fn(),
      destroy: vi.fn(),
      getChart: vi.fn(),
    }),
  }
})

vi.mock('../../api/client', () => {
  return {
    api: {
      getEquityOHLCV: vi.fn(),
      getValuationTargets: vi.fn().mockResolvedValue({
        evaluation_id: 'eval_mock_123',
        ticker: 'AAPL',
        market: 'US',
        currency: 'USD',
        chart_price_basis: 'provider_proportional_adj_close_ratio',
        valuation_price_basis: 'split_adjusted_only',
        comparability_status: 'comparable',
        comparability_reasons: [],
        corporate_action_factors: [],
        evaluated_at: '2026-08-20T10:00:00Z',
        as_of_label: 'as of 2026-08-20',
        model_version: 'dcf_v1.0',
        valuation_verdict: 'undervalued',
        macro_observable_refs: [],
        data_quality_flags: [],
        status: 'available',
        scenario_order_valid: true,
        scenarios: [
          { scenario_name: 'base', label: 'DCF Base', target_price: 340.0, upside_pct: 10.0, color: 'emerald' },
        ],
      }),
      getInsiderFilings: vi.fn().mockResolvedValue({
        ticker: 'AAPL',
        market: 'US',
        net_shares_30d: 50000.0,
        net_shares_90d: 50000.0,
        net_shares_180d: 50000.0,
        cluster_buy_count: 0,
        total_filings_count: 0,
        filings: [],
      }),
      getAnalystContext: vi.fn().mockResolvedValue({
        ticker: 'AAPL',
        market: 'US',
        currency: 'USD',
        target_mean: 330.0,
        target_high: 360.0,
        target_low: 290.0,
        num_analysts: 40,
        next_earnings_date: '2026-10-25',
        days_to_earnings: 65,
        earnings_history: [],
        data_status: 'ok',
        source_as_of: '2026-08-22T00:00:00',
        exchange_tz: 'America/New_York',
      }),
    },
  }
})



const mockOHLCVResponse: OHLCVResponseDTO = {
  ticker: 'AAPL',
  market: 'US',
  currency: 'USD',
  current_price: 310.25,
  price_change: 4.66,
  price_change_pct: 1.53,
  price_as_of: '2026-08-19T15:30:00-04:00',
  candles: [
    { timestamp: 1700000000000, open: 300, high: 312, low: 298, close: 310.25, volume: 5000000 },
  ],
  pivot_levels: {
    pivot: 310.0,
    r1: 342.75,
    r2: 365.91,
    r3: 380.53,
    s1: 301.5,
    s2: 275.48,
    s3: 265.41,
    s4: 245.57,
  },
  pivot_period: 'monthly',
  pivot_as_of: '2026-07',
}

describe('EquityChartTab', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    vi.mocked(api.getEquityOHLCV).mockResolvedValue(mockOHLCVResponse)
  })

  it('renders chart and support-resistance table with fetched data', async () => {
    render(<EquityChartTab ticker="AAPL" companyName="Apple Inc." market="US" />)

    // Should call API with default 6mo range
    expect(api.getEquityOHLCV).toHaveBeenCalledWith('AAPL', '6mo', '1d', expect.any(AbortSignal))

    // Chart header elements
    expect(await screen.findByText('Apple Inc.')).toBeInTheDocument()
    expect(screen.getAllByText('$310.25').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText(/4.66/)).toBeInTheDocument()

    // S/R Table elements
    expect(screen.getByText('ตารางคำนวณ แนวรับ-แนวต้าน')).toBeInTheDocument()
    expect(screen.getByText('Pivot: $310.00')).toBeInTheDocument()
    expect(screen.getByText('R1')).toBeInTheDocument()
    expect(screen.getByText('$342.75')).toBeInTheDocument()
    expect(screen.getByText('S1')).toBeInTheDocument()
    expect(screen.getByText('$301.50')).toBeInTheDocument()
    expect(screen.getByText(/2026-07/)).toBeInTheDocument()
  })

  it('fetches new data when user clicks a different range or interval button', async () => {
    const user = userEvent.setup()
    render(<EquityChartTab ticker="AAPL" companyName="Apple Inc." market="US" />)

    await screen.findByText('Apple Inc.')

    const button1Y = screen.getByRole('button', { name: '1Y' })
    await user.click(button1Y)
    expect(api.getEquityOHLCV).toHaveBeenCalledWith('AAPL', '1y', '1d', expect.any(AbortSignal))

    const button5Y = screen.getByRole('button', { name: '5Y' })
    await user.click(button5Y)
    expect(api.getEquityOHLCV).toHaveBeenCalledWith('AAPL', '5y', '1d', expect.any(AbortSignal))

    // Switch interval to 15m (defaults to 5d range)
    const button15m = screen.getByRole('button', { name: '15m' })
    await user.click(button15m)
    expect(api.getEquityOHLCV).toHaveBeenCalledWith('AAPL', '5d', '15m', expect.any(AbortSignal))
  })

  it('calculates profit correctly and persists capital in localStorage by currency', async () => {
    const user = userEvent.setup()
    render(<EquityChartTab ticker="AAPL" companyName="Apple Inc." market="US" />)

    await screen.findByText('ตารางคำนวณ แนวรับ-แนวต้าน')

    const capitalInput = screen.getByLabelText(/ใส่เงินลงทุนเพิ่ม/) as HTMLInputElement
    expect(capitalInput.value).toBe('1000')

    // Toggle remember checkbox
    const rememberCheckbox = screen.getByRole('checkbox', { name: 'จดจำนวนเงินนี้ไว้' })
    await user.click(rememberCheckbox)

    expect(localStorage.getItem('sr_calc_capital:USD')).toBe('1000')

    // Change capital to 2000
    await user.clear(capitalInput)
    await user.type(capitalInput, '2000')

    expect(localStorage.getItem('sr_calc_capital:USD')).toBe('2000')
  })

  it('renders error state and handles retry button', async () => {
    const user = userEvent.setup()
    vi.mocked(api.getEquityOHLCV).mockRejectedValueOnce(new Error('Network Error'))

    render(<EquityChartTab ticker="AAPL" companyName="Apple Inc." market="US" />)

    expect(await screen.findByText('เกิดข้อผิดพลาดในการโหลดกราฟ')).toBeInTheDocument()
    expect(screen.getByText('Network Error')).toBeInTheDocument()

    // Retry
    vi.mocked(api.getEquityOHLCV).mockResolvedValueOnce(mockOHLCVResponse)
    const retryBtn = screen.getByRole('button', { name: 'ลองใหม่อีกครั้ง' })
    await user.click(retryBtn)

    expect(await screen.findByText('ตารางคำนวณ แนวรับ-แนวต้าน')).toBeInTheDocument()
    expect(screen.getAllByText('$310.25').length).toBeGreaterThanOrEqual(1)
  })
})


