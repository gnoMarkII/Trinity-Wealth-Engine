import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { RunEquityAnalysisModal } from './RunEquityAnalysisModal'
import { api } from '../../api/client'

vi.mock('../../api/client', () => ({
  api: {
    createKanbanCard: vi.fn(),
    dispatchJob: vi.fn(),
    getActualPortfolioState: vi.fn(),
    getActualWatchlist: vi.fn(),
  },
}))

describe('RunEquityAnalysisModal', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getActualPortfolioState).mockResolvedValue({
      last_updated: null,
      fx_rates: {},
      summary: { total_value_thb: 0, total_cost_basis_thb: 0, total_unrealized_profit: 0, passive_income_ytd: 0 },
      allocation_targets: [],
      holdings: [],
      price_refresh_info: null,
    })
    vi.mocked(api.getActualWatchlist).mockResolvedValue({
      last_updated: null,
      items: [],
    })
  })

  it('renders correctly with default values', () => {
    render(<RunEquityAnalysisModal onClose={vi.fn()} />)
    expect(screen.getByText('📊 วิเคราะห์หุ้นและดึงข่าวล่าสุด')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('เช่น AAPL, NVDA, PTT.BK')).toBeInTheDocument()
  })

  it('auto-detects market TH for tickers ending with .BK when manually typed', async () => {
    render(<RunEquityAnalysisModal onClose={vi.fn()} />)
    const input = screen.getByPlaceholderText('เช่น AAPL, NVDA, PTT.BK')

    fireEvent.change(input, { target: { value: 'PTT.BK' } })

    const selects = screen.getAllByRole('combobox') as HTMLSelectElement[]
    const marketSelect = selects.find((s) => s.value === 'TH' || s.value === 'US')
    expect(marketSelect?.value).toBe('TH')
  })

  it('filters asset_type === Stock and displays portfolio & watchlist options', async () => {
    vi.mocked(api.getActualPortfolioState).mockResolvedValueOnce({
      last_updated: null,
      fx_rates: {},
      summary: { total_value_thb: 0, total_cost_basis_thb: 0, total_unrealized_profit: 0, passive_income_ytd: 0 },
      allocation_targets: [],
      holdings: [
        { symbol: 'AAPL', asset_type: 'Stock', units: 10, bucket_id: null, avg_cost_usd: 150, avg_cost_thb: null, current_price_usd: 180, current_price_thb: 6300, market_value_thb: 63000, unrealized_pnl_percent: 20, unrealized_pnl_value: 300, market_cap_tier: null, yield_on_cost: null, company_name: 'Apple Inc.', pe_ratio: null, eps: null, payout_ratio: null, market_cap_value: null, dividend_per_share: null, dividend_yield: null, accumulated_dividend_thb: null, fundamentals_updated_at: null },
        { symbol: 'PTT', asset_type: 'Stock', units: 100, bucket_id: null, avg_cost_usd: null, avg_cost_thb: 34.5, current_price_usd: null, current_price_thb: 35, market_value_thb: 3500, unrealized_pnl_percent: 1.4, unrealized_pnl_value: 50, market_cap_tier: null, yield_on_cost: null, company_name: 'PTT Public Co.', pe_ratio: null, eps: null, payout_ratio: null, market_cap_value: null, dividend_per_share: null, dividend_yield: null, accumulated_dividend_thb: null, fundamentals_updated_at: null },
        { symbol: 'SPY', asset_type: 'ETF', units: 5, bucket_id: null, avg_cost_usd: 400, avg_cost_thb: null, current_price_usd: 450, current_price_thb: 15750, market_value_thb: 78750, unrealized_pnl_percent: 12.5, unrealized_pnl_value: 250, market_cap_tier: null, yield_on_cost: null, company_name: 'SPDR S&P 500', pe_ratio: null, eps: null, payout_ratio: null, market_cap_value: null, dividend_per_share: null, dividend_yield: null, accumulated_dividend_thb: null, fundamentals_updated_at: null },
      ],
      price_refresh_info: null,
    })

    vi.mocked(api.getActualWatchlist).mockResolvedValueOnce({
      last_updated: null,
      items: [
        { symbol: 'NVDA', asset_type: 'Stock', target_price: 120, added_date: '2026-01-01', notes: 'Watch for dips' },
        { symbol: 'BTC', asset_type: 'Crypto', target_price: 60000, added_date: '2026-01-01', notes: 'Bitcoin' },
      ],
    })

    render(<RunEquityAnalysisModal onClose={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: '💼 AAPL' })).toBeInTheDocument()
      expect(screen.getByRole('button', { name: '💼 PTT' })).toBeInTheDocument()
      expect(screen.queryByRole('button', { name: /SPY/ })).not.toBeInTheDocument()
      expect(screen.getByRole('button', { name: '⭐ NVDA' })).toBeInTheDocument()
      expect(screen.queryByRole('button', { name: /BTC/ })).not.toBeInTheDocument()
    })
  })

  it('sets market TH for portfolio holding with avg_cost_thb even if symbol lacks .BK suffix', async () => {
    vi.mocked(api.getActualPortfolioState).mockResolvedValueOnce({
      last_updated: null,
      fx_rates: {},
      summary: { total_value_thb: 0, total_cost_basis_thb: 0, total_unrealized_profit: 0, passive_income_ytd: 0 },
      allocation_targets: [],
      holdings: [
        { symbol: 'CPALL', asset_type: 'Stock', units: 100, bucket_id: null, avg_cost_usd: null, avg_cost_thb: 60, current_price_usd: null, current_price_thb: 65, market_value_thb: 6500, unrealized_pnl_percent: 8.3, unrealized_pnl_value: 500, market_cap_tier: null, yield_on_cost: null, company_name: 'CP ALL', pe_ratio: null, eps: null, payout_ratio: null, market_cap_value: null, dividend_per_share: null, dividend_yield: null, accumulated_dividend_thb: null, fundamentals_updated_at: null },
      ],
      price_refresh_info: null,
    })

    render(<RunEquityAnalysisModal onClose={vi.fn()} />)

    const pill = await screen.findByRole('button', { name: '💼 CPALL' })
    fireEvent.click(pill)

    const input = screen.getByPlaceholderText('เช่น AAPL, NVDA, PTT.BK') as HTMLInputElement
    expect(input.value).toBe('CPALL')

    const selects = screen.getAllByRole('combobox') as HTMLSelectElement[]
    const marketSelect = selects.find((s) => s.value === 'TH' || s.value === 'US')
    expect(marketSelect?.value).toBe('TH')
  })

  it('handles fail-soft loading gracefully when portfolio or watchlist API fails', async () => {
    vi.mocked(api.getActualPortfolioState).mockRejectedValueOnce(new Error('Network error'))
    vi.mocked(api.getActualWatchlist).mockRejectedValueOnce(new Error('Network error'))

    render(<RunEquityAnalysisModal onClose={vi.fn()} />)

    await waitFor(() => {
      expect(screen.getByPlaceholderText('เช่น AAPL, NVDA, PTT.BK')).toBeInTheDocument()
    })
    // Modal still functions
    const input = screen.getByPlaceholderText('เช่น AAPL, NVDA, PTT.BK')
    fireEvent.change(input, { target: { value: 'MSFT' } })
    expect((input as HTMLInputElement).value).toBe('MSFT')
  })

  it('submits form, calls createKanbanCard and dispatchJob, then triggers onDispatched and onClose', async () => {
    const handleClose = vi.fn()
    const handleDispatched = vi.fn()

    vi.mocked(api.createKanbanCard).mockResolvedValue({
      created: true,
      card: {
        card_id: 'card-123',
        title: 'วิเคราะห์หุ้น AAPL (US)',
        prompt: 'วิเคราะห์หุ้น AAPL (US) และดึงข่าวล่าสุดพร้อมประเมิน Valuation',
        column_name: 'backlog',
        job_id: null,
        flow: 'manager',
        scope: 'both',
        display_seq: 1,
        discord_notify: true,
        is_verified: true,
        created_at: 1000,
        updated_at: 1000,
      },
    })

    vi.mocked(api.dispatchJob).mockResolvedValue({
      job_id: 'job-456',
      status: 'running',
      card_id: 'card-123',
      error_message: null,
      current_node: 'supervisor',
      interrupt_payload: null,
      log_count: 0,
      created_at: 1000,
      updated_at: 1000,
    })

    render(<RunEquityAnalysisModal onClose={handleClose} onDispatched={handleDispatched} />)

    const input = screen.getByPlaceholderText('เช่น AAPL, NVDA, PTT.BK')
    fireEvent.change(input, { target: { value: 'aapl' } })

    const submitBtn = screen.getByRole('button', { name: '📋 สร้างการ์ด (รอกดยอมรับก่อนรัน)' })
    fireEvent.click(submitBtn)

    await waitFor(() => {
      expect(api.createKanbanCard).toHaveBeenCalledWith(
        'วิเคราะห์หุ้น AAPL (US)',
        'manager',
        'วิเคราะห์หุ้น AAPL (US) และดึงข่าวล่าสุดพร้อมประเมิน Valuation',
        'both'
      )
      expect(api.dispatchJob).toHaveBeenCalledWith(
        'วิเคราะห์หุ้น AAPL (US) และดึงข่าวล่าสุดพร้อมประเมิน Valuation',
        'card-123',
        'manager',
        'both',
        true
      )
      expect(handleDispatched).toHaveBeenCalledWith('job-456', 'card-123', 'AAPL')
      expect(handleClose).toHaveBeenCalled()
    })
  })
})

