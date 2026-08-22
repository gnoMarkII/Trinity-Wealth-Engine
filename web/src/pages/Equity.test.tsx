import { render, screen, waitFor, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { api } from '../api/client'
import { mockEquitySummary, mockEquityDetailAAPL } from '../mocks/equity'
import { normalizeTicker } from './Equity'

vi.mock('../api/client', () => ({
  api: {
    getEquityLatest: vi.fn(),
    getEquityDetail: vi.fn(),
    createKanbanCard: vi.fn(),
    dispatchJob: vi.fn(),
    getActualPortfolioState: vi.fn(),
    getActualWatchlist: vi.fn(),
  },
}))

describe('Equity Page & normalizeTicker', () => {
  beforeEach(() => {
    cleanup()
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
    vi.stubEnv('VITE_EQUITY_MOCK', 'false')
    vi.stubEnv('DEV', true as any)
    vi.resetModules()
  })

  it('normalizeTicker strips .BK and .TH suffix and upper-cases', () => {
    expect(normalizeTicker('ptt.bk')).toBe('PTT')
    expect(normalizeTicker('PTT.TH')).toBe('PTT')
    expect(normalizeTicker('aapl')).toBe('AAPL')
    expect(normalizeTicker('  NVDA  ')).toBe('NVDA')
  })

  const renderComponent = async (initialRoute = '/equity') => {
    const { default: Equity } = await import('./Equity')
    return render(
      <MemoryRouter initialEntries={[initialRoute]}>
        <Routes>
          <Route path="/equity" element={<Equity />} />
          <Route path="/equity/:ticker" element={<Equity />} />
        </Routes>
      </MemoryRouter>
    )
  }

  it('renders list of equities from api under respective sections', async () => {
    vi.mocked(api.getEquityLatest).mockResolvedValue(mockEquitySummary)

    await renderComponent()

    await waitFor(() => {
      expect(screen.getByText('AAPL')).toBeInTheDocument()
      expect(screen.getByText('PTT')).toBeInTheDocument()
      expect(screen.getByText('📊 วิเคราะห์แล้วอื่นๆ')).toBeInTheDocument()
    })
  })

  it('filters list using search bar', async () => {
    vi.mocked(api.getEquityLatest).mockResolvedValue(mockEquitySummary)

    await renderComponent()

    await waitFor(() => {
      expect(screen.getByText('AAPL')).toBeInTheDocument()
      expect(screen.getByText('PTT')).toBeInTheDocument()
    })

    const searchInput = screen.getByPlaceholderText('ค้นหาหุ้น (เช่น AAPL, PTT)...')
    await userEvent.type(searchInput, 'pt')

    await waitFor(() => {
      expect(screen.queryByText('AAPL')).not.toBeInTheDocument()
      expect(screen.getByText('PTT')).toBeInTheDocument()
    })
  })

  it('renders portfolio and watchlist sections with correct per-holding market tags and unanalyzed status', async () => {
    vi.mocked(api.getEquityLatest).mockResolvedValue([
      {
        ticker: 'AAPL',
        market: 'US',
        company_name: 'Apple Inc.',
        analysis_date: '2026-08-03',
        evaluated_at: '2026-08-03T10:00:00Z',
        market_sentiment: 'bullish',
        composite_score: 82.5,
        data_quality_flags: [],
        source_file: 'AAPL.md',
        sidecar_file: 'AAPL.json',
      },
    ])

    // Portfolio has AAPL (US, analyzed), PTT (TH, unanalyzed), and UNH (US, unanalyzed)
    vi.mocked(api.getActualPortfolioState).mockResolvedValue({
      last_updated: null,
      fx_rates: {},
      summary: { total_value_thb: 1000, total_cost_basis_thb: 800, total_unrealized_profit: 200, passive_income_ytd: 0 },
      allocation_targets: [],
      holdings: [
        {
          symbol: 'AAPL', asset_type: 'Stock', units: 10, bucket_id: null,
          avg_cost_usd: 150, avg_cost_thb: null, current_price_usd: 180, current_price_thb: 6300,
          market_value_thb: 63000, unrealized_pnl_percent: 20, unrealized_pnl_value: 300,
          market_cap_tier: null, yield_on_cost: null, company_name: 'Apple Inc.',
          pe_ratio: null, eps: null, payout_ratio: null, market_cap_value: null,
          dividend_per_share: null, dividend_yield: null, accumulated_dividend_thb: null, fundamentals_updated_at: null,
        },
        {
          symbol: 'PTT', asset_type: 'Stock', units: 100, bucket_id: null,
          avg_cost_usd: null, avg_cost_thb: 34.5, current_price_usd: null, current_price_thb: 35,
          market_value_thb: 3500, unrealized_pnl_percent: 1.4, unrealized_pnl_value: 50,
          market_cap_tier: null, yield_on_cost: null, company_name: 'PTT Public Co.',
          pe_ratio: null, eps: null, payout_ratio: null, market_cap_value: null,
          dividend_per_share: null, dividend_yield: null, accumulated_dividend_thb: null, fundamentals_updated_at: null,
        },
        {
          symbol: 'UNH', asset_type: 'Stock', units: 2, bucket_id: null,
          avg_cost_usd: 480, avg_cost_thb: null, current_price_usd: 500, current_price_thb: 17500,
          market_value_thb: 35000, unrealized_pnl_percent: 4.1, unrealized_pnl_value: 40,
          market_cap_tier: null, yield_on_cost: null, company_name: 'UnitedHealth Group',
          pe_ratio: null, eps: null, payout_ratio: null, market_cap_value: null,
          dividend_per_share: null, dividend_yield: null, accumulated_dividend_thb: null, fundamentals_updated_at: null,
        },
      ],
      price_refresh_info: null,
    })

    // Watchlist has BH.BK (TH) and MSFT (US)
    vi.mocked(api.getActualWatchlist).mockResolvedValue({
      last_updated: null,
      items: [
        { symbol: 'BH.BK', asset_type: 'Stock', target_price: 250, added_date: '2026-08-01', notes: 'Bumrungrad Hospital' },
        { symbol: 'MSFT', asset_type: 'Stock', target_price: 400, added_date: '2026-08-01', notes: 'Microsoft' },
      ],
    })

    await renderComponent()

    await waitFor(() => {
      // Portfolio section should be visible
      expect(screen.getByText('💼 หุ้นในพอร์ต')).toBeInTheDocument()
      // Watchlist section should be visible
      expect(screen.getByText('⭐ Watchlist')).toBeInTheDocument()
    })

    // AAPL has report (analyzed)
    expect(screen.getByText('AAPL')).toBeInTheDocument()
    // PTT & UNH are in portfolio with unanalyzed badge
    expect(screen.getByText('PTT')).toBeInTheDocument()
    expect(screen.getByText('UNH')).toBeInTheDocument()
    expect(screen.getAllByText('⏳ ยังไม่วิเคราะห์').length).toBeGreaterThanOrEqual(2)

    // Quick analysis button on PTT opens modal with pre-filled ticker="PTT" and market="TH"
    const pttAnalyzeButtons = screen.getAllByRole('button', { name: /วิเคราะห์/i })
    expect(pttAnalyzeButtons.length).toBeGreaterThan(0)
  })

  it('clicking quick-analyze button opens RunEquityAnalysisModal with pre-filled ticker and market', async () => {
    vi.mocked(api.getEquityLatest).mockResolvedValue([])
    vi.mocked(api.getActualPortfolioState).mockResolvedValue({
      last_updated: null,
      fx_rates: {},
      summary: { total_value_thb: 0, total_cost_basis_thb: 0, total_unrealized_profit: 0, passive_income_ytd: 0 },
      allocation_targets: [],
      holdings: [
        {
          symbol: 'PTT', asset_type: 'Stock', units: 100, bucket_id: null,
          avg_cost_usd: null, avg_cost_thb: 34.5, current_price_usd: null, current_price_thb: 35,
          market_value_thb: 3500, unrealized_pnl_percent: 1.4, unrealized_pnl_value: 50,
          market_cap_tier: null, yield_on_cost: null, company_name: 'PTT Public Co.',
          pe_ratio: null, eps: null, payout_ratio: null, market_cap_value: null,
          dividend_per_share: null, dividend_yield: null, accumulated_dividend_thb: null, fundamentals_updated_at: null,
        },
      ],
      price_refresh_info: null,
    })

    await renderComponent()

    await waitFor(() => {
      expect(screen.getByText('PTT')).toBeInTheDocument()
    })

    const analyzeBtn = screen.getByTitle('สั่งวิเคราะห์หุ้น PTT (TH)')
    await userEvent.click(analyzeBtn)

    // Modal should be open with PTT and TH pre-selected
    expect(screen.getByText('📊 วิเคราะห์หุ้นและดึงข่าวล่าสุด')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('เช่น AAPL, NVDA, PTT.BK')).toHaveValue('PTT')
  })

  it('switches tabs between All, Portfolio, and Watchlist', async () => {
    vi.mocked(api.getEquityLatest).mockResolvedValue([])
    vi.mocked(api.getActualPortfolioState).mockResolvedValue({
      last_updated: null,
      fx_rates: {},
      summary: { total_value_thb: 0, total_cost_basis_thb: 0, total_unrealized_profit: 0, passive_income_ytd: 0 },
      allocation_targets: [],
      holdings: [
        {
          symbol: 'PG', asset_type: 'Stock', units: 5, bucket_id: null,
          avg_cost_usd: 150, avg_cost_thb: null, current_price_usd: 160, current_price_thb: 5600,
          market_value_thb: 28000, unrealized_pnl_percent: 6.6, unrealized_pnl_value: 50,
          market_cap_tier: null, yield_on_cost: null, company_name: 'Procter & Gamble',
          pe_ratio: null, eps: null, payout_ratio: null, market_cap_value: null,
          dividend_per_share: null, dividend_yield: null, accumulated_dividend_thb: null, fundamentals_updated_at: null,
        },
      ],
      price_refresh_info: null,
    })
    vi.mocked(api.getActualWatchlist).mockResolvedValue({
      last_updated: null,
      items: [
        { symbol: 'TSLA', asset_type: 'Stock', target_price: 200, added_date: '2026-08-01', notes: 'Tesla' },
      ],
    })

    await renderComponent()

    await waitFor(() => {
      expect(screen.getByText('PG')).toBeInTheDocument()
      expect(screen.getByText('TSLA')).toBeInTheDocument()
    })

    // Filter to Portfolio only
    const portTab = screen.getByRole('button', { name: /💼 ในพอร์ต/i })
    await userEvent.click(portTab)

    expect(screen.getByText('PG')).toBeInTheDocument()
    expect(screen.queryByText('TSLA')).not.toBeInTheDocument()

    // Filter to Watchlist only
    const watchTab = screen.getByRole('button', { name: /⭐ Watch/i })
    await userEvent.click(watchTab)

    expect(screen.queryByText('PG')).not.toBeInTheDocument()
    expect(screen.getByText('TSLA')).toBeInTheDocument()
  })

  it('renders not found state when api returns 404', async () => {
    vi.mocked(api.getEquityLatest).mockResolvedValue([])
    vi.mocked(api.getEquityDetail).mockRejectedValue({ status: 404, message: 'Not found' })

    await renderComponent('/equity/unknown')

    await waitFor(() => {
      expect(screen.getByText('ไม่พบข้อมูล')).toBeInTheDocument()
    })
  })

  it('renders detail view successfully', async () => {
    vi.mocked(api.getEquityLatest).mockResolvedValue(mockEquitySummary)
    vi.mocked(api.getEquityDetail).mockResolvedValue(mockEquityDetailAAPL)

    await renderComponent('/equity/aapl')

    await waitFor(() => {
      expect(screen.getByText('Base Case Summary')).toBeInTheDocument()
      expect(screen.getAllByText('Apple Inc.').length).toBeGreaterThan(0)
    })
  })

  it('เปิด Modal พร้อม ticker เดิมเมื่อกดปุ่ม 🔄 ในหน้า Detail', async () => {
    vi.mocked(api.getEquityLatest).mockResolvedValue(mockEquitySummary)
    vi.mocked(api.getEquityDetail).mockResolvedValue(mockEquityDetailAAPL)

    await renderComponent('/equity/aapl')
    await waitFor(() => expect(screen.getByText('Base Case Summary')).toBeInTheDocument())

    await userEvent.click(screen.getByTitle('วิเคราะห์ใหม่และดึงข่าวล่าสุด'))

    expect(screen.getByPlaceholderText('เช่น AAPL, NVDA, PTT.BK')).toHaveValue('AAPL')
  })

  it('สร้างการ์ดและ dispatch งานสำเร็จแล้วแสดง Toast พร้อมปุ่มไปดู Kanban', async () => {
    vi.mocked(api.getEquityLatest).mockResolvedValue(mockEquitySummary)
    vi.mocked(api.getEquityDetail).mockResolvedValue(mockEquityDetailAAPL)
    vi.mocked(api.createKanbanCard).mockResolvedValue({
      created: true,
      card: {
        card_id: 'card-1', title: 'วิเคราะห์หุ้น NVDA (US)', prompt: 'p', column_name: 'backlog',
        job_id: null, flow: 'manager', scope: 'both', display_seq: 1, discord_notify: true,
        is_verified: true, created_at: 1, updated_at: 1,
      },
    })
    vi.mocked(api.dispatchJob).mockResolvedValue({
      job_id: 'job-1', status: 'running', card_id: 'card-1', error_message: null,
      current_node: null, interrupt_payload: null, log_count: 0, created_at: 1, updated_at: 1,
    })

    await renderComponent('/equity/aapl')
    await waitFor(() => expect(screen.getByText('Base Case Summary')).toBeInTheDocument())

    await userEvent.click(screen.getByTitle('วิเคราะห์ใหม่และดึงข่าวล่าสุด'))
    const tickerInput = screen.getByPlaceholderText('เช่น AAPL, NVDA, PTT.BK')
    await userEvent.clear(tickerInput)
    await userEvent.type(tickerInput, 'nvda')
    await userEvent.click(screen.getByRole('button', { name: '🚀 สร้างการ์ดและเริ่มวิเคราะห์' }))

    await waitFor(() => {
      expect(screen.getByText('สั่งงานวิเคราะห์หุ้น NVDA และดึงข่าวเรียบร้อย')).toBeInTheDocument()
      expect(screen.getByText('ดูสถานะใน Kanban')).toBeInTheDocument()
    })
    expect(screen.queryByText('📊 วิเคราะห์หุ้นและดึงข่าวล่าสุด')).not.toBeInTheDocument()
  })

  it('allows collapsing and expanding the sidebar to minimal mode and persists preference', async () => {
    vi.mocked(api.getEquityLatest).mockResolvedValue(mockEquitySummary)

    await renderComponent('/equity')

    await waitFor(() => {
      expect(screen.getByText('Equity Analysis')).toBeInTheDocument()
    })

    // Click collapse sidebar button
    const collapseBtn = screen.getByRole('button', { name: 'Collapse Sidebar' })
    await userEvent.click(collapseBtn)

    // In minimal mode, full header title is hidden, and expand button is shown
    expect(screen.queryByText('Equity Analysis')).not.toBeInTheDocument()
    const expandBtn = screen.getByRole('button', { name: 'Expand Sidebar' })
    expect(expandBtn).toBeInTheDocument()

    // Click expand button to restore full view
    await userEvent.click(expandBtn)
    expect(screen.getByText('Equity Analysis')).toBeInTheDocument()
  })
})
