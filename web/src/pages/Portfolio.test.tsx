import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import Portfolio from './Portfolio'
import { api } from '../api/client'

vi.mock('../api/client', () => ({
  api: {
    getActualPortfolioState: vi.fn(),
    getActualBucketAllocations: vi.fn(),
    getActualWatchlist: vi.fn(),
    getActualGoals: vi.fn(),
    getActualPerformance: vi.fn(),
    getActualJournal: vi.fn(),
  },
}))

describe('Portfolio Page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getActualPortfolioState).mockResolvedValue({
      last_updated: '2026-07-16T10:00:00Z',
      fx_rates: { USD: 35.0 },
      summary: {
        total_value_thb: 1000000,
        total_cost_basis_thb: 900000,
        total_unrealized_profit: 100000,
        passive_income_ytd: 20000,
      },
      allocation_targets: [
        { bucket_id: 'CORE', name: 'Core Equities', target_percent: 60, color: null },
      ],
      holdings: [
        {
          symbol: 'AAPL',
          asset_type: 'US_STOCK',
          units: 10,
          bucket_id: 'CORE',
          avg_cost_usd: 150,
          avg_cost_thb: 5250,
          current_price_usd: 180,
          current_price_thb: 6300,
          market_value_thb: 63000,
          unrealized_pnl_percent: 20,
          unrealized_pnl_value: 10500,
          market_cap_tier: 'Mega',
          yield_on_cost: 1.5,
          company_name: 'Apple Inc.',
          pe_ratio: 28.5,
          eps: 6.31,
          payout_ratio: 15,
          market_cap_value: 2800000000000,
          dividend_per_share: 0.96,
          dividend_yield: 0.53,
          accumulated_dividend_thb: null,
          fundamentals_updated_at: 1721123456,
        },
      ],
      price_refresh_info: null,
    })

    vi.mocked(api.getActualBucketAllocations).mockResolvedValue({
      warning: null,
      summaries: [
        {
          bucket_id: 'CORE',
          name: 'Core Equities',
          target_percent: 60,
          actual_value_thb: 60000,
          actual_percent: 60,
          variance: 0,
          color: null,
        },
      ],
    })

    vi.mocked(api.getActualWatchlist).mockResolvedValue({
      last_updated: null,
      items: [],
    })

    vi.mocked(api.getActualGoals).mockResolvedValue({
      n_goals: 0,
      goals: [],
      generated_at: null,
    })

    vi.mocked(api.getActualPerformance).mockResolvedValue([])
    vi.mocked(api.getActualJournal).mockResolvedValue([])
  })

  it('renders Actual Portfolio Hub title and fetches initial data', async () => {
    render(
      <MemoryRouter>
        <Portfolio />
      </MemoryRouter>
    )

    expect(screen.getByText('Actual Portfolio Hub')).toBeInTheDocument()

    await waitFor(() => {
      expect(api.getActualPortfolioState).toHaveBeenCalledWith(false, false)
      expect(api.getActualBucketAllocations).toHaveBeenCalled()
    })

    expect(screen.getByText(/Total Portfolio NAV/i)).toBeInTheDocument()
    expect(screen.getByText('Core Equities')).toBeInTheDocument()
  })

  it('switches to Holdings tab when tab control is clicked', async () => {
    render(
      <MemoryRouter>
        <Portfolio />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(api.getActualPortfolioState).toHaveBeenCalled()
    })

    const holdingsTabBtn = screen.getByRole('button', { name: /Holdings/i })
    fireEvent.click(holdingsTabBtn)

    await waitFor(() => {
      expect(screen.getByText('AAPL')).toBeInTheDocument()
      expect(screen.getByText('Apple Inc.')).toBeInTheDocument()
    })
  })

  it('renders Top-Level Executive Dashboard and switches between Goals and Performance', async () => {
    vi.mocked(api.getActualGoals).mockResolvedValue({
      n_goals: 1,
      goals: [
        {
          name: 'Retirement Fund',
          target_amount_thb: 5000000,
          goal_type: 'Retirement',
          current_amount_thb: 1000000,
          progress_pct: 20,
          deadline: '2035-12-31',
          deadline_days_left: 3500,
          notes: 'Test goal',
        },
      ],
      generated_at: '2026-07-16T10:00:00Z',
    })

    vi.mocked(api.getActualPerformance).mockResolvedValue([
      {
        Date: '2026-07-16',
        Total_NAV: 1000000,
        Total_Cost: 900000,
        Unrealized_PnL: 100000,
        Cash_Balance: 50000,
      },
    ])

    render(
      <MemoryRouter>
        <Portfolio />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(screen.getByText(/Executive Dashboard: Goals & Performance/i)).toBeInTheDocument()
    })

    // By default goals tab is active in the executive section
    expect(screen.getByText('Retirement Fund')).toBeInTheDocument()

    // Switch to Performance history inside Executive Dashboard
    const perfBtn = screen.getByRole('button', { name: /Performance History/i })
    fireEvent.click(perfBtn)

    await waitFor(() => {
      expect(screen.getByText(/Performance History \(1 จุดข้อมูล\)/i)).toBeInTheDocument()
      expect(screen.getByText('2026-07-16')).toBeInTheDocument()
    })
  })
})
