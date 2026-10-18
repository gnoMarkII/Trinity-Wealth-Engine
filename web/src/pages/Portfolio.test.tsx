import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { MemoryRouter, useNavigate, useSearchParams } from 'react-router-dom'
import Portfolio from './Portfolio'
import { api } from '../api/client'

function CurrentQuery() {
  const [searchParams] = useSearchParams()
  return <output data-testid="current-query">{searchParams.toString()}</output>
}

function RouteHistoryControls() {
  const navigate = useNavigate()
  return (
    <div>
      <button type="button" onClick={() => navigate(-1)}>Route back</button>
      <button type="button" onClick={() => navigate(1)}>Route forward</button>
    </div>
  )
}

vi.mock('../api/client', () => ({
  api: {
    getTransactions: vi.fn(),
    getActualPortfolioState: vi.fn(),
    getActualBucketAllocations: vi.fn(),
    getActualWatchlist: vi.fn(),
    getActualGoals: vi.fn(),
    getActualPerformance: vi.fn(),
    getActualJournal: vi.fn(),
    listPortfolios: vi.fn(),
    removeHolding: vi.fn(),
  },
}))

describe('Portfolio Page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.listPortfolios).mockResolvedValue([
      { id: 'default', name: 'พอร์ตลงทุนหลัก', is_default: true, created_at: '2026-07-16T10:00:00Z' },
    ])
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
    vi.mocked(api.getTransactions).mockResolvedValue({
      portfolio_id: 'default',
      transactions: [],
      summary: { total_buy_count: 0, total_sell_count: 0, total_buy_thb: 0, total_sell_thb: 0, total_realized_pnl_thb: 0 },
    })
  })

  it('renders Actual Portfolio Hub title and fetches initial data', async () => {
    render(
      <MemoryRouter>
        <Portfolio />
      </MemoryRouter>
    )

    expect(screen.getByText('Actual Portfolio Hub')).toBeInTheDocument()

    await waitFor(() => {
      expect(api.getActualPortfolioState).toHaveBeenCalledWith(false, false, 'default')
      expect(api.getActualBucketAllocations).toHaveBeenCalled()
    })

    expect(screen.getByText(/Total Portfolio NAV/i)).toBeInTheDocument()
    expect(screen.getAllByText('Core Equities').length).toBeGreaterThanOrEqual(1)
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

  it('infers the PVD transaction view from a holdings symbol in the URL', async () => {
    render(
      <MemoryRouter initialEntries={['/?tab=transactions&symbol=PVD_SINSA_GTO']}>
        <Portfolio />
      </MemoryRouter>
    )

    await waitFor(() => expect(api.getTransactions).toHaveBeenCalledWith('default'))
    expect(screen.getByRole('tab', { name: 'PVD' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tab', { name: 'สินทรัพย์' })).toHaveAttribute('aria-selected', 'false')
  })

  it('lets an explicit tx_view override symbol-based inference', async () => {
    render(
      <MemoryRouter initialEntries={['/?tab=transactions&tx_view=assets&symbol=PVD_SINSA_GTO']}>
        <Portfolio />
      </MemoryRouter>
    )

    await waitFor(() => expect(api.getTransactions).toHaveBeenCalledWith('default'))
    expect(screen.getByRole('tab', { name: 'สินทรัพย์' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('tab', { name: 'PVD' })).toHaveAttribute('aria-selected', 'false')
  })

  it('clears a conflicting URL symbol when a user changes the transaction view', async () => {
    render(
      <MemoryRouter initialEntries={['/?tab=transactions&symbol=PVD_SINSA_GTO']}>
        <Portfolio />
        <CurrentQuery />
      </MemoryRouter>
    )

    await waitFor(() => expect(api.getTransactions).toHaveBeenCalledWith('default'))
    fireEvent.click(screen.getByRole('tab', { name: 'สินทรัพย์' }))
    await waitFor(() => {
      expect(screen.getByRole('tab', { name: 'สินทรัพย์' })).toHaveAttribute('aria-selected', 'true')
      expect(screen.getByTestId('current-query')).toHaveTextContent('tx_view=assets')
      expect(screen.getByTestId('current-query')).not.toHaveTextContent('symbol=')
    })
  })

  it('opens the asset view with a symbol filter from Holdings', async () => {
    render(
      <MemoryRouter initialEntries={['/?tab=overview']}>
        <Portfolio />
        <CurrentQuery />
      </MemoryRouter>
    )

    await waitFor(() => expect(api.getActualPortfolioState).toHaveBeenCalled())
    fireEvent.click(screen.getByRole('button', { name: /Holdings/i }))
    const transactionButton = await screen.findByTitle('ดูประวัติ Transactions ของ AAPL')
    fireEvent.click(transactionButton)

    await waitFor(() => {
      expect(screen.getByRole('tab', { name: 'สินทรัพย์' })).toHaveAttribute('aria-selected', 'true')
      expect(screen.getByTestId('current-query')).toHaveTextContent('tx_view=assets')
      expect(screen.getByTestId('current-query')).toHaveTextContent('symbol=AAPL')
    })
    expect(api.getTransactions).toHaveBeenCalledWith('default')
  })

  it('opens the PVD statements workflow from Summary while preserving the portfolio URL', async () => {
    vi.mocked(api.getActualPortfolioState).mockResolvedValue({
      last_updated: '2026-10-14T16:34:48Z', fx_rates: {},
      summary: { total_value_thb: 1200, total_cost_basis_thb: 1000, total_unrealized_profit: 200, passive_income_ytd: 0 },
      allocation_targets: [],
      holdings: [{
        symbol: 'SINSA_FI', asset_type: 'PVD', units: 120, status: 'active', archived_at: null,
        bucket_id: 'retirement_pvd', avg_cost_usd: null, avg_cost_thb: 8.33,
        current_price_usd: null, current_price_thb: 10, market_value_thb: 1200,
        unrealized_pnl_percent: 20, unrealized_pnl_value: 200,
        market_cap_tier: null, yield_on_cost: null, company_name: 'Fixed Income',
        pe_ratio: null, eps: null, payout_ratio: null, market_cap_value: null,
        dividend_per_share: null, dividend_yield: null, accumulated_dividend_thb: null,
        fundamentals_updated_at: null,
        pvd_valuation: {
          payload_version: 1, fund_code: 'SINSA_FI', sub_fund_code: 'SINSA_FI',
          units: '120', statement_nav: '10', statement_market_value: '1200',
          reconciliation_adjustment: '0', cost_basis: '1000', as_of_date: '2026-06-30',
          statement_fingerprint: 'synthetic', archive_id: 'synthetic',
        },
      }],
      pvd_statements: [], price_refresh_info: { SINSA_FI: 'Statement NAV 10.00 THB (2026-06-30)' },
    })
    render(
      <MemoryRouter initialEntries={['/?portfolio_id=port-pvd&tab=overview&symbol=AAPL']}>
        <Portfolio />
        <CurrentQuery />
      </MemoryRouter>
    )

    await waitFor(() => expect(api.getActualPortfolioState).toHaveBeenCalledWith(false, false, 'port-pvd'))
    fireEvent.click(await screen.findByRole('button', { name: 'ดู/อัปเดต Statement PVD' }))

    await waitFor(() => {
      expect(screen.getByTestId('current-query')).toHaveTextContent('portfolio_id=port-pvd')
      expect(screen.getByTestId('current-query')).toHaveTextContent('tab=transactions')
      expect(screen.getByTestId('current-query')).toHaveTextContent('tx_view=pvd')
      expect(screen.getByTestId('current-query')).not.toHaveTextContent('symbol=')
      expect(screen.getByRole('tab', { name: 'PVD' })).toHaveAttribute('aria-selected', 'true')
    })
  })

  it('ignores an old holding mutation response after switching portfolios', async () => {
    vi.mocked(api.listPortfolios).mockResolvedValue([
      { id: 'default', name: 'พอร์ตลงทุนหลัก', is_default: true, created_at: '2026-07-16T10:00:00Z' },
      { id: 'port-a', name: 'Portfolio A', is_default: false, created_at: '2026-07-16T10:00:00Z' },
      { id: 'port-b', name: 'Portfolio B', is_default: false, created_at: '2026-07-16T10:00:00Z' },
    ])
    const oldState = {
      last_updated: '2026-07-16T10:00:00Z', fx_rates: {},
      summary: { total_value_thb: 1000000, total_cost_basis_thb: 900000, total_unrealized_profit: 100000, passive_income_ytd: 0 },
      allocation_targets: [],
      holdings: [{
        symbol: 'OLD_ASSET', asset_type: 'Stock', units: 1, bucket_id: null,
        avg_cost_usd: null, avg_cost_thb: 10, current_price_usd: null, current_price_thb: 10,
        market_value_thb: 10, unrealized_pnl_percent: 0, unrealized_pnl_value: 0,
        market_cap_tier: null, yield_on_cost: null, company_name: 'Old portfolio asset',
        pe_ratio: null, eps: null, payout_ratio: null, market_cap_value: null,
        dividend_per_share: null, dividend_yield: null, accumulated_dividend_thb: null,
        fundamentals_updated_at: null,
      }],
      price_refresh_info: null,
    }
    const newState = {
      ...oldState,
      summary: { total_value_thb: 200000, total_cost_basis_thb: 180000, total_unrealized_profit: 20000, passive_income_ytd: 0 },
      holdings: [{ ...oldState.holdings[0], symbol: 'NEW_ASSET', company_name: 'New portfolio asset' }],
    }
    vi.mocked(api.getActualPortfolioState).mockImplementation(async (_refresh, _force, portfolioId) =>
      portfolioId === 'port-b' ? newState as any : oldState as any
    )
    let resolveRemove!: (state: any) => void
    const pendingRemove = new Promise<any>((resolve) => { resolveRemove = resolve })
    vi.mocked(api.removeHolding).mockReturnValue(pendingRemove)
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true)

    render(
      <MemoryRouter initialEntries={['/?portfolio_id=port-a&tab=holdings']}>
        <Portfolio />
      </MemoryRouter>
    )

    await waitFor(() => expect(api.getActualPortfolioState).toHaveBeenCalledWith(false, false, 'port-a'))
    fireEvent.click(await screen.findByTitle('ลบ Holding ออกจากพอร์ต'))
    await waitFor(() => expect(api.removeHolding).toHaveBeenCalledWith('OLD_ASSET', 'port-a'))
    fireEvent.change(screen.getAllByRole('combobox')[0]!, { target: { value: 'port-b' } })
    await waitFor(() => expect(api.getActualPortfolioState).toHaveBeenCalledWith(false, false, 'port-b'))
    expect(await screen.findByText('NEW_ASSET')).toBeInTheDocument()

    resolveRemove(oldState as any)
    await waitFor(() => expect(screen.getByText('฿200,000.00')).toBeInTheDocument())
    expect(screen.getByText('NEW_ASSET')).toBeInTheDocument()
    expect(screen.queryByText('OLD_ASSET')).not.toBeInTheDocument()
    confirmSpy.mockRestore()
  })

  it('restores the selected transaction subtab with browser back and forward navigation', async () => {
    render(
      <MemoryRouter initialEntries={['/?tab=transactions&tx_view=assets']}>
        <Portfolio />
        <RouteHistoryControls />
      </MemoryRouter>
    )

    await waitFor(() => expect(api.getTransactions).toHaveBeenCalledWith('default'))
    fireEvent.click(screen.getByRole('tab', { name: 'PVD' }))
    await waitFor(() => expect(screen.getByRole('tab', { name: 'PVD' })).toHaveAttribute('aria-selected', 'true'))

    fireEvent.click(screen.getByRole('button', { name: 'Route back' }))
    await waitFor(() => expect(screen.getByRole('tab', { name: 'สินทรัพย์' })).toHaveAttribute('aria-selected', 'true'))
    fireEvent.click(screen.getByRole('button', { name: 'Route forward' }))
    await waitFor(() => expect(screen.getByRole('tab', { name: 'PVD' })).toHaveAttribute('aria-selected', 'true'))
  })

  it('renders Goals executive section and Overview Performance Card', async () => {
    vi.mocked(api.getActualGoals).mockResolvedValue({
      n_goals: 1,
      goals: [
        {
          name: 'Retirement Fund',
          target_amount_thb: 5000000,
          goal_type: 'nav_target',
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
      expect(screen.getByText(/Portfolio Goals & Targets/i)).toBeInTheDocument()
      expect(screen.getByText('Retirement Fund')).toBeInTheDocument()
    })

    // Overview Tab renders the streamlined Performance & NAV Growth card
    await waitFor(() => {
      expect(screen.getByText(/ผลการดำเนินงานและการเติบโต/i)).toBeInTheDocument()
      expect(screen.getByText('2026-07-16')).toBeInTheDocument()
    })
  })

  it('renders reset portfolio button on principles tab and opens reset modal when clicked', async () => {
    render(
      <MemoryRouter initialEntries={['/portfolio?tab=principles&portfolio_id=default']}>
        <Portfolio />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /ล้างพอร์ต/i })).toBeInTheDocument()
    })

    const resetBtn = screen.getByRole('button', { name: /ล้างพอร์ต/i })
    fireEvent.click(resetBtn)

    await waitFor(() => {
      expect(screen.getByText(/ยืนยันการล้างข้อมูลพอร์ต/i)).toBeInTheDocument()
    })
  })
})

