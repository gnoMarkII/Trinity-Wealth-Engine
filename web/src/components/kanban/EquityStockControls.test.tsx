import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import EquityStockControls, { parseEquityStock, updateEquityPromptAndTitle } from './EquityStockControls'
import { api } from '../../api/client'

vi.mock('../../api/client', () => ({
  api: {
    getActualPortfolioState: vi.fn(),
    getActualWatchlist: vi.fn(),
  },
}))

describe('EquityStockControls', () => {
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

  it('parses ticker and market correctly', () => {
    expect(parseEquityStock('วิเคราะห์หุ้น AAPL (US) และดึงข่าวล่าสุด')).toEqual({ ticker: 'AAPL', market: 'US' })
    expect(parseEquityStock('วิเคราะห์หุ้น PTT.BK (TH) และดึงข่าวล่าสุด')).toEqual({ ticker: 'PTT.BK', market: 'TH' })
    expect(parseEquityStock('วิเคราะห์หุ้น NVDA และดึงข่าว')).toEqual({ ticker: 'NVDA', market: 'US' })
  })

  it('updates prompt and title correctly', () => {
    const originalPrompt = 'วิเคราะห์หุ้น AAPL (US) และดึงข่าวล่าสุดพร้อมประเมิน Valuation'
    const originalTitle = 'วิเคราะห์หุ้นและดึงข่าว (Equity & News)'
    const res = updateEquityPromptAndTitle(originalPrompt, originalTitle, 'NVDA', 'US')
    expect(res.prompt).toBe('วิเคราะห์หุ้น NVDA (US) และดึงข่าวล่าสุดพร้อมประเมิน Valuation')
    expect(res.title).toBe('วิเคราะห์หุ้น NVDA (US)')
  })

  it('renders control and handles option selection', async () => {
    vi.mocked(api.getActualPortfolioState).mockResolvedValueOnce({
      last_updated: null,
      fx_rates: {},
      summary: { total_value_thb: 0, total_cost_basis_thb: 0, total_unrealized_profit: 0, passive_income_ytd: 0 },
      allocation_targets: [],
      holdings: [
        { symbol: 'NVDA', asset_type: 'Stock', units: 5, bucket_id: null, avg_cost_usd: 100, avg_cost_thb: null, current_price_usd: 120, current_price_thb: 4200, market_value_thb: 21000, unrealized_pnl_percent: 20, unrealized_pnl_value: 100, market_cap_tier: null, yield_on_cost: null, company_name: 'NVIDIA Corp.', pe_ratio: null, eps: null, payout_ratio: null, market_cap_value: null, dividend_per_share: null, dividend_yield: null, accumulated_dividend_thb: null, fundamentals_updated_at: null },
      ],
      price_refresh_info: null,
    })

    const handleChange = vi.fn()

    render(
      <EquityStockControls
        prompt="วิเคราะห์หุ้น AAPL และดึงข่าวล่าสุด"
        title="วิเคราะห์หุ้นและดึงข่าว (Equity & News)"
        onChange={handleChange}
      />
    )

    const pill = await screen.findByRole('button', { name: '💼 NVDA' })
    fireEvent.click(pill)

    expect(handleChange).toHaveBeenCalledWith({
      title: 'วิเคราะห์หุ้น NVDA (US)',
      prompt: 'วิเคราะห์หุ้น NVDA (US) และดึงข่าวล่าสุด',
    })
  })
})
