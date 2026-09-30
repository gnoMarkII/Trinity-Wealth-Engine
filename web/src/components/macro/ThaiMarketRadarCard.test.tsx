import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ThaiMarketRadarCard } from './ThaiMarketRadarCard'
import { api } from '../../api/client'

vi.mock('../../api/client', () => ({
  api: {
    getThaiInvestorFlow: vi.fn(),
    getThaiRetailGold: vi.fn(),
    getThaiMarketValuation: vi.fn(),
    getThaiMarketBreadth: vi.fn(),
  },
}))

describe('ThaiMarketRadarCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getThaiInvestorFlow).mockResolvedValue({
      market: 'SET',
      as_of: '2026-09-25',
      total_value: 45000000000,
      investors: [
        { investor_type: 'Foreign Investors (นักลงทุนต่างชาติ)', buy_value: 20000000000, sell_value: 18000000000, net_value: 2000000000 },
        { investor_type: 'Local Institutions (สถาบันในประเทศ)', buy_value: 5000000000, sell_value: 6000000000, net_value: -1000000000 },
        { investor_type: 'Proprietary Trading (บัญชี บล.)', buy_value: 4000000000, sell_value: 4500000000, net_value: -500000000 },
        { investor_type: 'Retail Investors (นักลงทุนรายย่อย)', buy_value: 16000000000, sell_value: 16500000000, net_value: -500000000 },
      ],
      source: 'Settrade',
      is_stale: false,
    })

    vi.mocked(api.getThaiRetailGold).mockResolvedValue({
      source: 'Gold Traders Association',
      unit: 'baht-weight (15.244 g, 96.5%)',
      bar: { buy: 40500, sell: 40600 },
      ornament: { buy: 39769.64, sell: 41100 },
      announced_at: '2026-09-25 09:05:00',
      revision: 1,
      is_stale: false,
    })

    vi.mocked(api.getThaiMarketValuation).mockResolvedValue({
      market: 'SET',
      as_of: '2026-09-25',
      market_cap: 17500000000000,
      pe_ratio: 14.85,
      pbv_ratio: 1.25,
      dividend_yield: 3.42,
      turnover_ratio: 1.95,
      source: 'Settrade',
      is_stale: false,
    })

    vi.mocked(api.getThaiMarketBreadth).mockResolvedValue({
      market: 'SET',
      as_of: '2026-09-25',
      gainers: 280,
      losers: 190,
      unchanged: 150,
      source: 'Settrade',
      is_stale: false,
    })
  })

  it('renders Thailand Market Radar header, fail-closed notice, and Terminal V2 metrics', async () => {
    render(<ThaiMarketRadarCard />)

    expect(screen.getByText('Thailand Market Radar & Stance')).toBeInTheDocument()
    expect(screen.getByText('TERMINAL V2')).toBeInTheDocument()

    // Dual-track fail-closed status and gaps
    expect(screen.getByText('Thailand Macro Regime: UNKNOWN')).toBeInTheDocument()
    expect(screen.getByText('NESDC Real GDP YoY')).toBeInTheDocument()
    expect(screen.getByText('MOC TPSO CPI YoY')).toBeInTheDocument()

    // Data should load
    await waitFor(() => {
      expect(screen.getByText('+2,000.0 M')).toBeInTheDocument()
    })

    expect(screen.getByText('40,600')).toBeInTheDocument()
    expect(screen.getByText('14.85x')).toBeInTheDocument()
    expect(screen.getByText('1.47x')).toBeInTheDocument() // 280 / 190 = 1.47
  })

  it('triggers refresh when clicking refresh button', async () => {
    const user = userEvent.setup()
    render(<ThaiMarketRadarCard />)

    await waitFor(() => {
      expect(screen.getByText('รีเฟรชสด')).toBeInTheDocument()
    })

    const refreshBtn = screen.getByRole('button', { name: /รีเฟรชสด/i })
    await user.click(refreshBtn)

    expect(api.getThaiInvestorFlow).toHaveBeenCalledTimes(2)
    expect(api.getThaiRetailGold).toHaveBeenCalledTimes(2)
    expect(api.getThaiMarketValuation).toHaveBeenCalledTimes(2)
    expect(api.getThaiMarketBreadth).toHaveBeenCalledTimes(2)
  })
})
