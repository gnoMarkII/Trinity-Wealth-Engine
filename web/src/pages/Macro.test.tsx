import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import Macro from './Macro'
import { api } from '../api/client'

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>()
  return {
    ...actual,
    api: {
      getMacroDashboard: vi.fn(),
      createKanbanCard: vi.fn(),
      dispatchJob: vi.fn(),
      getTreasuryYieldCurve: vi.fn().mockResolvedValue({
        observation_date: '2026-09-26',
        yields: [
          { maturity: '2 Yr', yield_percent: 4.15 },
          { maturity: '10 Yr', yield_percent: 4.25 },
        ],
        spread_10y_2y_bps: 10,
        fetched_at: 1727337600,
        source: 'US Treasury',
        unit: '%',
        is_stale: false,
      }),
      getFinancialStress: vi.fn().mockResolvedValue({
        as_of_date: '2026-09-24',
        published_at: '2026-09-26',
        fsi_value: -1.25,
        regime: 'Below Average Stress',
        categories: [
          { label: 'Credit', value: -0.4 },
          { label: 'Equity Valuation', value: -0.2 },
        ],
        trend_90d: [],
        source: 'OFR',
        data_lag_days: 2,
        is_stale: false,
      }),
      getMetalsCot: vi.fn().mockResolvedValue(null),
      getGlobalPolicyRates: vi.fn().mockResolvedValue({
        as_of_date: '2026-09-26',
        rates: [
          { country: 'US', rate_value: 5.25, rate_type: 'Fed Funds' },
          { country: 'TH', rate_value: 2.50, rate_type: 'BOT Repo' },
        ],
        spreads_vs_bot_repo: { US: 275 },
      }),
      getCommodityVolatility: vi.fn().mockResolvedValue([]),
      getTreasuryAuctionDemand: vi.fn().mockResolvedValue(null),
      getNationalDebt: vi.fn().mockResolvedValue([]),
      getThaiInvestorFlow: vi.fn().mockResolvedValue({
        market: 'SET',
        as_of: '2026-09-26',
        total_value: 45000000000,
        investors: [
          { investor_type: 'Foreign Investors', buy_value: 20000000000, sell_value: 22000000000, net_value: -2000000000 },
        ],
        source: 'Settrade',
        is_stale: false,
      }),
      getThaiRetailGold: vi.fn().mockResolvedValue({
        source: 'GTA Thailand',
        bar: { buy: 41000, sell: 41100 },
        ornament: { buy: 40200, sell: 41600 },
        announced_at: '2026-09-26',
        revision: 1,
        is_stale: false,
      }),
      getThaiMarketValuation: vi.fn().mockResolvedValue({
        market: 'SET',
        pe_ratio: 16.5,
        pbv_ratio: 1.4,
        dividend_yield: 3.2,
        source: 'Settrade',
        is_stale: false,
      }),
      getThaiMarketBreadth: vi.fn().mockResolvedValue({
        market: 'SET',
        gainers: 250,
        losers: 200,
        unchanged: 150,
        source: 'Settrade',
        is_stale: false,
      }),
    },
  }
})

const mockMacroDashboard = {
  overall_regime: 'Goldilocks',
  time_horizon: '12m',
  conviction_level: 'High',
  conviction_rationale: 'Growth is robust while inflation continues to moderate.',
  quant_narrative_alignment: 'Aligned',
  key_assumptions: ['Fed rate cuts expected'],
  regime_probabilities: { Goldilocks: 0.6, Reflation: 0.2, Stagflation: 0.1, Recession: 0.1 },
  warnings: [],
  asset_allocation: [],
  pair_trades: [],
  risk_scenarios: [],
  evaluated_at: '2026-08-08',
}

describe('Macro Page - Dual Track Cockpit', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders macro cockpit with provenance badges and triggers analysis update', async () => {
    vi.mocked(api.getMacroDashboard).mockResolvedValue(mockMacroDashboard as any)
    vi.mocked(api.createKanbanCard).mockResolvedValue({
      created: true,
      card: {
        card_id: 'macro-card-1',
        title: 'วิเคราะห์ภาวะเศรษฐกิจมหภาค (Macro Analysis)',
        prompt: 'p',
        column_name: 'backlog',
        job_id: null,
        flow: 'manager',
        scope: 'both',
        display_seq: 1,
        discord_notify: true,
        is_verified: true,
        created_at: 1,
        updated_at: 1,
      },
    })
    vi.mocked(api.dispatchJob).mockResolvedValue({
      job_id: 'job-macro-1',
      status: 'running',
      card_id: 'macro-card-1',
      error_message: null,
      current_node: null,
      interrupt_payload: null,
      log_count: 0,
      created_at: 1,
      updated_at: 1,
    })

    render(
      <MemoryRouter>
        <Macro />
      </MemoryRouter>
    )

    // Check Header and Provenance badges
    expect(screen.getByText('Macro Cockpit (ห้องควบคุมภาวะเศรษฐกิจมหภาค)')).toBeInTheDocument()
    expect(screen.getAllByText('ข้อมูลจากผู้ให้บริการ')[0]).toBeInTheDocument()
    expect(screen.getAllByText('ระบบคำนวณ')[0]).toBeInTheDocument()
    expect(screen.getAllByText('AI วิเคราะห์')[0]).toBeInTheDocument()

    // Default tab is AI Analysis
    await waitFor(() => {
      expect(screen.getAllByText(/Goldilocks/i)[0]).toBeInTheDocument()
      expect(screen.getByText(/บทวิเคราะห์สภาวะเศรษฐกิจและการจัดสรรสินทรัพย์เชิงลึก/)).toBeInTheDocument()
    })

    const updateBtn = screen.getByRole('button', { name: /อัปเดตบทวิเคราะห์/ })
    expect(updateBtn).toBeInTheDocument()

    await userEvent.click(updateBtn)

    await waitFor(() => {
      expect(api.createKanbanCard).toHaveBeenCalledWith(
        'วิเคราะห์ภาวะเศรษฐกิจมหภาค (Macro Analysis)',
        'manager',
        'วิเคราะห์ภาวะเศรษฐกิจมหภาค (Macro Intelligence & Regime Analysis) ล่าสุดพร้อมประเมิน Asset Allocation',
        'both'
      )
      expect(api.dispatchJob).toHaveBeenCalledWith(
        'วิเคราะห์ภาวะเศรษฐกิจมหภาค (Macro Intelligence & Regime Analysis) ล่าสุดพร้อมประเมิน Asset Allocation',
        'macro-card-1',
        'manager',
        'both'
      )
      expect(screen.getByText('สั่งงานวิเคราะห์ภาวะเศรษฐกิจมหภาคเรียบร้อย')).toBeInTheDocument()
    })
  })

  it('switches between AI, US, Thailand, and Cross-Border tabs', async () => {
    vi.mocked(api.getMacroDashboard).mockResolvedValue(mockMacroDashboard as any)

    render(
      <MemoryRouter>
        <Macro />
      </MemoryRouter>
    )

    // Switch to US tab
    const usTabBtn = screen.getByRole('button', { name: /สหรัฐอเมริกา \(US\)/i })
    await userEvent.click(usTabBtn)

    await waitFor(() => {
      expect(screen.getByText('US Treasury Yield Curve (Cross-Section)')).toBeInTheDocument()
      expect(screen.getByText('US Financial Stress Index (OFR FSI)')).toBeInTheDocument()
    })

    // Switch to Thailand tab
    const thaiTabBtn = screen.getByRole('button', { name: /ประเทศไทย \(TH\)/i })
    await userEvent.click(thaiTabBtn)

    await waitFor(() => {
      expect(screen.getByText(/สถานะเศรษฐกิจไทย: ยังประเมินไม่ได้/i)).toBeInTheDocument()
      expect(screen.getByText(/SET 4-Investor Flow/i)).toBeInTheDocument()
      expect(screen.getByText(/SET Market Breadth/i)).toBeInTheDocument()
    })

    // Switch to Cross-Border tab
    const crossBorderTabBtn = screen.getByRole('button', { name: /ความเชื่อมโยง US–TH/i })
    await userEvent.click(crossBorderTabBtn)

    await waitFor(() => {
      expect(screen.getByText(/Fed Funds vs BoT 1D Repo/i)).toBeInTheDocument()
      expect(screen.getByText(/US 10Y Yield vs Foreign SET Flow/i)).toBeInTheDocument()
    })
  })

  it('isolates AI failure so market observables remain visible', async () => {
    vi.mocked(api.getMacroDashboard).mockRejectedValue(new Error('Strategy snapshot not found'))

    render(
      <MemoryRouter>
        <Macro />
      </MemoryRouter>
    )

    // Even if AI report fails, the page does NOT crash or blank out
    // and automatically falls back to US tab displaying pure market observables
    await waitFor(() => {
      expect(screen.getByText('US Treasury Yield Curve (Cross-Section)')).toBeInTheDocument()
      expect(screen.getByText('US Financial Stress Index (OFR FSI)')).toBeInTheDocument()
    })
  })
})
