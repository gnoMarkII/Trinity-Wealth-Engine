import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { ThailandMacroSection } from './ThailandMacroSection'
import { CrossBorderSection } from './CrossBorderSection'
import { FedBotRatePair } from './FedBotRatePair'
import { CryptoLiquidityCard } from '../CryptoLiquidityCard'

describe('Macro market data integrity', () => {
  it('preserves fractional basis points from the policy rate provider', () => {
    render(<FedBotRatePair ratesData={{ as_of_date: '2026-10-04',
      rates: [{ country: 'US', rate_value: 3.625 }, { country: 'TH', rate_value: 1.0 }],
      spreads_vs_bot_repo: { US: 262.5 } }} />)
    expect(screen.getByText('+262.5 bps')).toBeInTheDocument()
    expect(screen.queryByText('+263 bps')).not.toBeInTheDocument()
  })
  it('uses refreshed Thai provider values ahead of the older AI report', () => {
    render(<ThailandMacroSection
      flow={{ market: 'SET', as_of: '2026-10-02', total_value: 1, source: 'Settrade', is_stale: false,
        investors: [{ investor_type: 'foreign', buy_value: 0, sell_value: 850e6, net_value: -850e6 }] }}
      breadth={{ market: 'SET', as_of: '2026-10-02', gainers: 265, losers: 158, unchanged: 228, source: 'Settrade', is_stale: false }}
      valuation={{ market: 'SET', as_of: '2026-10-02', pe_ratio: 15.64, source: 'Settrade', is_stale: false }}
      gold={{ source: 'GTA', unit: 'THB', announced_at: '2026-10-04', is_stale: false, bar: { buy: 65750, sell: 65950 }, ornament: { buy: 64430, sell: 66750 } }}
      globalPolicyRates={{ as_of_date: '2026-10-04', spreads_vs_bot_repo: { US: 262.5 } }}
      aiData={{ evaluated_at: '2026-10-01', regional_assessments: { Thailand: { economic_state: 'Reflation', data_gaps: [], confidence: 0.9 } },
        thailand_market_stance: { investor_flow: { foreign_net_mb: 1234 }, market_breadth: { advance_decline_ratio: 0.5 },
          valuation: { pe_ratio: 27 }, physical_gold: { bar_sell_thb: 50000 }, policy_spread_bps: 275 } } as any}
    />)
    expect(screen.getByText('-850 ลบ.')).toBeInTheDocument()
    expect(screen.getAllByText('1.68x')).toHaveLength(2)
    expect(screen.getAllByText('15.64x')).toHaveLength(2)
    expect(screen.getByText('65,950 ฿')).toBeInTheDocument()
    expect(screen.getByText('+262.5 bps')).toBeInTheDocument()
    expect(screen.queryByText('27.00x')).not.toBeInTheDocument()
    expect(screen.queryByText('50,000 ฿')).not.toBeInTheDocument()
    expect(screen.queryByText('Real GDP YoY')).not.toBeInTheDocument()
  })

  it('renders nullable Thai snapshot metrics without crashing', () => {
    render(<ThailandMacroSection flow={null} breadth={null} valuation={null} gold={null}
      aiData={{ evaluated_at: '2026-10-01', thailand_market_stance: {
        investor_flow: { foreign_net_mb: null }, market_breadth: { advance_decline_ratio: null },
        valuation: { pe_ratio: null }, physical_gold: { bar_sell_thb: null },
      } } as any}
    />)
    expect(screen.getAllByText('ไม่มีข้อมูล').length).toBeGreaterThan(0)
  })

  it('distinguishes zero foreign net flow from unavailable data', () => {
    const { rerender } = render(<CrossBorderSection globalPolicyRates={null} yieldCurve={null}
      flow={{ market: 'SET', as_of: '2026-10-02', total_value: 0, source: 'Settrade', is_stale: false,
        investors: [{ investor_type: 'foreign', buy_value: 0, sell_value: 0, net_value: 0 }] }} />)
    expect(screen.getByText('0.0 M')).toBeInTheDocument()
    rerender(<CrossBorderSection globalPolicyRates={null} yieldCurve={null} flow={null} />)
    expect(screen.queryByText('+— M')).not.toBeInTheDocument()
    expect(screen.queryByText('0.0 M')).not.toBeInTheDocument()
  })

  it('does not display invalid fiscal registry evidence as current Thai debt', () => {
    render(<ThailandMacroSection flow={null} breadth={null} valuation={null} gold={null}
      aiData={{ evaluated_at: '2026-10-04', observable_registry: {
        obs_th_debt_to_gdp_mof: { value: '66.09', is_valid: false, status: 'stale' },
        obs_th_public_debt_mof: { value: '12,595,731.58', is_valid: false, status: 'stale' },
      } } as any} />)
    expect(screen.queryByText('66.09%')).not.toBeInTheDocument()
    expect(screen.getAllByText('รอข้อมูล MOF')).toHaveLength(2)
  })

  it('shows zero ETF flow, stale status, and source date for crypto data', () => {
    render(<CryptoLiquidityCard data={{
      etf_daily_net_inflow_usd: 0, top_stablecoins: [], liquidity_regime: 'Neutral',
      as_of_date: '2026-10-02', fetched_at: 0, source: 'DeFiLlama', is_stale: true,
      stale_reason: 'Benchmark unavailable', limitations: 'Partial coverage',
    }} />)
    expect(screen.getByText('+$0.0M')).toBeInTheDocument()
    expect(screen.getByText(/Benchmark unavailable/)).toBeInTheDocument()
    expect(screen.getByText(/ข้อมูล ณ 2026-10-02/)).toBeInTheDocument()
    expect(screen.getByText('Partial coverage')).toBeInTheDocument()
  })

  it('keeps crypto failures visible on the dashboard', () => {
    render(<CryptoLiquidityCard data={null} error="Provider unavailable" />)
    expect(screen.getByRole('status')).toHaveTextContent('Provider unavailable')
  })
})
