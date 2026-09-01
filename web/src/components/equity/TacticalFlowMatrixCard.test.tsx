import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import TacticalFlowMatrixCard from './TacticalFlowMatrixCard'

describe('TacticalFlowMatrixCard', () => {
  const mockTactical = {
    price_stage: 'STAGE_2_MARKUP',
    atr_14: 3.25,
    key_support_level: 154.46,
    key_resistance_level: 175.50,
    buy_zone_min: 151.39,
    buy_zone_max: 160.60,
    invalidation_stop_loss: 149.85,
    tactical_target_price: 169.91,
    tactical_risk_reward_ratio: 0.24,
    horizon_timeframe: '1-3M',
    status: 'AVAILABLE',
  }

  const mockInsider = {
    status: 'neutral_no_signal',
    data_status: 'ok',
    open_market_p_count_90d: 0,
    open_market_p_value_usd: 0,
    open_market_s_count_90d: 2,
    open_market_s_value_usd: 500000,
    c_suite_p_count: 0,
  }

  const mockSmartMoney = {
    overall_smart_money_flag: 'neutral' as const,
    insider_signal: 'neutral' as const,
    insider_buy_count_90d: 0,
    insider_sell_count_90d: 2,
    institutional_ownership_pct: 78.5,
    insider_ownership_pct: 12.3,
    short_interest_pct: 3.8,
    short_squeeze_risk: false,
  }

  const mockSentiment = {
    evaluated_at: '2026-08-29',
    market_sentiment: 'bullish' as const,
    key_themes: ['Unified SASE', 'AI Capex'],
    tail_risks: ['Firewall refresh delay'],
    sources_summary: 'SEC + News',
    report_references: [],
  }

  it('renders Tactical setup and Flow matrix correctly', () => {
    render(
      <TacticalFlowMatrixCard
        tactical={mockTactical}
        insider={mockInsider}
        smartMoney={mockSmartMoney}
        sentiment={mockSentiment}
      />
    )

    expect(screen.getByText('STAGE 2 MARKUP')).toBeInTheDocument()
    expect(screen.getByText('$151.39 - $160.60')).toBeInTheDocument()
    expect(screen.getByText('$149.85')).toBeInTheDocument()
    expect(screen.getByText('$169.91')).toBeInTheDocument()
    expect(screen.getByText('78.5%')).toBeInTheDocument()
    expect(screen.getByText('Unified SASE')).toBeInTheDocument()
    expect(screen.getByText('Firewall refresh delay')).toBeInTheDocument()
  })

  it('returns null when no data provided', () => {
    const { container } = render(<TacticalFlowMatrixCard />)
    expect(container.firstChild).toBeNull()
  })
})
