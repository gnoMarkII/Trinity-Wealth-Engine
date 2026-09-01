import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { InstitutionalThesisCard } from './InstitutionalThesisCard'

describe('InstitutionalThesisCard', () => {
  const mockScorecard = {
    core_conviction_score: 8.2,
    execution_readiness_score: 7.5,
    action_stance: 'ACCUMULATE_NOW' as const,
    fundamental_quality_score: 88,
    guidance_expectation_score: 78,
    valuation_margin_score: 80,
    coverage_pct: 100,
    applicable_pillars_count: 4,
  }

  const mockPiotroski = {
    f_score: 8,
    profitability_points: 4,
    leverage_liquidity_points: 2,
    operating_efficiency_points: 2,
    is_eligible: true,
    status: 'available',
  }

  const mockReverseDcf = {
    target_price_12m: 95.5,
    upside_12m_pct: 19.4,
    intrinsic_value_today: 85.0,
    market_implied_growth_pct: 7.5,
    solver_status: 'converged',
    status: 'available',
    is_eligible: true,
  }

  const mockTactical = {
    price_stage: 'STAGE_2_MARKUP',
    current_price: 80.0,
    buy_zone_min: 76.0,
    buy_zone_max: 82.0,
    invalidation_stop_loss: 72.0,
    tactical_target_price: 98.0,
    tactical_risk_reward_ratio: 2.25,
    status: 'available',
  }

  const mockInsider = {
    status: 'bullish_cluster',
    data_status: 'available',
    open_market_p_count_90d: 3,
    c_suite_p_count: 2,
  }

  const mockFalsifiers = [
    {
      falsifier_id: 'TEST_MARGIN_FLOOR',
      metric_name: 'Operating Margin',
      condition: 'EBIT margin falls below 25.5%',
      threshold_value: 25.5,
      source_ref: 'json-pointer:///fixed_parameters/base_ebit_margin_pct',
      narrative_explanation: 'Operating margin breakdown falsifier',
    },
  ]

  it('renders Action Stance and Dual Scores correctly', () => {
    render(
      <InstitutionalThesisCard
        scorecard={mockScorecard}
        piotroski={mockPiotroski}
        reverseDcf={mockReverseDcf}
        tactical={mockTactical}
        insider={mockInsider}
        falsifiers={mockFalsifiers}
      />
    )

    expect(screen.getByText('ACCUMULATE NOW')).toBeInTheDocument()
    expect(screen.getByText('8.2')).toBeInTheDocument()
    expect(screen.getByText('7.5')).toBeInTheDocument()
    expect(screen.getByText('8 / 9')).toBeInTheDocument()
    expect(screen.getByText('$95.5')).toBeInTheDocument()
    expect(screen.getByText('+19.4%')).toBeInTheDocument()
    expect(screen.getByText('$76 - $82')).toBeInTheDocument()
    expect(screen.getByText(/bullish cluster/i)).toBeInTheDocument()
    expect(screen.getAllByText(/Operating Margin/i).length).toBeGreaterThanOrEqual(1)
  })

  it('renders null if scorecard is missing', () => {
    const { container } = render(<InstitutionalThesisCard scorecard={null} />)
    expect(container.firstChild).toBeNull()
  })
})
