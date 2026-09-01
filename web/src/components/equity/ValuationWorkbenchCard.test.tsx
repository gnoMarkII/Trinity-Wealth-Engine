import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import ValuationWorkbenchCard from './ValuationWorkbenchCard'

describe('ValuationWorkbenchCard', () => {
  const mockReverseDcf = {
    target_price_12m: 117.24,
    upside_12m_pct: -29.4,
    intrinsic_value_today: 108.50,
    market_implied_growth_pct: 24.05,
    market_implied_margin_pct: 36.06,
    enterprise_value: 48000000000,
    equity_value: 45000000000,
    sum_pv_5y_fcf: 12000000000,
    terminal_value_pv: 36000000000,
    solver_status: 'converged' as const,
  }

  const mockDcf = {
    valuation_verdict: 'overvalued' as const,
    wacc_pct: 9.5,
    cost_of_equity_pct: 10.2,
    cost_of_debt_pct: 5.0,
    risk_free_rate_pct: 4.25,
    erp_pct: 5.5,
    observable_refs: [],
    scenarios: {
      bear: { target_price: 85.0, upside_pct: -45.0, probability_weight: 0.25, margin_of_safety_pct: 0 },
      base: { target_price: 117.24, upside_pct: -29.4, probability_weight: 0.50, margin_of_safety_pct: 0 },
      bull: { target_price: 165.0, upside_pct: 15.0, probability_weight: 0.25, margin_of_safety_pct: 0 },
    },
    sensitivity_matrix: [],
  }

  const mockPiotroski = {
    f_score: 6,
    status: 'AVAILABLE',
    profitability_points: 3,
    leverage_liquidity_points: 2,
    operating_efficiency_points: 1,
  }

  const mockQuality = {
    roic_pct: 28.5,
    fcf_margin_pct: 38.2,
    fcf_yield_pct: 5.4,
    ocf_to_net_income: 1.25,
    solvency_score: 76.1,
  }

  it('renders Reverse DCF Expectation Gap metrics correctly', () => {
    render(
      <ValuationWorkbenchCard
        reverseDcf={mockReverseDcf}
        dcf={mockDcf}
        piotroski={mockPiotroski}
        qualityMetrics={mockQuality}
      />
    )

    expect(screen.getByText('24.1%')).toBeInTheDocument()
    expect(screen.getByText('36.1%')).toBeInTheDocument()
    expect(screen.getAllByText('$117.24').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('overvalued')).toBeInTheDocument()
    expect(screen.getByText('6 / 9')).toBeInTheDocument()
    expect(screen.getByText('28.5%')).toBeInTheDocument()
    expect(screen.getByText('Reverse DCF Verdict (12M)')).toBeInTheDocument()
  })

  it('keeps Reverse DCF verdict badge overvalued even if Generic DCF scenarios are highly bullish', () => {
    const bullishGenericDcf = {
      ...mockDcf,
      scenarios: {
        bear: { target_price: 261.47, upside_pct: 57.5, probability_weight: 0.25, margin_of_safety_pct: 0 },
        base: { target_price: 437.26, upside_pct: 163.4, probability_weight: 0.50, margin_of_safety_pct: 0 },
        bull: { target_price: 555.45, upside_pct: 234.6, probability_weight: 0.25, margin_of_safety_pct: 0 },
      },
    }

    render(
      <ValuationWorkbenchCard
        reverseDcf={mockReverseDcf}
        dcf={bullishGenericDcf}
        piotroski={mockPiotroski}
        qualityMetrics={mockQuality}
      />
    )

    expect(screen.getByText('overvalued')).toBeInTheDocument()
    expect(screen.getByText('Reverse DCF Verdict (12M)')).toBeInTheDocument()
  })

  it('returns null when no data provided', () => {
    const { container } = render(<ValuationWorkbenchCard />)
    expect(container.firstChild).toBeNull()
  })
})
