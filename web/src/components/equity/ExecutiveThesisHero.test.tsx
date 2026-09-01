import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import ExecutiveThesisHero, { getStanceBadgeStyle } from './ExecutiveThesisHero'

describe('ExecutiveThesisHero', () => {
  const mockScorecard = {
    action_stance: 'BREAKOUT_BUY',
    core_conviction_score: 6.9,
    execution_readiness_score: 5.5,
    fundamental_quality_score: 83.3,
    guidance_expectation_score: 90.0,
    valuation_margin_score: 30.0,
    coverage_pct: 100.0,
    applicable_pillars_count: 4,
  }

  it('renders Action Stance and Conviction score correctly', () => {
    render(
      <ExecutiveThesisHero
        scorecard={mockScorecard}
        baseCaseSummary="Solid base case summary"
      />
    )

    expect(screen.getByText(/BREAKOUT BUY/)).toBeInTheDocument()
    expect(screen.getByText('6.9 / 10')).toBeInTheDocument()
    expect(screen.getByText('5.5 / 10')).toBeInTheDocument()
    expect(screen.getByText('83.3')).toBeInTheDocument()
    expect(screen.getByText('Solid base case summary')).toBeInTheDocument()
  })

  it('renders Earnings Call highlight and handles view click', () => {
    const onViewMock = vi.fn()
    const mockCall = {
      title: 'Q2 2026',
      ticker: 'FTNT',
      period: 'Q2 2026',
      vault_path: 'path/to/note',
      highlights: 'Highlights of earnings call',
      date: '2026-08-29',
      last_updated: '2026-08-29',
      has_full_transcript: true,
    }

    render(
      <ExecutiveThesisHero
        scorecard={mockScorecard}
        latestEarningsCall={mockCall}
        onViewEarningsCall={onViewMock}
      />
    )

    expect(screen.getByText(/Earnings Call Highlights \(Q2 2026\)/)).toBeInTheDocument()
    const btn = screen.getByRole('button', { name: /ดูฉบับเต็ม/ })
    fireEvent.click(btn)
    expect(onViewMock).toHaveBeenCalledTimes(1)
  })

  it('returns correct badge style for various stances', () => {
    expect(getStanceBadgeStyle('BREAKOUT_BUY').bg).toContain('emerald')
    expect(getStanceBadgeStyle('TRIM').bg).toContain('amber')
    expect(getStanceBadgeStyle('AVOID').bg).toContain('rose')
    expect(getStanceBadgeStyle('HOLD').bg).toContain('sky')
  })
})
