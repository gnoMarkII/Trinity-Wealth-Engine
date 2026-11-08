import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { VariantPerceptionCard } from './VariantPerceptionCard'
import type { VariantPerceptionSnapshotDTO } from '../../api/types'

const mockSnapshot: VariantPerceptionSnapshotDTO = {
  ticker: 'GOOGL',
  period: '2026Q3',
  metric_name: 'cloud_revenue',
  reported_value: '11500.00',
  consensus_value: '10800.00',
  fund_thesis_value: '11400.00',
  street_surprise: '700.00',
  street_surprise_pct: '6.48',
  variant_spread: '600.00',
  variant_spread_pct: '5.56',
  variant_realization: '100.00',
  variant_realization_pct: '0.88',
  alpha_verdict: 'ALPHA_CONFIRMED',
  unit: 'USD_M',
}

describe('VariantPerceptionCard Component', () => {
  it('renders loading skeleton when isLoading is true', () => {
    render(<VariantPerceptionCard data={mockSnapshot} isLoading={true} />)
    expect(screen.getByTestId('variant-perception-loading')).toBeInTheDocument()
  })

  it('renders 3-way estimates and spreads accurately', () => {
    render(<VariantPerceptionCard data={mockSnapshot} />)

    // Header & Badge
    expect(screen.getByText('$GOOGL')).toBeInTheDocument()
    expect(screen.getByText('2026Q3')).toBeInTheDocument()
    expect(screen.getByText('CLOUD REVENUE')).toBeInTheDocument()
    expect(screen.getByTestId('verdict-alpha-confirmed')).toBeInTheDocument()

    // 3-way pillars
    expect(screen.getByTestId('consensus-value')).toHaveTextContent('10800.00')
    expect(screen.getByTestId('fund-thesis-value')).toHaveTextContent('11400.00')
    expect(screen.getByTestId('reported-value')).toHaveTextContent('11500.00')

    // Spreads
    expect(screen.getByTestId('street-surprise')).toHaveTextContent('+700.00 USD_M')
    expect(screen.getByTestId('variant-spread')).toHaveTextContent('+600.00 USD_M')
    expect(screen.getByTestId('variant-realization')).toHaveTextContent('+100.00 USD_M')
  })

  it('renders CONSENSUS_CORRECT verdict badge properly', () => {
    const consensusSnapshot: VariantPerceptionSnapshotDTO = {
      ...mockSnapshot,
      alpha_verdict: 'CONSENSUS_CORRECT',
    }
    render(<VariantPerceptionCard data={consensusSnapshot} />)
    expect(screen.getByTestId('verdict-consensus-correct')).toBeInTheDocument()
  })
})

