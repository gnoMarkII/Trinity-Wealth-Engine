import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { StructuredThesisCard } from './StructuredThesisCard'
import type { ThesisClaimDTO } from '../../api/types'

const mockClaim: ThesisClaimDTO = {
  claim_id: 'TH-NVDA-v1',
  ticker: 'NVDA',
  version: 1,
  prior_claim_id: null,
  as_of_utc: '2026-11-01T00:00:00Z',
  claim_statement: 'NVDA maintains >70% gross margins through Blackwell GPU ramp.',
  falsifiers: [
    {
      falsifier_id: 'FAL-MARGIN',
      metric_name: 'gross_margin',
      operator: '<',
      threshold_value: '70.0',
      unit: 'PERCENT',
      period_basis: 'QUARTERLY',
      invalidation_impact: 'BREAKS_BUY_CASE',
      narrative_explanation: 'Margin below 70% breaks core pricing power premise.',
    },
  ],
  linked_catalysts: [
    {
      catalyst_id: 'CAT-Q3',
      event_name: 'Q3 Earnings Call',
      expected_date_str: '2026-11-20',
      confidence_window_days: 5,
      source_locator: 'SEC 8-K',
      status: 'PENDING',
    },
  ],
  status: 'ACTIVE',
  evidence_tier: 'DOCUMENTED',
  author_role: 'HUMAN_PM',
  checksum_sha256: 'a1b2c3d4e5f6789012345678',
}

describe('StructuredThesisCard Component', () => {
  it('renders thesis statement, falsifiers, and catalysts correctly', () => {
    render(<StructuredThesisCard ticker="NVDA" initialClaim={mockClaim} />)

    // Header & Evidence Tier
    expect(screen.getByText('Institutional Investment Thesis (v1)')).toBeInTheDocument()
    expect(screen.getByText('DOCUMENTED')).toBeInTheDocument()
    expect(screen.getByText('ACTIVE')).toBeInTheDocument()

    // Core Statement
    expect(
      screen.getByText('NVDA maintains >70% gross margins through Blackwell GPU ramp.')
    ).toBeInTheDocument()

    // Falsifier Floor Bounds
    expect(screen.getByText('gross_margin')).toBeInTheDocument()
    expect(screen.getByText('BREAKS BUY CASE')).toBeInTheDocument()
    expect(screen.getByText('Floor: < 70.0 PERCENT (QUARTERLY)')).toBeInTheDocument()
    expect(
      screen.getByText('Margin below 70% breaks core pricing power premise.')
    ).toBeInTheDocument()

    // Catalyst Timeline
    expect(screen.getByText('Q3 Earnings Call')).toBeInTheDocument()
    expect(screen.getByText('2026-11-20')).toBeInTheDocument()
    expect(screen.getByText('Source: SEC 8-K')).toBeInTheDocument()
  })

  it('renders loading state when initialClaim is null', () => {
    render(<StructuredThesisCard ticker="NVDA" initialClaim={null} />)
    expect(screen.getByTestId('structured-thesis-card-loading')).toBeInTheDocument()
  })
})

