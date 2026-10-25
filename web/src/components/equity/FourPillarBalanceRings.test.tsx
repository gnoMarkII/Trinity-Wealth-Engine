import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import FourPillarBalanceRings from './FourPillarBalanceRings'

describe('FourPillarBalanceRings', () => {
  it('renders all four pillars with scores and weights', () => {
    render(
      <FourPillarBalanceRings
        fundamentalScore={74.3}
        guidanceScore={35.0}
        valuationScore={72.5}
        executionReadiness={4.3}
        isReweighted={false}
      />
    )

    expect(screen.getByTestId('four-pillar-rings')).toBeInTheDocument()
    expect(screen.getByText('74.3')).toBeInTheDocument()
    expect(screen.getByText('35.0')).toBeInTheDocument()
    expect(screen.getByText('72.5')).toBeInTheDocument()
    expect(screen.getByText('43')).toBeInTheDocument() // 4.3 * 10
    expect(screen.getByText('40% Wt')).toBeInTheDocument()
    expect(screen.getAllByText('30% Wt')).toHaveLength(2)
  })

  it('handles reweighted mode where valuation is excluded', () => {
    render(
      <FourPillarBalanceRings
        fundamentalScore={74.3}
        guidanceScore={35.0}
        valuationScore={72.5}
        executionReadiness={4.3}
        isReweighted={true}
        reweightingMetadata={{
          excluded_pillars: ['valuation'],
          reason: 'Macro Anomaly ERP <= 0',
        }}
      />
    )

    expect(screen.getByText('55% Wt (Reweighted)')).toBeInTheDocument()
    expect(screen.getByText('45% Wt (Reweighted)')).toBeInTheDocument()
    expect(screen.getByText('0% (Excluded)')).toBeInTheDocument()
    expect(screen.getByText('Info')).toBeInTheDocument()
  })
})
