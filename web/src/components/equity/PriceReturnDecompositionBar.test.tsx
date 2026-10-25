import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import PriceReturnDecompositionBar from './PriceReturnDecompositionBar'
import type { PriceMoveDecompositionDTO } from '../../api/types'

describe('PriceReturnDecompositionBar', () => {
  const mockDecomp: PriceMoveDecompositionDTO = {
    horizon_label: '1 Year',
    start_date: '2023-10-01',
    end_date: '2024-10-01',
    start_price: 44.5,
    end_price: 67.05,
    total_return_pct: 50.6,
    start_eps: 1.25,
    end_eps: 1.75,
    eps_growth_pct: 40.0,
    start_pe: 35.6,
    end_pe: 38.3,
    pe_expansion_pct: 7.5,
    primary_driver: 'earnings',
    diagnosis_tier: 'fundamental_driven_compounder',
    diagnosis_summary: 'Share price appreciation was primarily driven by strong core earnings growth.',
  }

  it('renders all 3 metric tiers with correct signs', () => {
    render(<PriceReturnDecompositionBar decomposition={mockDecomp} />)

    expect(screen.getByTestId('price-return-decomposition-bar')).toBeInTheDocument()
    expect(screen.getByText('+50.6%')).toBeInTheDocument()
    expect(screen.getByText('+40.0%')).toBeInTheDocument()
    expect(screen.getByText('+7.5%')).toBeInTheDocument()
    expect(screen.getByText('FUNDAMENTAL DRIVEN COMPOUNDER')).toBeInTheDocument()
    expect(
      screen.getByText('Share price appreciation was primarily driven by strong core earnings growth.')
    ).toBeInTheDocument()
  })
})
