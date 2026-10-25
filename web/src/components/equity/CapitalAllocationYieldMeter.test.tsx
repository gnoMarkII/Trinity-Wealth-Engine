import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import CapitalAllocationYieldMeter from './CapitalAllocationYieldMeter'

describe('CapitalAllocationYieldMeter', () => {
  it('renders zero dividend company with buyback yield and net shareholder yield', () => {
    render(
      <CapitalAllocationYieldMeter
        dividendYieldPct={0}
        buybackYieldPct={2.1}
        netShareholderYieldPct={-0.96}
        fcfYieldPct={3.8}
      />
    )

    expect(screen.getByTestId('capital-allocation-yield-meter')).toBeInTheDocument()
    expect(screen.getByText('0.00%')).toBeInTheDocument()
    expect(screen.getByText('+2.10%')).toBeInTheDocument()
    expect(screen.getByText('-0.96%')).toBeInTheDocument()
    expect(screen.getByText('Zero Dividend (Pure Reinvestment / Buybacks)')).toBeInTheDocument()
    expect(screen.getByText('FCF Yield: 3.8%')).toBeInTheDocument()
  })
})
