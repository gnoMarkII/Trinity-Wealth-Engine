import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { EarningsBridgeCard } from './EarningsBridgeCard'
import type { EarningsBridgeDTO } from '../../api/types'

const mockBridgeData: EarningsBridgeDTO = {
  ticker: 'GOOGL',
  period: 'Q3 2026',
  as_of_date: '2026-11-05',
  revenue_surprise: {
    metric_name: 'revenue',
    reported: '88270.00',
    consensus: '86300.00',
    delta: '1970.00',
    surprise_pct: '2.28',
    direction: 'BEAT',
    unit: 'USD_M',
  },
  eps_surprise: {
    metric_name: 'eps',
    reported: '2.12',
    consensus: '1.85',
    delta: '0.27',
    surprise_pct: '14.59',
    direction: 'BEAT',
    unit: 'USD',
  },
  ebit_margin_reported: '32.0',
  ebit_margin_consensus: '30.8',
  revenue_drivers: [
    {
      name: 'Cloud Infrastructure Scaling',
      category: 'VOLUME',
      impact_value: '1250.00',
      unit: 'USD_M',
      narrative: 'Strong workload expansion',
    },
    {
      name: 'Ad Pricing Mix',
      category: 'PRICE_MIX',
      impact_value: '820.00',
      unit: 'USD_M',
      narrative: 'Higher retail CPC',
    },
    {
      name: 'FX Currency Drag',
      category: 'FX',
      impact_value: '-100.00',
      unit: 'USD_M',
      narrative: 'Stronger USD',
    },
  ],
  sum_drivers: '1970.00',
  unallocated_delta: '0.00',
  is_mathematically_balanced: true,
  thesis_reconciliation_verdict: 'THESIS_CONFIRMED',
  thesis_reconciliation_narrative: 'Core cloud scale confirms investment thesis.',
}

describe('EarningsBridgeCard Component', () => {
  it('renders loading state when isLoading is true', () => {
    const { container } = render(<EarningsBridgeCard data={mockBridgeData} isLoading={true} />)
    expect(container.querySelector('.animate-pulse')).toBeInTheDocument()
  })

  it('renders surprise metrics and BEAT badges', () => {
    render(<EarningsBridgeCard data={mockBridgeData} />)
    expect(screen.getByText('Reported vs Expected Earnings Bridge')).toBeInTheDocument()
    expect(screen.getByText('+2.28% BEAT')).toBeInTheDocument()
    expect(screen.getByText('+14.59% BEAT')).toBeInTheDocument()
    expect(screen.getByText('88270.00')).toBeInTheDocument()
    expect(screen.getByText('2.12')).toBeInTheDocument()
  })

  it('renders exact balance mathematical invariant confirmation', () => {
    render(<EarningsBridgeCard data={mockBridgeData} />)
    expect(screen.getByText(/Mathematical Sum Invariant Verified/i)).toBeInTheDocument()
    expect(screen.getByText('EXACT BALANCE')).toBeInTheDocument()
  })

  it('renders unbalanced warning when delta does not equal driver sum', () => {
    const unbalancedData: EarningsBridgeDTO = {
      ...mockBridgeData,
      is_mathematically_balanced: false,
      unallocated_delta: '50.00',
    }
    render(<EarningsBridgeCard data={unbalancedData} />)
    expect(screen.getByText(/Unbalanced Causal Variance/i)).toBeInTheDocument()
    expect(screen.getByText('UNBALANCED')).toBeInTheDocument()
  })

  it('renders driver rows with correct categories and impact values', () => {
    render(<EarningsBridgeCard data={mockBridgeData} />)
    expect(screen.getByText('Cloud Infrastructure Scaling')).toBeInTheDocument()
    expect(screen.getByText('VOLUME')).toBeInTheDocument()
    expect(screen.getByText('+1250.00')).toBeInTheDocument()

    expect(screen.getByText('FX Currency Drag')).toBeInTheDocument()
    expect(screen.getByText('FX')).toBeInTheDocument()
    expect(screen.getByText('-100.00')).toBeInTheDocument()
  })

  it('renders thesis reconciliation verdict and narrative', () => {
    render(<EarningsBridgeCard data={mockBridgeData} />)
    expect(screen.getByText(/THESIS CONFIRMED/i)).toBeInTheDocument()
    expect(screen.getByText(/Core cloud scale confirms investment thesis/i)).toBeInTheDocument()
  })
})
