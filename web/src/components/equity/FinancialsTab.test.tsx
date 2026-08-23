import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { FinancialsTab } from './FinancialsTab'
import { api } from '../../api/client'
import type { FinancialStatementsDTO } from '../../api/types'

vi.mock('../../api/client', () => ({
  api: {
    getEquityFinancials: vi.fn(),
  },
  ApiError: class ApiError extends Error {},
}))

const mockFinancialsData: FinancialStatementsDTO = {
  schema_version: 6,
  ticker: 'AAPL',
  market: 'US',
  currency: 'USD',
  provider: 'edgartools',
  provider_symbol: 'AAPL',
  data_status: 'ok',
  coverage_status: 'complete',
  core_coverage_status: 'complete',
  expanded_coverage_status: 'complete',
  expanded_data_status: 'complete',
  core_coverage_pct: 100,
  expanded_coverage_pct: 100,
  missing_required_items: [],
  missing_expanded_items: [],
  validation_warnings: [],
  expanded_validation_warnings: [],
  expanded_error_count: 0,
  warnings: [],
  quarterly: [
    {
      statement_type: 'income',
      period_kind: 'duration',
      periods: [
        {
          period_key: '2024-Q3',
          fiscal_year: 2024,
          fiscal_quarter: 3,
          period_end_date: '2024-09-30',
          period_kind: 'duration',
          duration_days: 91,
          form_type: '10-Q',
          filing_url: 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/10q.htm',
          is_derived: false,
          items: {
            revenue: { value: 94930000000, yoy_growth_pct: 6.1, source_type: 'reported', is_derived: false },
            eps_diluted: { value: 0.97, yoy_growth_pct: 4.3, source_type: 'reported', is_derived: false },
          },
        },
        {
          period_key: '2024-Q4',
          fiscal_year: 2024,
          fiscal_quarter: 4,
          period_end_date: '2024-12-31',
          period_kind: 'duration',
          duration_days: 91,
          form_type: '10-K',
          filing_url: 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/10k.htm',
          is_derived: true,
          items: {
            revenue: { value: 124300000000, yoy_growth_pct: 7.2, source_type: 'derived', derivation: 'FY - Q1 - Q2 - Q3', is_derived: true },
            eps_diluted: { value: 1.64, yoy_growth_pct: 5.5, source_type: 'reported', source_concept: '8-K Exhibit 99.1 (Item 2.02)', is_derived: false },
          },
        },
      ],
      line_items: [
        {
          canonical_key: 'revenue',
          display_label: 'Revenue / Total Sales',
          unit_type: 'currency',
          is_primary_highlight: true,
        },
        {
          canonical_key: 'eps_diluted',
          display_label: 'Diluted EPS',
          unit_type: 'per_share',
          is_primary_highlight: true,
        },
      ],
    },
    {
      statement_type: 'balance_sheet',
      period_kind: 'instant',
      periods: [
        {
          period_key: '2024-Q3',
          fiscal_year: 2024,
          fiscal_quarter: 3,
          period_end_date: '2024-09-30',
          period_kind: 'instant',
          duration_days: null,
          form_type: '10-Q',
          filing_url: 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/10q.htm',
          is_derived: false,
          items: {
            total_assets: { value: 364980000000, yoy_growth_pct: 3.5, source_type: 'reported', is_derived: false },
          },
        },
        {
          period_key: '2024-Q4',
          fiscal_year: 2024,
          fiscal_quarter: 4,
          period_end_date: '2024-12-31',
          period_kind: 'instant',
          duration_days: null,
          form_type: '10-K',
          filing_url: 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/10k.htm',
          is_derived: false,
          items: {
            total_assets: { value: 380000000000, yoy_growth_pct: 4.1, source_type: 'reported', is_derived: false },
          },
        },
      ],
      line_items: [
        {
          canonical_key: 'total_assets',
          display_label: 'Total Assets',
          unit_type: 'currency',
          is_primary_highlight: true,
        },
      ],
    },
    {
      statement_type: 'cash_flow',
      period_kind: 'duration',
      periods: [
        {
          period_key: '2024-Q3',
          fiscal_year: 2024,
          fiscal_quarter: 3,
          period_end_date: '2024-09-30',
          period_kind: 'duration',
          duration_days: 91,
          form_type: '10-Q',
          filing_url: 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/10q.htm',
          is_derived: false,
          items: {
            operating_cash_flow: { value: 29500000000, yoy_growth_pct: 12.0, source_type: 'reported', is_derived: false },
            free_cash_flow: { value: 26800000000, yoy_growth_pct: 14.5, source_type: 'derived', is_derived: true },
          },
        },
      ],
      line_items: [
        {
          canonical_key: 'operating_cash_flow',
          display_label: 'Cash Flow from Operating Activities',
          unit_type: 'currency',
          is_primary_highlight: true,
        },
        {
          canonical_key: 'free_cash_flow',
          display_label: 'Free Cash Flow (FCF)',
          unit_type: 'currency',
          is_primary_highlight: true,
        },
      ],
    },
  ],
  annual: [],
  summary_chart_quarterly: [
    {
      period_key: '2024-Q3',
      date: '2024-09-30',
      revenue: 94930000000,
      gross_profit: 43879000000,
      operating_income: 29599000000,
      net_income: 22956000000,
      free_cash_flow: 26800000000,
      operating_margin_pct: 31.18,
      net_margin_pct: 24.18,
    },
  ],
  summary_chart_annual: [],
  ratios_quarterly: [
    {
      period_key: '2024-Q3',
      period_end_date: '2024-09-30',
      gross_margin_pct: 46.2,
      operating_margin_pct: 31.2,
      net_margin_pct: 24.2,
      fcf_margin_pct: 28.2,
      debt_to_equity: 1.45,
      current_ratio: 0.98,
    },
  ],
  ratios_annual: [],
  synced_at: '2026-08-22T08:00:00Z',
}

describe('FinancialsTab Component', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getEquityFinancials).mockResolvedValue(mockFinancialsData)
  })

  it('renders statement selector and income statement with cell-level derived markers', async () => {
    render(<FinancialsTab ticker="AAPL" market="US" />)

    await waitFor(() => {
      expect(screen.getByText('Income Statement')).toBeInTheDocument()
      expect(screen.getByText('Balance Sheet')).toBeInTheDocument()
      expect(screen.getByText('Cash Flow')).toBeInTheDocument()
      expect(screen.getByText('Key Ratios')).toBeInTheDocument()
    })

    expect(screen.getByText('Revenue / Total Sales')).toBeInTheDocument()
    expect(screen.getByText('Diluted EPS')).toBeInTheDocument()
    expect(screen.getAllByText('2024-Q3').length).toBeGreaterThan(0)
    expect(screen.getAllByText('2024-Q4').length).toBeGreaterThan(0)
    // Cell-level * badge for derived cell
    expect(screen.getByText('*')).toBeInTheDocument()
    // Data quality badge
    expect(screen.getByText('SEC Core Data Verified')).toBeInTheDocument()
  })

  it('formats per_share without scaling and formats currency with scale', async () => {
    render(<FinancialsTab ticker="AAPL" market="US" />)

    await waitFor(() => {
      expect(screen.getByText('Revenue / Total Sales')).toBeInTheDocument()
    })
    expect(screen.getByText('$0.97')).toBeInTheDocument()
    expect(screen.getAllByText('$94.93 B').length).toBeGreaterThan(0)
  })

  it('displays YoY growth badges', async () => {
    render(<FinancialsTab ticker="AAPL" market="US" />)

    await waitFor(() => {
      expect(screen.getByText('+6.1% YoY')).toBeInTheDocument()
      expect(screen.getByText('+4.3% YoY')).toBeInTheDocument()
    })
  })

  it('switches to Balance Sheet and Cash Flow', async () => {
    render(<FinancialsTab ticker="AAPL" market="US" />)

    await waitFor(() => {
      expect(screen.getByText('Revenue / Total Sales')).toBeInTheDocument()
    })

    // Click Balance Sheet
    fireEvent.click(screen.getByText('Balance Sheet'))
    expect(screen.getByText('Total Assets')).toBeInTheDocument()

    // Click Cash Flow
    fireEvent.click(screen.getByText('Cash Flow'))
    expect(screen.getByText('Cash Flow from Operating Activities')).toBeInTheDocument()
    expect(screen.getByText('Free Cash Flow (FCF)')).toBeInTheDocument()
  })

  it('switches to Key Ratios view', async () => {
    render(<FinancialsTab ticker="AAPL" market="US" />)

    await waitFor(() => {
      expect(screen.getByText('Income Statement')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByText('Key Ratios'))
    expect(screen.getByText('Financial & Operational Ratios')).toBeInTheDocument()
    expect(screen.getByText('Gross Margin (%)')).toBeInTheDocument()
    expect(screen.getByText('46.2%')).toBeInTheDocument()
    expect(screen.getByText('Current Ratio (Assets / Liab)')).toBeInTheDocument()
    expect(screen.getByText('0.98x')).toBeInTheDocument()
  })

  it('renders SEC EDGAR filing link when provider is edgartools', async () => {
    render(<FinancialsTab ticker="AAPL" market="US" />)

    await waitFor(() => {
      const secLinks = screen.getAllByRole('link', { name: /↗ 10-Q|↗ 10-K/i })
      expect(secLinks.length).toBeGreaterThan(0)
      expect(secLinks[0]).toHaveAttribute('href', 'https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/10q.htm')
      expect(secLinks[0]).toHaveAttribute('target', '_blank')
    })
  })

  it('renders fallback warning banner when data has warnings', async () => {
    const dataWithWarn: FinancialStatementsDTO = {
      ...mockFinancialsData,
      warnings: ['US fallback via yfinance', 'Historical periods partially available'],
    }
    vi.mocked(api.getEquityFinancials).mockResolvedValueOnce(dataWithWarn)

    render(<FinancialsTab ticker="AAPL" market="US" />)

    await waitFor(() => {
      expect(screen.getByText('US fallback via yfinance')).toBeInTheDocument()
      expect(screen.getByText('Historical periods partially available')).toBeInTheDocument()
    })
  })

  it('renders empty state when data_status is empty', async () => {
    const emptyData: FinancialStatementsDTO = {
      ...mockFinancialsData,
      data_status: 'empty',
      quarterly: [],
      annual: [],
    }
    vi.mocked(api.getEquityFinancials).mockResolvedValueOnce(emptyData)

    render(<FinancialsTab ticker="UNKNOWN" market="US" />)

    await waitFor(() => {
      expect(screen.getByText('No Financial Statements Available')).toBeInTheDocument()
    })
  })
})
