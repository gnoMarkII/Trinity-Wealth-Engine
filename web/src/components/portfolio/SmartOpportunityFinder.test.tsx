import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import SmartOpportunityFinder from './SmartOpportunityFinder'
import { api } from '../../api/client'
import type { PortfolioOpportunitiesResponseDTO } from '../../api/types'

vi.mock('../../api/client', () => ({
  api: {
    getPortfolioOpportunities: vi.fn(),
    upsertWatchlistItem: vi.fn(),
  },
}))

describe('SmartOpportunityFinder', () => {
  const mockResponse: PortfolioOpportunitiesResponseDTO = {
    portfolio_id: 'default',
    evaluation_id: 'opp_20261019_001',
    evaluated_at: '2026-10-19T10:00:00Z',
    readiness: 'ready',
    total_candidates_scanned: 2,
    passed_candidates_count: 2,
    warnings: ['Benchmark proxy used for TH equities'],
    missing_requirements: [],
    snapshot_meta: { nav_thb: 1000000.0 },
    candidates: [
      {
        candidate: {
          security_id: 'US:MSFT',
          issuer_id: 'MSFT',
          symbol: 'MSFT',
          market: 'US',
          currency: 'USD',
          asset_type: 'EQUITY',
          company_name: 'Microsoft Corp',
          sector_name: 'Information Technology',
          sector_name_th: 'เทคโนโลยี',
          source_pools: ['POOL_A_FUND_LOOK_THROUGH'],
          source_refs: ['fund:SCBS&P500E'],
          already_directly_held: false,
          already_watchlisted: false,
          current_direct_value_thb: 0,
          current_indirect_value_thb: 30000,
          current_total_exposure_thb: 30000,
          current_total_weight_pct: 3.0,
        },
        policy_fit: 'pass',
        policy_reasons: ['No exclusions'],
        suggested_bucket_id: 'CORE',
        applied_bucket_name: 'Core Growth',
        evidence_readiness: 'sufficient',
        composite_score: 9.0,
        valuation_upside_pct: 25.0,
        target_price: 500.0,
        target_price_basis: 'DCF',
        moat_rating: 'Wide',
        evidence_reasons: ['Wide moat'],
        macro_alignment: 'tailwind',
        macro_stage: 'Leading',
        macro_reasons: ['Tech leading'],
        capacity_status: 'available',
        remaining_capacity_thb: 120000,
        issuer_room_thb: 120000,
        sector_room_thb: 270000,
        capacity_reasons: ['Room available'],
        research_priority: 1,
        priority_reasons: ['High conviction'],
        is_actionable: true,
      },
      {
        candidate: {
          security_id: 'TH:CPALL',
          issuer_id: 'CPALL',
          symbol: 'CPALL',
          market: 'TH',
          currency: 'THB',
          asset_type: 'EQUITY',
          company_name: 'CP ALL PCL',
          sector_name: 'Consumer Staples',
          sector_name_th: 'สินค้าอุปโภคบริโภค',
          source_pools: ['POOL_B_EQUITY_SIDECAR'],
          source_refs: ['sidecar:CPALL'],
          already_directly_held: false,
          already_watchlisted: false,
          current_direct_value_thb: 0,
          current_indirect_value_thb: 0,
          current_total_exposure_thb: 0,
          current_total_weight_pct: 0,
        },
        policy_fit: 'pass',
        policy_reasons: ['Passes ROIC'],
        suggested_bucket_id: 'DIV',
        applied_bucket_name: 'Dividend Shield',
        evidence_readiness: 'sufficient',
        composite_score: 7.5,
        valuation_upside_pct: 15.0,
        target_price: 75.0,
        target_price_basis: 'PE Multiple',
        moat_rating: 'Narrow',
        evidence_reasons: ['Convenience store network'],
        macro_alignment: 'neutral',
        macro_stage: 'Improving',
        macro_reasons: ['Domestic recovery'],
        capacity_status: 'available',
        remaining_capacity_thb: 150000,
        issuer_room_thb: 150000,
        sector_room_thb: 300000,
        capacity_reasons: ['Room available'],
        research_priority: 2,
        priority_reasons: ['Secondary priority'],
        is_actionable: true,
      },
    ],
  }

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getPortfolioOpportunities).mockResolvedValue(mockResponse)
  })

  it('renders candidates and KPI summary when open', async () => {
    render(
      <SmartOpportunityFinder
        isOpen={true}
        portfolioId="default"
        onClose={vi.fn()}
      />
    )

    expect(screen.getByText(/3-Dimensional Smart Opportunity Finder/i)).toBeInTheDocument()

    await waitFor(() => {
      expect(api.getPortfolioOpportunities).toHaveBeenCalledWith('default')
      expect(screen.getByText('MSFT')).toBeInTheDocument()
      expect(screen.getByText('CPALL')).toBeInTheDocument()
    })

    expect(screen.getByText('READY')).toBeInTheDocument()
    expect(screen.getByText(/Benchmark proxy used for TH equities/i)).toBeInTheDocument()
  })

  it('filters candidates by market', async () => {
    render(
      <SmartOpportunityFinder
        isOpen={true}
        portfolioId="default"
        onClose={vi.fn()}
      />
    )

    await waitFor(() => {
      expect(screen.getByText('MSFT')).toBeInTheDocument()
      expect(screen.getByText('CPALL')).toBeInTheDocument()
    })

    const marketSelect = screen.getByDisplayValue('ตลาด: ทั้งหมด')
    fireEvent.change(marketSelect, { target: { value: 'US' } })

    expect(screen.getByText('MSFT')).toBeInTheDocument()
    expect(screen.queryByText('CPALL')).not.toBeInTheDocument()
  })

  it('filters candidates by search input', async () => {
    render(
      <SmartOpportunityFinder
        isOpen={true}
        portfolioId="default"
        onClose={vi.fn()}
      />
    )

    await waitFor(() => {
      expect(screen.getByText('MSFT')).toBeInTheDocument()
    })

    const searchInput = screen.getByPlaceholderText(/ค้นหา Symbol, ชื่อหุ้น หรือ Sector/i)
    fireEvent.change(searchInput, { target: { value: 'CPALL' } })

    expect(screen.queryByText('MSFT')).not.toBeInTheDocument()
    expect(screen.getByText('CPALL')).toBeInTheDocument()
  })

  it('opens WatchlistModal with prefilled draft when clicking Add to Watchlist', async () => {
    render(
      <SmartOpportunityFinder
        isOpen={true}
        portfolioId="default"
        onClose={vi.fn()}
      />
    )

    await waitFor(() => {
      expect(screen.getByText('MSFT')).toBeInTheDocument()
    })

    const addButtons = screen.getAllByRole('button', { name: /\+ Add to Watchlist/i })
    fireEvent.click(addButtons[0]!)

    // Watchlist modal should open with MSFT prefilled
    expect(screen.getByText('เพิ่มสินทรัพย์เฝ้าระวัง (Add to Watchlist)')).toBeInTheDocument()
    expect(screen.getByDisplayValue('MSFT')).toBeInTheDocument()
    expect(screen.getByDisplayValue('500')).toBeInTheDocument()
    expect(screen.getByDisplayValue('CORE')).toBeInTheDocument()
  })
})
