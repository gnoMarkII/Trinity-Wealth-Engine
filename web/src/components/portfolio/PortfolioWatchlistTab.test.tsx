import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import PortfolioWatchlistTab from './PortfolioWatchlistTab'
import { api } from '../../api/client'
import type { ActualWatchlistItemDTO } from '../../api/types'

vi.mock('../../api/client', () => ({
  api: {
    getPortfolioOpportunities: vi.fn(),
    removeWatchlistItem: vi.fn(),
    upsertWatchlistItem: vi.fn(),
  },
}))

describe('PortfolioWatchlistTab', () => {
  const mockItems: ActualWatchlistItemDTO[] = [
    {
      symbol: 'MSFT',
      asset_type: 'Stock',
      target_price: 450.0,
      added_date: '2026-10-15',
      notes: 'Leading tech AI proxy',
      suggested_bucket_id: 'CORE',
      evaluation_ref: 'eval_123',
    },
    {
      symbol: 'TLWorld',
      asset_type: 'ETF',
      target_price: null,
      added_date: '2026-10-18',
      notes: 'Global equity diversifier',
      suggested_bucket_id: null,
    },
  ]

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getPortfolioOpportunities).mockResolvedValue({
      portfolio_id: 'default',
      evaluation_id: 'test_eval',
      evaluated_at: '2026-10-19T10:00:00Z',
      readiness: 'ready',
      total_candidates_scanned: 0,
      passed_candidates_count: 0,
      candidates: [],
      warnings: [],
      missing_requirements: [],
      snapshot_meta: {},
    })
  })

  it('renders watchlist table with Strategy Bucket and target prices', () => {
    render(
      <PortfolioWatchlistTab
        portfolioId="default"
        items={mockItems}
        lastUpdated="2026-10-19T10:00:00Z"
        onSuccess={vi.fn()}
      />
    )

    expect(screen.getByText('Watchlist (2 รายการ)')).toBeInTheDocument()
    expect(screen.getByText('MSFT')).toBeInTheDocument()
    expect(screen.getByText('CORE')).toBeInTheDocument()
    expect(screen.getByText('450.00')).toBeInTheDocument()
    expect(screen.getByText('Leading tech AI proxy')).toBeInTheDocument()

    expect(screen.getByText('TLWorld')).toBeInTheDocument()
    expect(screen.getByText('N/A')).toBeInTheDocument()
  })

  it('opens 3D Smart Opportunity Finder drawer on button click', async () => {
    render(
      <PortfolioWatchlistTab
        portfolioId="default"
        items={mockItems}
        lastUpdated="2026-10-19T10:00:00Z"
        onSuccess={vi.fn()}
      />
    )

    const finderBtn = screen.getByRole('button', { name: /3D Opportunity Finder/i })
    fireEvent.click(finderBtn)

    await waitFor(() => {
      expect(screen.getByText(/3-Dimensional Smart Opportunity Finder/i)).toBeInTheDocument()
    })
  })

  it('opens add watchlist modal on button click', () => {
    render(
      <PortfolioWatchlistTab
        portfolioId="default"
        items={mockItems}
        lastUpdated="2026-10-19T10:00:00Z"
        onSuccess={vi.fn()}
      />
    )

    const addBtn = screen.getByRole('button', { name: /\+ Add to Watchlist/i })
    fireEvent.click(addBtn)

    expect(screen.getByText('เพิ่มสินทรัพย์เฝ้าระวัง (Add to Watchlist)')).toBeInTheDocument()
  })
})
