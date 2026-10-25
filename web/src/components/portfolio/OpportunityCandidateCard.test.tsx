import { render, screen, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import OpportunityCandidateCard from './OpportunityCandidateCard'
import type { CandidateEvaluationResultDTO } from '../../api/types'

describe('OpportunityCandidateCard', () => {
  const mockResult: CandidateEvaluationResultDTO = {
    candidate: {
      security_id: 'US:MSFT',
      issuer_id: 'MSFT',
      symbol: 'MSFT',
      market: 'US',
      currency: 'USD',
      asset_type: 'EQUITY',
      company_name: 'Microsoft Corporation',
      sector_name: 'Information Technology',
      sector_name_th: 'เทคโนโลยีสารสนเทศ',
      source_pools: ['POOL_A_FUND_LOOK_THROUGH', 'POOL_B_EQUITY_SIDECAR'],
      source_refs: ['fund:SCBS&P500E', 'sidecar:MSFT'],
      already_directly_held: false,
      already_watchlisted: false,
      current_direct_value_thb: 0.0,
      current_indirect_value_thb: 45000.0,
      current_total_exposure_thb: 45000.0,
      current_total_weight_pct: 4.5,
    },
    policy_fit: 'pass',
    policy_reasons: ['Complies with ROIC >= 15% rule', 'No axis exclusions'],
    suggested_bucket_id: 'CORE',
    applied_bucket_name: 'Core Growth',
    evidence_readiness: 'sufficient',
    composite_score: 85.0,
    valuation_upside_pct: 22.4,
    target_price: 520.0,
    target_price_basis: 'Consensus DCF',
    moat_rating: 'Wide',
    evidence_reasons: ['Wide moat rating confirmed'],
    macro_alignment: 'tailwind',
    macro_stage: 'Leading',
    macro_reasons: ['Tech sector in Leading quadrant'],
    capacity_status: 'available',
    remaining_capacity_thb: 105000.0,
    issuer_room_thb: 105000.0,
    sector_room_thb: 255000.0,
    capacity_reasons: ['Single stock cap room: 10.5%'],
    research_priority: 1,
    priority_reasons: ['Top priority conviction candidate'],
    is_actionable: true,
  }

  it('renders symbol, company name, sector, and 3D badges correctly', () => {
    const onAdd = vi.fn()
    render(<OpportunityCandidateCard result={mockResult} onAddToWatchlist={onAdd} />)

    expect(screen.getByText('MSFT')).toBeInTheDocument()
    expect(screen.getByText('Microsoft Corporation')).toBeInTheDocument()
    expect(screen.getByText('เทคโนโลยีสารสนเทศ')).toBeInTheDocument()
    expect(screen.getByText(/P1 Top Priority/i)).toBeInTheDocument()
    expect(screen.getByText(/Tailwind \(Leading\)/i)).toBeInTheDocument()
    expect(screen.getByText(/Axis: Pass/i)).toBeInTheDocument()
    expect(screen.getByText(/Room: ฿105,000/i)).toBeInTheDocument()
    expect(screen.getByText('85.0 / 100')).toBeInTheDocument()
    expect(screen.getByText('+22.4%')).toBeInTheDocument()
    expect(screen.getByText('$520.00')).toBeInTheDocument()
    expect(screen.getByText('Wide')).toBeInTheDocument()
  })

  it('toggles expandable evaluation details and reasons', () => {
    const onAdd = vi.fn()
    render(<OpportunityCandidateCard result={mockResult} onAddToWatchlist={onAdd} />)

    const toggleBtn = screen.getByText(/ดูรายละเอียดเหตุผล & หลักฐานประเมิน 3 มิติ/i)
    fireEvent.click(toggleBtn)

    expect(screen.getByText(/Complies with ROIC >= 15% rule/i)).toBeInTheDocument()
    expect(screen.getByText(/Tech sector in Leading quadrant/i)).toBeInTheDocument()
    expect(screen.getByText(/Single stock cap room: 10.5%/i)).toBeInTheDocument()

    // Collapse
    fireEvent.click(screen.getByText(/ซ่อนรายละเอียดการวิเคราะห์/i))
    expect(screen.queryByText(/Complies with ROIC >= 15% rule/i)).not.toBeInTheDocument()
  })

  it('calls onAddToWatchlist when button clicked', () => {
    const onAdd = vi.fn()
    render(<OpportunityCandidateCard result={mockResult} onAddToWatchlist={onAdd} />)

    const addBtn = screen.getByRole('button', { name: /\+ Add to Watchlist/i })
    fireEvent.click(addBtn)

    expect(onAdd).toHaveBeenCalledTimes(1)
    expect(onAdd).toHaveBeenCalledWith(mockResult)
  })

  it('renders badges for already directly held or in watchlist', () => {
    const heldResult: CandidateEvaluationResultDTO = {
      ...mockResult,
      candidate: {
        ...mockResult.candidate,
        already_directly_held: true,
        already_watchlisted: true,
      },
    }
    render(<OpportunityCandidateCard result={heldResult} onAddToWatchlist={vi.fn()} />)

    expect(screen.getByText('✓ ถือตรงแล้ว')).toBeInTheDocument()
    expect(screen.getByText('★ ใน Watchlist')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /\+ Add to Watchlist/i })).not.toBeInTheDocument()
  })
})
