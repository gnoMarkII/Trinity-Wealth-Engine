import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { PositionReviewModal } from './PositionReviewModal'
import { api } from '../../api/client'
import type { PositionReviewResponseDTO } from '../../api/types'

vi.mock('../../api/client', () => ({
  api: {
    getPositionReview: vi.fn(),
  },
}))

describe('PositionReviewModal', () => {
  const mockReport: PositionReviewResponseDTO = {
    ticker: 'NVDA',
    company_name: 'NVIDIA Corporation',
    current_price: 135.5,
    as_of_date: '2026-10-23',
    action_stance: 'ACCUMULATE',
    action_stance_reason: 'Strong earnings growth exceeds multiple expansion',
    holding_summary: {
      is_held: true,
      units: 100,
      avg_cost: 110.0,
      unrealized_pnl_pct: 23.18,
      current_weight_pct: 8.5,
      bucket_id: 'bucket_growth',
      entry_date: '2025-06-15',
      added_direction: 'single_entry',
    },
    decompositions: [
      {
        horizon_label: '1Y',
        start_date: '2025-10-23',
        end_date: '2026-10-23',
        start_price: 90.0,
        end_price: 135.5,
        total_return_pct: 50.56,
        start_eps: 2.5,
        end_eps: 3.5,
        eps_growth_pct: 40.0,
        start_pe: 36.0,
        end_pe: 38.71,
        pe_expansion_pct: 7.53,
        primary_driver: 'EPS_GROWTH',
        diagnosis_tier: 'FUNDAMENTAL_DRIVEN',
        diagnosis_summary: 'ผลตอบแทนราคาขับเคลื่อนด้วยการเติบโตของกำไรเป็นหลัก (Healthy Fundamental Compounder)',
      },
    ],
    then_vs_now: {
      is_held: true,
      period_label: 'since_entry',
      then_date: '2025-06-15',
      now_date: '2026-10-23',
      then_price: 110.0,
      now_price: 135.5,
      then_pe: 35.0,
      now_pe: 38.71,
      then_growth_pct: 45.0,
      now_growth_pct: 40.0,
      what_was_true_then: 'Hopper data center transition surging.',
      what_is_true_now: 'Blackwell architecture volume ramp underway.',
      thesis_reflection_question: 'หากคุณไม่ได้ถือหุ้น NVDA ที่ต้นทุน $110.00 อยู่ในปัจจุบัน คุณจะยอมเปิดสถานะใหม่ที่ราคา $135.50 ในวันนี้หรือไม่?',
    },
    triggers: [
      {
        trigger_id: 'trg_falsifier_1',
        title: 'Gross Margin Compression below 70%',
        condition_text: 'Gross margin falls below 70.0% for 2 consecutive quarters',
        category: 'fundamental_floor',
        case_impact: 'BULL_CASE_DENTED',
        impact_description: 'Competitive moat eroding; trim position by 25%.',
        is_currently_triggered: false,
        metric_current_value: 74.8,
        metric_threshold_value: 70.0,
      },
      {
        trigger_id: 'trg_portfolio_weight',
        title: 'Portfolio Concentration Limit (15%)',
        condition_text: 'Holding weight exceeds 15.0% maximum allocation limit',
        category: 'portfolio_limit',
        case_impact: 'ALLOCATION_LIMIT_BREACHED',
        impact_description: 'Rebalance to trim down to target allocation.',
        is_currently_triggered: false,
        metric_current_value: 8.5,
        metric_threshold_value: 15.0,
      },
    ],
    honest_gaps: [
      'Decomposition uses Trailing 12-Month EPS rather than forward consensus EPS.',
    ],
  }

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getPositionReview).mockResolvedValue(mockReport)
  })

  it('does not render anything when isOpen is false', () => {
    const { container } = render(
      <PositionReviewModal isOpen={false} onClose={vi.fn()} ticker="NVDA" />
    )
    expect(container.firstChild).toBeNull()
  })

  it('fetches data and renders decomposition, then-vs-now, and triggers when open', async () => {
    render(<PositionReviewModal isOpen={true} onClose={vi.fn()} ticker="NVDA" />)

    expect(screen.getByText(/กำลังคำนวณการแยกตัวประกอบราคา/)).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByText('NVDA Position Review')).toBeInTheDocument()
    })

    // Verify Stance
    expect(screen.getByText('ACCUMULATE')).toBeInTheDocument()

    // Verify Holding Summary
    expect(screen.getByText('ถือครองอยู่ (Held)')).toBeInTheDocument()
    expect(screen.getByText('8.50%')).toBeInTheDocument()
    expect(screen.getAllByText('$110.00').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText('+23.18%')).toBeInTheDocument()

    // Verify Decomposition
    expect(screen.getByText('1. Price Move Decomposition')).toBeInTheDocument()
    expect(screen.getByText('FUNDAMENTAL DRIVEN')).toBeInTheDocument()
    expect(screen.getByText('+50.6%')).toBeInTheDocument()
    expect(screen.getByText('+40.0%')).toBeInTheDocument()
    expect(screen.getByText('+7.5%')).toBeInTheDocument()

    // Verify Then vs Now
    expect(screen.getByText('2. Historical Thesis Reconstruction (Then vs Now)')).toBeInTheDocument()
    expect(screen.getByText('Hopper data center transition surging.')).toBeInTheDocument()
    expect(screen.getByText('Blackwell architecture volume ramp underway.')).toBeInTheDocument()
    expect(screen.getByText(/Anti-Anchoring Reflection Question/)).toBeInTheDocument()

    // Verify Triggers Table
    expect(screen.getByText('3. Portfolio-Aware Decision Triggers (Worth Watching)')).toBeInTheDocument()
    expect(screen.getByText('Gross Margin Compression below 70%')).toBeInTheDocument()
    expect(screen.getByText('Bull Case Dented')).toBeInTheDocument()
  })

  it('triggers onClose when close button is clicked', async () => {
    const handleClose = vi.fn()
    render(<PositionReviewModal isOpen={true} onClose={handleClose} ticker="NVDA" />)

    await waitFor(() => {
      expect(screen.getByText('NVDA Position Review')).toBeInTheDocument()
    })

    const closeBtn = screen.getByRole('button', { name: 'Close' })
    fireEvent.click(closeBtn)

    expect(handleClose).toHaveBeenCalledTimes(1)
  })
})
