import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import PositionReviewCard from './PositionReviewCard'
import type { PositionReviewResponseDTO } from '../../api/types'

describe('PositionReviewCard', () => {
  const mockReviewHeld: PositionReviewResponseDTO = {
    ticker: 'NVDA',
    company_name: 'NVIDIA Corporation',
    current_price: 130.50,
    as_of_date: '2026-10-30',
    action_stance: 'ACCUMULATE_BULLISH',
    action_stance_reason: 'Strong AI datacenter demand and solid EPS growth',
    holding_summary: {
      is_held: true,
      units: 50,
      avg_cost: 110.00,
      unrealized_pnl_pct: 18.64,
      current_weight_pct: 7.85,
      bucket_id: 'bucket_growth',
      entry_date: '2026-02-15',
      added_direction: 'BUY',
    },
    decompositions: [
      {
        horizon_label: '1Y',
        start_date: '2025-10-30',
        end_date: '2026-10-30',
        start_price: 95.0,
        end_price: 130.5,
        total_return_pct: 37.37,
        start_eps: 2.10,
        end_eps: 3.20,
        eps_growth_pct: 52.38,
        start_pe: 45.2,
        end_pe: 40.8,
        pe_expansion_pct: -9.73,
        primary_driver: 'eps_growth',
        diagnosis_tier: 'fundamental_compounder',
        diagnosis_summary: 'การเติบโตของราคาขับเคลื่อนด้วยกำไรจริง (Earnings Growth)',
      },
    ],
    then_vs_now: {
      is_held: true,
      period_label: '1Y Ago',
      then_date: '2025-10-30',
      now_date: '2026-10-30',
      then_price: 95.0,
      now_price: 130.5,
      then_pe: 45.2,
      now_pe: 40.8,
      then_growth_pct: 45.0,
      now_growth_pct: 52.0,
      what_was_true_then: 'คาดการณ์ว่า Blackwell architecture จะช่วยหนุนยอดขาย',
      what_is_true_now: 'ยอดขาย Blackwell เติบโตทำสถิติสูงสุดใหม่',
      thesis_reflection_question: 'หากวันนี้ไม่มีหุ้น NVDA คุณจะยังซื้อที่ราคานี้หรือไม่?',
    },
    triggers: [
      {
        trigger_id: 'TRIG_1',
        title: 'Margin Compression Alert',
        condition_text: 'Gross margin falls below 72%',
        category: 'falsifier',
        case_impact: 'TRIM_ON_BOUNCE',
        impact_description: 'ลดสัดส่วนหากกำไรขั้นต้นชะลอตัว',
        is_currently_triggered: false,
        metric_current_value: 75.2,
        metric_threshold_value: 72.0,
      },
    ],
    honest_gaps: [],
  }

  const mockReviewNotHeld: PositionReviewResponseDTO = {
    ...mockReviewHeld,
    ticker: 'GOOGL',
    holding_summary: {
      is_held: false,
      units: 0,
      avg_cost: null,
      unrealized_pnl_pct: null,
      current_weight_pct: 0,
      bucket_id: null,
      entry_date: null,
      added_direction: null,
    },
  }

  it('renders held position summary, decomposition, and then vs now cleanly', () => {
    render(<PositionReviewCard ticker="NVDA" initialData={mockReviewHeld} />)

    expect(screen.getByTestId('position-review-card')).toBeInTheDocument()
    expect(screen.getByText('💼 ทบทวนสถานะพอร์ตและสมมติฐานการลงทุน (Position Review)')).toBeInTheDocument()
    expect(screen.getByText('50 หุ้น')).toBeInTheDocument()
    expect(screen.getByText('$110.00')).toBeInTheDocument()
    expect(screen.getByText('+18.64%')).toBeInTheDocument()
    expect(screen.getByText('7.85%')).toBeInTheDocument()
    expect(screen.getByText('หากวันนี้ไม่มีหุ้น NVDA คุณจะยังซื้อที่ราคานี้หรือไม่?')).toBeInTheDocument()
    expect(screen.getByText('Margin Compression Alert')).toBeInTheDocument()
  })

  it('renders watchlist simulation banner when stock is not held in portfolio', () => {
    render(<PositionReviewCard ticker="GOOGL" initialData={mockReviewNotHeld} />)

    expect(screen.getByText(/โหมดวิเคราะห์จำลอง \(Simulation & Watchlist Mode\)/)).toBeInTheDocument()
    expect(screen.getByText(/คุณยังไม่มีหุ้น GOOGL ในพอร์ต/)).toBeInTheDocument()
  })
})

