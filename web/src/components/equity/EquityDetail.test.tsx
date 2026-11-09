import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { EquityDetail } from './EquityDetail'
import { mockEquityDetailAAPL } from '../../mocks/equity'
import { api } from '../../api/client'

vi.mock('../../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/client')>()
  return {
    ...actual,
    api: {
      ...actual.api,
      getEarningsCalls: vi.fn().mockResolvedValue({ items: [] }),
      getLatestEquityNotebookLMExport: vi.fn().mockResolvedValue(null),
      getVariantPerception: vi.fn().mockResolvedValue(null),
      getPositionReview: vi.fn().mockResolvedValue(null),
      getActiveThesis: vi.fn().mockResolvedValue(null),
      getNasdaqConsensus: vi.fn().mockResolvedValue(null),
      getOptionsChain: vi.fn().mockResolvedValue(null),
      getOptionsMaxPain: vi.fn().mockResolvedValue(null),
      getEquityNews: vi.fn().mockResolvedValue([]),
      getEquityOHLCV: vi.fn().mockResolvedValue({ items: [] }),
    },
  }
})

describe('EquityDetail', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getEarningsCalls).mockResolvedValue({ items: [] } as any)
    vi.mocked(api.getLatestEquityNotebookLMExport).mockResolvedValue(null)
    vi.mocked(api.getVariantPerception).mockResolvedValue(null as any)
    vi.mocked(api.getPositionReview).mockResolvedValue(null as any)
    vi.mocked(api.getActiveThesis).mockResolvedValue(null)
    vi.mocked(api.getNasdaqConsensus).mockResolvedValue(null as any)
    vi.mocked(api.getOptionsChain).mockResolvedValue(null as any)
    vi.mocked(api.getOptionsMaxPain).mockResolvedValue(null as any)
  })

  it('renders loading state', () => {
    render(<EquityDetail status="loading" />)
    expect(screen.getByText(/กำลังโหลด/)).toBeInTheDocument()
  })

  it('renders not-found state', () => {
    render(<EquityDetail status="not-found" />)
    expect(screen.getByText('ไม่พบข้อมูล')).toBeInTheDocument()
    expect(screen.getByText(/ยังไม่มีการวิเคราะห์สำหรับหุ้นตัวนี้/)).toBeInTheDocument()
  })

  it('renders error state with default message', () => {
    render(<EquityDetail status="error" />)
    expect(screen.getByText('เกิดข้อผิดพลาดในการโหลดข้อมูล')).toBeInTheDocument()
    expect(screen.getByText('ไม่สามารถติดต่อ Backend API ได้')).toBeInTheDocument()
  })

  it('renders error state with custom message', () => {
    render(<EquityDetail status="error" errorMessage="Network Error" />)
    expect(screen.getByText('Network Error')).toBeInTheDocument()
  })

  it('renders detail view successfully', () => {
    render(<EquityDetail status="success" data={mockEquityDetailAAPL} />)
    
    expect(screen.getByText('AAPL')).toBeInTheDocument()
    expect(screen.getByText('(US)')).toBeInTheDocument()
    expect(screen.getByText('Apple Inc.')).toBeInTheDocument()
    
    expect(screen.getAllByText(/Base Case/).length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText(mockEquityDetailAAPL.base_case_summary)).toBeInTheDocument()
    
    expect(screen.getByText('bullish')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '📈 Chart' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '📑 Financials' })).toBeInTheDocument()
  })

  it('renders InstitutionalThesisCard, VariantPerceptionCard, and opens PositionReviewModal on click', async () => {
    vi.mocked(api.getVariantPerception).mockResolvedValue({
      ticker: 'AAPL',
      period: '2026Q3',
      metric_name: 'services_revenue',
      reported_value: '25000.00',
      consensus_value: '24000.00',
      fund_thesis_value: '24900.00',
      street_surprise: '1000.00',
      street_surprise_pct: '4.17',
      variant_spread: '900.00',
      variant_spread_pct: '3.75',
      variant_realization: '100.00',
      variant_realization_pct: '0.40',
      alpha_verdict: 'ALPHA_CONFIRMED',
      unit: 'USD_M',
    })

    vi.mocked(api.getPositionReview).mockResolvedValue({
      ticker: 'AAPL',
      company_name: 'Apple Inc.',
      current_price: 210.0,
      as_of_date: '2026-08-03',
      action_stance: 'ACCUMULATE',
      action_stance_reason: 'Services compounding justifies valuation.',
      holding_summary: {
        is_held: true,
        units: 10,
        avg_cost: 150.0,
        unrealized_pnl_pct: 40.0,
        current_weight_pct: 12.0,
        bucket_id: 'CORE',
        entry_date: '2025-01-10',
        added_direction: 'single_entry',
      },
      decompositions: [
        {
          horizon_label: '1Y',
          start_date: '2025-08-03',
          end_date: '2026-08-03',
          start_price: 170.0,
          end_price: 210.0,
          total_return_pct: 23.53,
          start_eps: 6.0,
          end_eps: 7.2,
          eps_growth_pct: 20.0,
          start_pe: 28.3,
          end_pe: 29.17,
          pe_expansion_pct: 3.07,
          primary_driver: 'EPS_GROWTH',
          diagnosis_tier: 'FUNDAMENTAL_DRIVEN',
          diagnosis_summary: 'Driven by EPS growth',
        },
      ],
      then_vs_now: {
        is_held: true,
        period_label: 'since_entry',
        then_date: '2025-01-10',
        now_date: '2026-08-03',
        then_price: 150.0,
        now_price: 210.0,
        then_pe: 25.0,
        now_pe: 29.17,
        then_growth_pct: 12.0,
        now_growth_pct: 15.0,
        what_was_true_then: 'Initial hardware moat thesis.',
        what_is_true_now: 'Services flywheel accelerating.',
        thesis_reflection_question: 'Would you buy AAPL at $210 today?',
      },
      triggers: [],
      honest_gaps: [],
    })

    const user = userEvent.setup()
    render(<EquityDetail status="success" data={mockEquityDetailAAPL} />)

    expect(screen.getByText('4-Pillars Investment Committee Engine')).toBeInTheDocument()
    expect(screen.getAllByText('ACCUMULATE NOW').length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('SEC Form 4 Conviction')).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByTestId('variant-perception-card')).toBeInTheDocument()
      expect(screen.getByText('ALPHA CONFIRMED')).toBeInTheDocument()
    })

    const reviewBtn = screen.getByRole('button', { name: /เปิด Position Review \(Full View\)/ })
    await user.click(reviewBtn)

    await waitFor(() => {
      expect(screen.getByText('AAPL Position Review')).toBeInTheDocument()
      expect(screen.getByText('1. Price Move Decomposition')).toBeInTheDocument()
    })
  })
})
