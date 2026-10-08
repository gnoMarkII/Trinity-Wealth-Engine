import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import InvestorPrinciplesTab from './InvestorPrinciplesTab'
import { api } from '../../../api/client'

vi.mock('../../../api/client', () => ({
  api: {
    getInterviewConfig: vi.fn(),
    getCurrentConfirmedEssence: vi.fn(),
    getCurrentEssenceSession: vi.fn(),
    getEssenceSession: vi.fn(),
    getEssenceSummary: vi.fn(),
    getFinancialContext: vi.fn(),
    getCurrentConfirmedAxis: vi.fn(),
    startEssenceSession: vi.fn(),
  },
}))

describe('InvestorPrinciplesTab', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getInterviewConfig).mockResolvedValue({
      interview_version: '1.0',
      prompt_version: '1.0',
      questions_count: 10,
      options_per_question: 4,
      topics: ['life_goals'],
    })
    vi.mocked(api.getCurrentConfirmedEssence).mockRejectedValue(new Error('Not found'))
    vi.mocked(api.getCurrentEssenceSession).mockRejectedValue(new Error('Not found'))
    vi.mocked(api.getFinancialContext).mockResolvedValue({
      snapshot_id: 'fc-1',
      portfolio_id: 'default',
      horizon_years: '5-10 ปี',
      unknown_fields: [],
      as_of: '2026-10-08',
      source: 'user_reported',
      readiness_issues: [],
      is_ready_for_numeric_policy: true,
    })
    vi.mocked(api.getCurrentConfirmedAxis).mockRejectedValue(new Error('Not found'))
  })

  it('renders 3 milestones in the top stepper and defaults to Milestone 1 welcome', async () => {
    render(<InvestorPrinciplesTab portfolioId="default" />)

    await waitFor(() => {
      expect(screen.getByText('ค้นหาแก่นแท้ (Essence)')).toBeInTheDocument()
      expect(screen.getByText('สร้างแกนหลัก (Axis)')).toBeInTheDocument()
      expect(screen.getByText('ออกแบบ Buckets')).toBeInTheDocument()
    })

    // Welcome card content
    expect(screen.getByText('ค้นหาแก่นแท้ในการลงทุนของคุณ')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /เริ่มค้นหาแก่นแท้/i })).toBeInTheDocument()
  })

  it('switches to Milestone 2 when Axis tab is clicked', async () => {
    render(<InvestorPrinciplesTab portfolioId="default" />)

    await waitFor(() => {
      expect(screen.getByText('สร้างแกนหลัก (Axis)')).toBeInTheDocument()
    })

    const axisTabBtn = screen.getByText('สร้างแกนหลัก (Axis)').closest('button')!
    fireEvent.click(axisTabBtn)

    // Should ask for confirmed essence first if not confirmed
    await waitFor(() => {
      expect(screen.getByText(/กรุณาค้นหาและยืนยันแก่นแท้การลงทุนก่อน/i)).toBeInTheDocument()
    })
  })

  it('switches to Milestone 3 when Buckets tab is clicked', async () => {
    render(<InvestorPrinciplesTab portfolioId="default" />)

    await waitFor(() => {
      expect(screen.getByText('ออกแบบ Buckets')).toBeInTheDocument()
    })

    const bucketsTabBtn = screen.getByText('ออกแบบ Buckets').closest('button')!
    fireEvent.click(bucketsTabBtn)

    // Should ask for confirmed axis first
    await waitFor(() => {
      expect(screen.getByText(/ต้องยืนยันแกนหลักการลงทุนของพอร์ตนี้ก่อน/i)).toBeInTheDocument()
    })
  })
})
