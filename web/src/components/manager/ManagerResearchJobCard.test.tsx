import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import ManagerResearchJobCard from './ManagerResearchJobCard'
import { api } from '../../api/client'

vi.mock('../../api/client', () => ({
  api: {
    getJobStatus: vi.fn(),
    getJobOutputs: vi.fn(),
    resumeJob: vi.fn(),
  },
  ApiError: class ApiError extends Error {},
}))

describe('ManagerResearchJobCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getJobStatus).mockRejectedValue(new Error('not mocked'))
  })

  it('renders compact active running indicator with pulse beacon and timer', () => {
    render(
      <ManagerResearchJobCard
        jobId="job_nvda_123"
        ticker="NVDA"
        jobType="equity_intel"
        status="queued"
      />
    )

    expect(screen.getByText(/AI กำลังวิเคราะห์หุ้น NVDA/i)).toBeInTheDocument()
    expect(screen.getByText(/อยู่ในคิว/i)).toBeInTheDocument()
    expect(screen.getByText(/ขั้นตอน:/i)).toBeInTheDocument()
    expect(screen.getByText(/0s/i)).toBeInTheDocument()
  })

  it('renders completed status and report view correctly', () => {
    render(
      <ManagerResearchJobCard
        jobId="job_macro_456"
        ticker="MACRO"
        jobType="macro_intel"
        status="completed"
      />
    )

    expect(screen.getByText(/รายงานผลการวิเคราะห์สภาวะเศรษฐกิจมหภาค/i)).toBeInTheDocument()
    expect(screen.getByText(/วิจัยเสร็จสมบูรณ์/i)).toBeInTheDocument()
    expect(screen.getByText(/job_macro_456/i)).toBeInTheDocument()
    expect(screen.getByText(/คัดลอก/i)).toBeInTheDocument()
  })

  it('renders pre-run approval buttons when awaiting_approval and resumes or cancels on click', async () => {
    vi.mocked(api.getJobStatus).mockResolvedValue({
      job_id: 'job_intc_pre',
      status: 'awaiting_approval',
      card_id: 'card_intc',
      error_message: null,
      current_node: null,
      interrupt_payload: {
        type: 'pre_run_approval',
        title: 'วิเคราะห์หุ้น INTC แบบละเอียด',
        instruction: 'วิเคราะห์หุ้น INTC แบบละเอียด',
        flow: 'manager',
        ticker: 'INTC',
      },
      log_count: 0,
      created_at: 1,
      updated_at: 1,
    })
    vi.mocked(api.resumeJob).mockResolvedValue({
      job_id: 'job_intc_pre',
      status: 'queued',
      card_id: 'card_intc',
      error_message: null,
      current_node: null,
      interrupt_payload: null,
      log_count: 0,
      created_at: 1,
      updated_at: 2,
    })

    render(
      <ManagerResearchJobCard
        jobId="job_intc_pre"
        ticker="INTC"
        jobType="equity_intel"
      />
    )

    await waitFor(() => {
      expect(screen.getByText(/รอยอมรับก่อนเริ่มรัน: วิเคราะห์หุ้น INTC/)).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: /✅ ยอมรับและเริ่มรัน/ }))

    await waitFor(() => {
      expect(api.resumeJob).toHaveBeenCalledWith('job_intc_pre', [], [], undefined, undefined, 'approve')
    })
  })

  it('transitions to cancelled state when user rejects pre-run approval', async () => {
    vi.mocked(api.getJobStatus).mockResolvedValue({
      job_id: 'job_intc_rej',
      status: 'awaiting_approval',
      card_id: 'card_intc_rej',
      error_message: null,
      current_node: null,
      interrupt_payload: {
        type: 'pre_run_approval',
        instruction: 'วิเคราะห์หุ้น INTC แบบละเอียด',
      },
      log_count: 0,
      created_at: 1,
      updated_at: 1,
    })
    vi.mocked(api.resumeJob).mockResolvedValue({
      job_id: 'job_intc_rej',
      status: 'error',
      card_id: 'card_intc_rej',
      error_message: 'ยกเลิกโดยผู้ใช้ก่อนเริ่มรัน',
      current_node: null,
      interrupt_payload: null,
      log_count: 0,
      created_at: 1,
      updated_at: 2,
    })

    render(
      <ManagerResearchJobCard
        jobId="job_intc_rej"
        ticker="INTC"
        jobType="equity_intel"
      />
    )

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /❌ ยกเลิก/ })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: /❌ ยกเลิก/ }))

    await waitFor(() => {
      expect(api.resumeJob).toHaveBeenCalledWith('job_intc_rej', [], [], undefined, undefined, 'reject')
      expect(screen.getByText(/ยกเลิกคำขอรันงานแล้ว/)).toBeInTheDocument()
    })
  })
})
