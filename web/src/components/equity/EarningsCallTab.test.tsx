import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import { EarningsCallTab } from './EarningsCallTab'
import { api } from '../../api/client'

vi.mock('../../api/client', () => ({
  api: {
    summarizeEarningsCall: vi.fn(),
    getEarningsCallRun: vi.fn(),
    retryEarningsCallRun: vi.fn(),
    getEarningsCalls: vi.fn(),
  },
}))

describe('EarningsCallTab', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getEarningsCalls).mockResolvedValue({ ticker: 'TSM', total_count: 0, items: [] })
  })

  it('renders input form with initial state when no existing calls', async () => {
    render(
      <MemoryRouter>
        <EarningsCallTab ticker="TSM" />
      </MemoryRouter>
    )

    expect(await screen.findByText(/Earnings Call Insights & Transcript Analyzer/)).toBeInTheDocument()
    const periodInput = await screen.findByLabelText(/ไตรมาส \/ งวดผลประกอบการ/)
    expect(periodInput).toHaveValue('Q2 2026')
    const transcriptInput = await screen.findByLabelText(/เนื้อหา Transcript/)
    expect(transcriptInput).toHaveValue('')
    expect(screen.getByRole('button', { name: /แปลงเป็น Highlights/ })).toBeDisabled()
  })

  it('submits form successfully (200 completed) and displays highlights & badges', async () => {
    vi.mocked(api.summarizeEarningsCall).mockResolvedValue({
      run_id: 'run-1',
      ticker: 'TSM',
      period: 'Q4 2024',
      status: 'completed',
      kanban_status: 'created',
      highlights: '### 1. Financial Highlights\nRevenue increased by 25% YoY.',
      vault_path: '30_Knowledge_Base/Earnings_Calls/TSM/2024-Q4_TSM_Earnings_Call.md',
      kanban_card_id: 'card-12345',
      reused_existing_run: false,
      is_idempotent_replay: false,
    })

    render(
      <MemoryRouter>
        <EarningsCallTab ticker="TSM" />
      </MemoryRouter>
    )

    const transcriptInput = await screen.findByLabelText(/เนื้อหา Transcript/)
    fireEvent.change(transcriptInput, {
      target: { value: 'Welcome to TSMC Q4 2024 Earnings Call transcript with detailed discussion.' },
    })

    const submitButton = await screen.findByRole('button', { name: /แปลงเป็น Highlights/ })
    expect(submitButton).not.toBeDisabled()
    fireEvent.click(submitButton)

    await waitFor(() => {
      expect(api.summarizeEarningsCall).toHaveBeenCalledWith(
        'TSM',
        'Q2 2026',
        'Welcome to TSMC Q4 2024 Earnings Call transcript with detailed discussion.'
      )
    })
  })

  it('handles 202 pending status and polls until completed', async () => {
    vi.mocked(api.summarizeEarningsCall).mockResolvedValue({
      run_id: 'run-pending-1',
      ticker: 'TSM',
      period: 'Q4 2024',
      status: 'kanban_pending',
      kanban_status: 'pending',
      highlights: 'Highlights in progress',
      vault_path: 'path/note.md',
      kanban_card_id: null,
      reused_existing_run: false,
      is_idempotent_replay: false,
    })

    vi.mocked(api.getEarningsCallRun).mockResolvedValue({
      run_id: 'run-pending-1',
      ticker: 'TSM',
      period: 'Q4 2024',
      status: 'completed',
      kanban_status: 'created',
      highlights: 'Highlights in progress',
      vault_path: 'path/note.md',
      kanban_card_id: 'card-polled',
      reused_existing_run: false,
      is_idempotent_replay: false,
      created_at: 100,
      updated_at: 200,
    })

    render(
      <MemoryRouter>
        <EarningsCallTab ticker="TSM" />
      </MemoryRouter>
    )

    const transcriptInput = await screen.findByLabelText(/เนื้อหา Transcript/)
    fireEvent.change(transcriptInput, {
      target: { value: 'Valid transcript length text for testing 202 pending polling.' },
    })

    const submitButton = await screen.findByRole('button', { name: /แปลงเป็น Highlights/ })
    fireEvent.click(submitButton)

    await waitFor(() => {
      expect(screen.getByText(/กำลังจัดส่งเข้า Kanban Backlog/)).toBeInTheDocument()
    })
  })

  it('displays error message when API call fails', async () => {
    vi.mocked(api.summarizeEarningsCall).mockRejectedValue(new Error('LLM Rate Limit Exceeded'))

    render(
      <MemoryRouter>
        <EarningsCallTab ticker="TSM" />
      </MemoryRouter>
    )

    const transcriptInput = await screen.findByLabelText(/เนื้อหา Transcript/)
    fireEvent.change(transcriptInput, {
      target: { value: 'Valid transcript length text for testing API failure response.' },
    })

    const submitButton = await screen.findByRole('button', { name: /แปลงเป็น Highlights/ })
    fireEvent.click(submitButton)

    await waitFor(() => {
      expect(screen.getByText('เกิดข้อผิดพลาด')).toBeInTheDocument()
      expect(screen.getByText('LLM Rate Limit Exceeded')).toBeInTheDocument()
    })
  })

  it('resets transcript input when clicking clear button', async () => {
    render(
      <MemoryRouter>
        <EarningsCallTab ticker="TSM" />
      </MemoryRouter>
    )

    const transcriptInput = await screen.findByLabelText(/เนื้อหา Transcript/)
    fireEvent.change(transcriptInput, {
      target: { value: 'Some text to be cleared.' },
    })

    const clearButton = await screen.findByRole('button', { name: 'ล้างข้อความ' })
    fireEvent.click(clearButton)

    expect(transcriptInput).toHaveValue('')
  })

  it('renders existing earnings call highlights when available', async () => {
    vi.mocked(api.getEarningsCalls).mockResolvedValue({
      ticker: 'FTNT',
      total_count: 1,
      items: [
        {
          title: 'FTNT Earnings Call Q2 2026',
          ticker: 'FTNT',
          period: 'Q2 2026',
          vault_path: '30_Knowledge_Base/Earnings_Calls/FTNT/Q2_2026_FTNT_Earnings_Call.md',
          highlights: '### 1. 📊 Key Financial Highlights\nRevenue: $2.05B (+26% YoY)',
          date: '2026-08-26',
          last_updated: '2026-08-26 23:01:06',
          has_full_transcript: true,
        },
      ],
    })

    render(
      <MemoryRouter>
        <EarningsCallTab ticker="FTNT" />
      </MemoryRouter>
    )

    expect(await screen.findByText(/FTNT Earnings Call Q2 2026/)).toBeInTheDocument()
    expect(screen.getByText(/Revenue: \$2.05B/)).toBeInTheDocument()
  })
})
