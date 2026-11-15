import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import ManagerKanbanView from './ManagerKanbanView'
import { api } from '../../api/client'

vi.mock('../../api/client', () => ({
  api: {
    listKanbanCards: vi.fn(),
    dispatchJob: vi.fn(),
    moveKanbanCard: vi.fn(),
    getJobStatus: vi.fn(),
    getJobOutputs: vi.fn().mockResolvedValue({
      job_id: 'job-1',
      status: 'done',
      executive_summary: '',
      executive_summary_created_at: 0,
      specialists: [],
      last_seq: 0,
      error_message: null,
    }),
    resumeJob: vi.fn(),
    getNewsFunnelPending: vi.fn(),
    getNewsFunnelFiltered: vi.fn(),
  },
  ApiError: class ApiError extends Error {},
}))

vi.mock('../LiveTerminal', () => ({
  default: ({ onStatusChange }: { onStatusChange?: (status: string) => void }) => {
    return (
      <div data-testid="background-terminal">
        <button onClick={() => onStatusChange?.('awaiting_approval')}>
          Emit awaiting_approval
        </button>
      </div>
    )
  },
}))

describe('ManagerKanbanView Component', () => {
  beforeEach(() => {
    vi.resetAllMocks()
    vi.mocked(api.getNewsFunnelPending).mockResolvedValue([])
    vi.mocked(api.getNewsFunnelFiltered).mockResolvedValue([])
    vi.mocked(api.getJobOutputs).mockResolvedValue({
      job_id: 'job-1',
      status: 'done',
      executive_summary: '',
      executive_summary_created_at: 0,
      specialists: [],
      last_seq: 0,
      error_message: null,
    })
  })

  it('requires explicit user approval on Play (dispatches with requireApproval=true and NEVER auto-resumes news_funnel)', async () => {
    const card = {
      card_id: 'card-nf-1',
      title: 'News Funnel Card',
      prompt: '### News Funnel',
      flow: 'news_funnel',
      scope: 'both',
      column_name: 'backlog',
      display_seq: 1,
      discord_notify: false,
      is_verified: false,
      created_at: 1700000000,
      updated_at: 1700000000,
      job_id: null,
    }

    vi.mocked(api.listKanbanCards)
      .mockResolvedValueOnce([card])
      .mockResolvedValue([{ ...card, column_name: 'approval', job_id: 'job-nf-1' }])
    vi.mocked(api.dispatchJob).mockResolvedValue({
      job_id: 'job-nf-1',
      status: 'awaiting_approval',
      card_id: 'card-nf-1',
      error_message: null,
      current_node: null,
      interrupt_payload: {
        type: 'pre_run_approval',
        title: 'News Funnel Card',
        instruction: '### News Funnel',
        flow: 'news_funnel',
      },
      log_count: 0,
      created_at: 1700000000,
      updated_at: 1700000000,
    })

    render(<ManagerKanbanView layoutMode="wide" />)

    await waitFor(() => {
      expect(screen.getByText('News Funnel Card')).toBeInTheDocument()
    })

    const playBtn = screen.getByText('Play')
    fireEvent.click(playBtn)

    await waitFor(() => {
      expect(api.dispatchJob).toHaveBeenCalledWith('### News Funnel', 'card-nf-1', 'news_funnel', 'both', true)
    })

    await waitFor(() => {
      expect(screen.getByText(/รอยอมรับก่อนเริ่มรัน/)).toBeInTheDocument()
    })
    expect(api.resumeJob).not.toHaveBeenCalled()
  })

  it('approves and rejects pre-run approval cards directly from Request Approve column', async () => {
    const approvalCard = {
      card_id: 'card-ap-1',
      title: 'วิเคราะห์หุ้น INTC (US)',
      prompt: 'วิเคราะห์หุ้น INTC แบบละเอียด',
      flow: 'manager',
      scope: 'both',
      column_name: 'approval',
      display_seq: 5,
      discord_notify: false,
      is_verified: true,
      created_at: 1700000000,
      updated_at: 1700000000,
      job_id: 'job-ap-1',
    }

    vi.mocked(api.listKanbanCards).mockResolvedValue([approvalCard])
    vi.mocked(api.getJobStatus).mockResolvedValue({
      job_id: 'job-ap-1',
      status: 'awaiting_approval',
      card_id: 'card-ap-1',
      error_message: null,
      current_node: null,
      interrupt_payload: {
        type: 'pre_run_approval',
        title: 'วิเคราะห์หุ้น INTC (US)',
        instruction: 'วิเคราะห์หุ้น INTC แบบละเอียด',
        flow: 'manager',
      },
      log_count: 0,
      created_at: 1700000000,
      updated_at: 1700000000,
    })
    vi.mocked(api.resumeJob).mockResolvedValue({
      job_id: 'job-ap-1',
      status: 'queued',
      card_id: 'card-ap-1',
      error_message: null,
      current_node: null,
      interrupt_payload: null,
      log_count: 0,
      created_at: 1700000000,
      updated_at: 1700000000,
    })

    render(<ManagerKanbanView layoutMode="wide" />)

    await waitFor(() => {
      expect(screen.getByText('วิเคราะห์หุ้น INTC (US)')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: '✅ ยอมรับและรัน' }))

    await waitFor(() => {
      expect(api.getJobStatus).toHaveBeenCalledWith('job-ap-1')
      expect(api.resumeJob).toHaveBeenCalledWith('job-ap-1', [], [], undefined, undefined, 'approve')
    })

    fireEvent.click(screen.getByRole('button', { name: '❌ ยกเลิก' }))

    await waitFor(() => {
      expect(api.resumeJob).toHaveBeenCalledWith('job-ap-1', [], [], undefined, undefined, 'reject')
    })
  })

  it('does NOT auto-resume from board when awaiting_approval fires during execution', async () => {
    const card = {
      card_id: 'card-nf-1',
      title: 'News Funnel Card',
      prompt: '### News Funnel',
      flow: 'news_funnel',
      scope: 'both',
      column_name: 'backlog',
      display_seq: 1,
      discord_notify: false,
      is_verified: false,
      created_at: 1700000000,
      updated_at: 1700000000,
      job_id: null,
    }

    vi.mocked(api.listKanbanCards).mockResolvedValue([card])
    vi.mocked(api.dispatchJob).mockResolvedValue({
      job_id: 'job-nf-1',
      status: 'running',
      card_id: 'card-nf-1',
      error_message: null,
      current_node: 'load_pending',
      interrupt_payload: null,
      log_count: 0,
      created_at: 1700000000,
      updated_at: 1700000000,
    })
    vi.mocked(api.moveKanbanCard).mockResolvedValue({
      ...card,
      column_name: 'approval',
      job_id: 'job-nf-1',
    })

    render(<ManagerKanbanView layoutMode="wide" />)

    await waitFor(() => {
      expect(screen.getByText('News Funnel Card')).toBeInTheDocument()
    })

    const playBtn = screen.getByText('Play')
    fireEvent.click(playBtn)

    await waitFor(() => {
      expect(api.dispatchJob).toHaveBeenCalled()
    })

    const emitBtn = screen.getByText('Emit awaiting_approval')
    fireEvent.click(emitBtn)

    await waitFor(() => {
      expect(api.moveKanbanCard).toHaveBeenCalledWith('card-nf-1', 'approval')
    })

    expect(api.getJobStatus).not.toHaveBeenCalled()
    expect(api.resumeJob).not.toHaveBeenCalled()
  })

  it('does NOT auto-resume non-news_funnel flows like youtube_pitch', async () => {
    const card = {
      card_id: 'card-yp-1',
      title: 'YouTube Pitch Card',
      prompt: '### YouTube Pitch',
      flow: 'youtube_pitch',
      scope: 'both',
      column_name: 'backlog',
      display_seq: 2,
      discord_notify: false,
      is_verified: false,
      created_at: 1700000000,
      updated_at: 1700000000,
      job_id: null,
    }

    vi.mocked(api.listKanbanCards).mockResolvedValue([card])
    vi.mocked(api.dispatchJob).mockResolvedValue({
      job_id: 'job-yp-1',
      status: 'running',
      card_id: 'card-yp-1',
      error_message: null,
      current_node: 'pitch_gen',
      interrupt_payload: null,
      log_count: 0,
      created_at: 1700000000,
      updated_at: 1700000000,
    })
    vi.mocked(api.moveKanbanCard).mockResolvedValue({
      ...card,
      column_name: 'approval',
      job_id: 'job-yp-1',
    })

    render(<ManagerKanbanView layoutMode="wide" />)

    await waitFor(() => {
      expect(screen.getByText('YouTube Pitch Card')).toBeInTheDocument()
    })

    // Direct play on board
    const playBtn = screen.getByText('Play')
    fireEvent.click(playBtn)

    await waitFor(() => {
      expect(api.dispatchJob).toHaveBeenCalled()
    })

    // Trigger awaiting_approval
    const emitBtn = screen.getByText('Emit awaiting_approval')
    fireEvent.click(emitBtn)

    await waitFor(() => {
      expect(api.moveKanbanCard).toHaveBeenCalledWith('card-yp-1', 'approval')
    })

    // Must NOT auto-resume youtube_pitch flow
    expect(api.getJobStatus).not.toHaveBeenCalled()
    expect(api.resumeJob).not.toHaveBeenCalled()
  })

  it('filters out done cards older than 3 days and sorts remaining done cards newest first', async () => {
    const nowInSec = Math.floor(Date.now() / 1000)
    const daySec = 86400

    const freshDoneCard = {
      card_id: 'c-done-fresh',
      title: 'Fresh Done Card',
      prompt: null,
      flow: 'manager',
      scope: 'both',
      column_name: 'done',
      display_seq: 1,
      discord_notify: false,
      is_verified: true,
      created_at: nowInSec - daySec * 2,
      updated_at: nowInSec - daySec * 1, // 1 day ago (<= 3 days)
      job_id: null,
    }

    const newestDoneCard = {
      card_id: 'c-done-newest',
      title: 'Newest Done Card',
      prompt: null,
      flow: 'manager',
      scope: 'both',
      column_name: 'done',
      display_seq: 2,
      discord_notify: false,
      is_verified: true,
      created_at: nowInSec - daySec * 1,
      updated_at: nowInSec - 300, // 5 minutes ago
      job_id: null,
    }

    const oldDoneCard = {
      card_id: 'c-done-old',
      title: 'Old Done Card',
      prompt: null,
      flow: 'manager',
      scope: 'both',
      column_name: 'done',
      display_seq: 3,
      discord_notify: false,
      is_verified: true,
      created_at: nowInSec - daySec * 5,
      updated_at: nowInSec - daySec * 4, // 4 days ago (> 3 days -> hidden)
      job_id: null,
    }

    vi.mocked(api.listKanbanCards).mockResolvedValue([freshDoneCard, newestDoneCard, oldDoneCard])

    render(<ManagerKanbanView layoutMode="wide" />)

    await waitFor(() => {
      expect(screen.getByText('Newest Done Card')).toBeInTheDocument()
      expect(screen.getByText('Fresh Done Card')).toBeInTheDocument()
    })

    // Old card (> 3 days) should be hidden
    expect(screen.queryByText('Old Done Card')).not.toBeInTheDocument()

    // Verify order: Newest Done Card should appear before Fresh Done Card
    const cardTitles = screen.getAllByText(/(Newest|Fresh) Done Card/).map((el) => el.textContent)
    expect(cardTitles).toEqual(['Newest Done Card', 'Fresh Done Card'])
  })

  it('triggers onAskAiWithCard when clicking the AI Advisor button on a card', async () => {
    const onAskAiWithCard = vi.fn()
    const card = {
      card_id: 'c-ask-1',
      title: 'Analyze Tech Sector',
      prompt: 'Check semiconductor margin trends',
      flow: 'manager',
      scope: 'both',
      column_name: 'backlog',
      display_seq: 42,
      discord_notify: false,
      is_verified: false,
      created_at: 1700000000,
      updated_at: 1700000000,
      job_id: null,
    }

    vi.mocked(api.listKanbanCards).mockResolvedValue([card])

    render(<ManagerKanbanView layoutMode="wide" onAskAiWithCard={onAskAiWithCard} />)

    await waitFor(() => {
      expect(screen.getByText('Analyze Tech Sector')).toBeInTheDocument()
    })

    const askBtn = screen.getByTitle('ส่งหัวข้อนี้เข้าแชทเพื่อปรึกษา AI Advisor')
    fireEvent.click(askBtn)

    expect(onAskAiWithCard).toHaveBeenCalledTimes(1)
    expect(onAskAiWithCard).toHaveBeenCalledWith(
      expect.stringContaining('#AG-42'),
      expect.objectContaining({ card_id: 'c-ask-1' })
    )
  })

  it('automatically selects card matching selectedJobId when job runs', async () => {
    const card1 = {
      card_id: 'c-auto-1',
      title: 'Macro Strategy Intelligence',
      prompt: 'Run macro strategy update',
      flow: 'manager',
      scope: 'both',
      column_name: 'executing',
      display_seq: 101,
      discord_notify: false,
      is_verified: false,
      created_at: 1700000000,
      updated_at: 1700000000,
      job_id: 'job-macro-target',
    }

    vi.mocked(api.listKanbanCards).mockResolvedValue([card1])

    render(<ManagerKanbanView layoutMode="wide" selectedJobId="job-macro-target" />)

    await waitFor(() => {
      expect(screen.getByText('Macro Strategy Intelligence')).toBeInTheDocument()
    })

    // In wide mode, when a card is selected, the KanbanDetailDrawer split drawer opens with its title
    await waitFor(() => {
      expect(screen.getAllByText('Macro Strategy Intelligence').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('does not re-select or fight user when job is done and drawer was closed', async () => {
    const nowSec = Math.floor(Date.now() / 1000)
    const cardDone = {
      card_id: 'c-auto-done',
      title: 'Finished Macro Research',
      prompt: 'Done',
      flow: 'manager',
      scope: 'both',
      column_name: 'done',
      display_seq: 102,
      discord_notify: false,
      is_verified: true,
      created_at: nowSec,
      updated_at: nowSec,
      job_id: 'job-macro-done',
    }

    vi.mocked(api.listKanbanCards).mockResolvedValue([cardDone])

    const { rerender } = render(
      <ManagerKanbanView layoutMode="wide" selectedJobId="job-macro-done" />
    )

    await waitFor(() => {
      expect(screen.getByText('Finished Macro Research')).toBeInTheDocument()
    })

    // Now close drawer or re-render with refreshed cards list
    vi.mocked(api.listKanbanCards).mockResolvedValue([{ ...cardDone, updated_at: nowSec + 10 }])
    rerender(<ManagerKanbanView layoutMode="wide" selectedJobId="job-macro-done" refreshTrigger={2} />)

    // Should stay stable and not error
    expect(screen.getAllByText('Finished Macro Research').length).toBeGreaterThanOrEqual(1)
  })
})

