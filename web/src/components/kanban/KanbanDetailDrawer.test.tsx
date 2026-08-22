import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import KanbanDetailDrawer from './KanbanDetailDrawer'
import { api } from '../../api/client'

vi.mock('../../api/client', () => ({
  api: {
    getNewsFunnelPending: vi.fn(),
    getNewsFunnelFiltered: vi.fn(),
    resumeJob: vi.fn(),
  },
  ApiError: class ApiError extends Error {},
}))

vi.mock('../LiveTerminal', () => ({
  default: ({ onAwaitingApproval }: { onAwaitingApproval?: (payload: any) => void }) => {
    return (
      <div data-testid="live-terminal">
        LiveTerminal
        <button
          onClick={() =>
            onAwaitingApproval?.({
              type: 'news_funnel_approval',
              candidates: [{ event_id: 'ev-1' }, { event_id: 'ev-2' }],
            })
          }
        >
          Trigger Approval
        </button>
      </div>
    )
  },
}))

describe('KanbanDetailDrawer', () => {
  beforeEach(() => {
    vi.resetAllMocks()
  })

  it('triggers onDispatchCard and auto-resumes with pre-selected IDs on awaiting_approval without showing ApprovalPanel', async () => {
    vi.mocked(api.getNewsFunnelPending).mockResolvedValue([
      {
        event_id: 'ev-1',
        canonical_title: 'News 1',
        comprehensive_summary: 'Summary 1',
        macro_impact_score: 8,
        asset_impact_score: 7,
        extracted_tickers: [],
        extracted_themes: [],
        primary_tags: [],
        sources: [],
        links: [],
      },
      {
        event_id: 'ev-2',
        canonical_title: 'News 2',
        comprehensive_summary: 'Summary 2',
        macro_impact_score: 9,
        asset_impact_score: 8,
        extracted_tickers: [],
        extracted_themes: [],
        primary_tags: [],
        sources: [],
        links: [],
      },
    ])
    vi.mocked(api.getNewsFunnelFiltered).mockResolvedValue([])
    vi.mocked(api.resumeJob).mockResolvedValue({
      job_id: 'job-1',
      status: 'running',
      card_id: 'card-1',
      error_message: null,
      current_node: 'synthesize',
      interrupt_payload: null,
      log_count: 0,
      created_at: 1700000000,
      updated_at: 1700000000,
    })

    const onDispatchCard = vi.fn()
    const onCardTransition = vi.fn()
    const card = {
      card_id: 'card-1',
      title: 'News Funnel High-Impact',
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

    const { rerender } = render(
      <KanbanDetailDrawer
        card={card}
        onClose={vi.fn()}
        onCardTransition={onCardTransition}
        onDispatchCard={onDispatchCard}
      />
    )

    await waitFor(() => {
      expect(screen.getByText('News 1')).toBeInTheDocument()
    })

    // Click "🚀 เริ่มสังเคราะห์" button in drawer
    const dispatchBtn = screen.getByText(/🚀 เริ่มสังเคราะห์ \(2 รายการ\)/)
    fireEvent.click(dispatchBtn)

    expect(onDispatchCard).toHaveBeenCalledWith(card, 'news_funnel')

    // Simulate card being dispatched and getting a job_id
    const runningCard = { ...card, job_id: 'job-1', column_name: 'executing' }
    rerender(
      <KanbanDetailDrawer
        card={runningCard}
        onClose={vi.fn()}
        onCardTransition={onCardTransition}
        onDispatchCard={onDispatchCard}
      />
    )

    // Trigger awaiting_approval from LiveTerminal
    const triggerApprovalBtn = screen.getByText('Trigger Approval')
    fireEvent.click(triggerApprovalBtn)

    // Should call resumeJob automatically with pre-selected event_ids ['ev-1', 'ev-2']
    await waitFor(() => {
      expect(api.resumeJob).toHaveBeenCalledWith('job-1', [], [], ['ev-1', 'ev-2'], undefined, 'approve', undefined, undefined)
    })

    // ApprovalPanel header should NOT be rendered
    expect(screen.queryByText(/รอการอนุมัติ — เลือกรายการข่าว/)).not.toBeInTheDocument()
  })

  it('does NOT render EquityStockControls on news_funnel card even if prompt contains "วิเคราะห์หุ้น"', async () => {
    vi.mocked(api.getNewsFunnelPending).mockResolvedValue([])
    vi.mocked(api.getNewsFunnelFiltered).mockResolvedValue([])

    const card = {
      card_id: 'card-nf-text',
      title: '[EVENING] News Funnel High-Impact (41 items)',
      prompt: '### 📰 รายการข่าว High-Impact\n- **สรุป:** นักวิเคราะห์หุ้นคาดการณ์ผลตอบแทน...',
      flow: 'news_funnel',
      scope: 'both',
      column_name: 'backlog',
      display_seq: 92,
      discord_notify: false,
      is_verified: false,
      created_at: 1700000000,
      updated_at: 1700000000,
      job_id: null,
    }

    render(
      <KanbanDetailDrawer
        card={card}
        onClose={vi.fn()}
        onCardTransition={vi.fn()}
        onDispatchCard={vi.fn()}
      />
    )

    // Should NOT show EquityStockControls label
    expect(screen.queryByText('เลือกรุ่นหุ้นสำหรับวิเคราะห์และดึงข่าว')).not.toBeInTheDocument()
  })

  it('renders Discord toggle button and calls toggleCardDiscord on click', async () => {
    vi.mocked(api.getNewsFunnelPending).mockResolvedValue([])
    vi.mocked(api.getNewsFunnelFiltered).mockResolvedValue([])
    const toggleMock = vi.fn().mockResolvedValue({ ok: true })
    vi.mocked(api).toggleCardDiscord = toggleMock

    const onCardTransition = vi.fn()
    const card = {
      card_id: 'card-nb-1',
      title: 'NotebookLM Podcast Deep Dive',
      prompt: 'memories/30_Knowledge_Base/NotebookLM_Sources/test.md',
      flow: 'notebooklm',
      scope: 'both',
      column_name: 'backlog',
      display_seq: 10,
      discord_notify: true,
      is_verified: true,
      created_at: 1700000000,
      updated_at: 1700000000,
      job_id: null,
    }

    render(
      <KanbanDetailDrawer
        card={card}
        onClose={vi.fn()}
        onCardTransition={onCardTransition}
        onDispatchCard={vi.fn()}
      />
    )

    expect(screen.getByText('โพสต์ลง Discord (นักข่าวส่วนตัว)')).toBeInTheDocument()
    expect(screen.getByText('เปิด')).toBeInTheDocument()

    const toggleBtn = screen.getByRole('switch', { name: /สลับการแจ้งเตือน Discord/i })
    fireEvent.click(toggleBtn)

    await waitFor(() => {
      expect(toggleMock).toHaveBeenCalledWith('card-nb-1', false)
      expect(onCardTransition).toHaveBeenCalled()
    })
  })
})

