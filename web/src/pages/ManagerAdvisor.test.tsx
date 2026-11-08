import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import ManagerAdvisor from './ManagerAdvisor'
import { api } from '../api/client'

vi.mock('../api/client', () => ({
  ApiError: class ApiError extends Error {},
  api: {
    listAdvisorSessions: vi.fn(),
    createAdvisorSession: vi.fn(),
    getAdvisorMessages: vi.fn(),
    getSandboxState: vi.fn(),
    getHermesSyncStatus: vi.fn(),
    getActiveAgentStatus: vi.fn(),
    getJobStatus: vi.fn(),
    getJobOutputs: vi.fn(),
    listKanbanCards: vi.fn().mockResolvedValue([]),
  },
  streamAdvisorChat: vi.fn(),
}))

describe('ManagerAdvisor Page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getActiveAgentStatus).mockResolvedValue({
      running: false,
      flow: null,
      node: null,
      job_id: null,
    })
    vi.mocked(api.listAdvisorSessions).mockResolvedValue([
      {
        id: 'session_1',
        title: 'พอร์ตหลัก 2026',
        created_at: '2026-10-31T00:00:00Z',
        updated_at: '2026-10-31T00:00:00Z',
        message_count: 2,
      },
    ])
    vi.mocked(api.getAdvisorMessages).mockResolvedValue([
      {
        id: 'msg_1',
        session_id: 'session_1',
        role: 'user',
        content: 'ภาพรวมพอร์ตเป็นอย่างไรบ้าง?',
        citations: ['@Portfolio'],
        created_at: '2026-10-31T00:00:00Z',
      },
      {
        id: 'msg_2',
        session_id: 'session_1',
        role: 'assistant',
        content: 'พอร์ตการลงทุนของคุณมีสัดส่วนสมดุลดีครับ',
        citations: ['@Portfolio'],
        created_at: '2026-10-31T00:00:01Z',
      },
    ])
  })

  it('renders top navigation and chat messages', async () => {
    render(
      <MemoryRouter>
        <ManagerAdvisor />
      </MemoryRouter>
    )

    expect(screen.getByText('Trinity AI Advisor')).toBeInTheDocument()
    expect(screen.getByText('พร้อมให้คำปรึกษา')).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByText('ภาพรวมพอร์ตเป็นอย่างไรบ้าง?')).toBeInTheDocument()
      expect(screen.getByText('พอร์ตการลงทุนของคุณมีสัดส่วนสมดุลดีครับ')).toBeInTheDocument()
    })
  })

  it('renders prompt composer with quick context pills', () => {
    render(
      <MemoryRouter>
        <ManagerAdvisor />
      </MemoryRouter>
    )

    expect(screen.getByPlaceholderText(/ปรึกษาการลงทุน/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '@Portfolio' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '@Macro' })).toBeInTheDocument()
    expect(screen.getByText('ส่งคำถาม')).toBeInTheDocument()
  })

  it('renders right-hand panel with tabs for Kanban Board and Trinity Swarm', async () => {
    render(
      <MemoryRouter>
        <ManagerAdvisor />
      </MemoryRouter>
    )

    expect(screen.getByText('บอร์ดงาน')).toBeInTheDocument()
    expect(screen.getByText('Trinity Swarm')).toBeInTheDocument()
    expect(screen.getByText('Morning Delta')).toBeInTheDocument()
    expect(screen.getByText('Morning Review')).toBeInTheDocument()
    expect(screen.getByText('Swarm & Jobs')).toBeInTheDocument()
  })
})

