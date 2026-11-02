import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import ManagerRightPanel from './ManagerRightPanel'

vi.mock('../../api/client', () => ({
  ApiError: class ApiError extends Error {},
  api: {
    listKanbanCards: vi.fn().mockResolvedValue([]),
    getJobStatus: vi.fn().mockResolvedValue({
      job_id: 'test_job_1',
      status: 'running',
      card_id: null,
      error_message: null,
      interrupt_payload: null,
      log_count: 5,
      current_node: 'macro_quant',
      created_at: Date.now(),
      updated_at: Date.now(),
    }),
    getJobOutputs: vi.fn().mockResolvedValue({
      job_id: 'test_job_1',
      status: 'done',
      executive_summary: 'ภาพรวมเศรษฐกิจแข็งแกร่ง',
      executive_summary_created_at: Date.now(),
      specialists: [],
      last_seq: 1,
      error_message: null,
    }),
    getActiveAgentStatus: vi.fn().mockResolvedValue({
      running: true,
      flow: 'manager',
      node: 'macro_quant',
      job_id: 'test_job_1',
    }),
  },
}))

describe('ManagerRightPanel', () => {
  it('renders kanban and swarm tabs when open', () => {
    const onTabChange = vi.fn()
    const onClose = vi.fn()

    render(
      <ManagerRightPanel
        isOpen={true}
        onClose={onClose}
        activeTab="kanban"
        onTabChange={onTabChange}
        selectedJobId="job_123"
        onSelectJob={vi.fn()}
        detectedJobIds={['job_123']}
        activeAgentStatus={{ running: true, flow: 'manager', node: 'macro_quant', job_id: 'job_123' }}
      />
    )

    expect(screen.getByText('บอร์ดงาน')).toBeInTheDocument()
    expect(screen.getByText('Trinity Swarm')).toBeInTheDocument()

    // Clicking Swarm tab calls onTabChange with 'swarm'
    fireEvent.click(screen.getByText('Trinity Swarm'))
    expect(onTabChange).toHaveBeenCalledWith('swarm')

    // Clicking close button calls onClose
    fireEvent.click(screen.getByTitle('ย่อแถบด้านข้าง'))
    expect(onClose).toHaveBeenCalled()
  })

  it('renders nothing when isOpen is false', () => {
    const { container } = render(
      <ManagerRightPanel
        isOpen={false}
        onClose={vi.fn()}
        activeTab="execution"
        onTabChange={vi.fn()}
        selectedJobId={null}
        onSelectJob={vi.fn()}
        detectedJobIds={[]}
      />
    )

    expect(container).toBeEmptyDOMElement()
  })

  it('renders TrinitySwarmView when activeTab is swarm', () => {
    render(
      <ManagerRightPanel
        isOpen={true}
        onClose={vi.fn()}
        activeTab="swarm"
        onTabChange={vi.fn()}
        selectedJobId={null}
        onSelectJob={vi.fn()}
        detectedJobIds={[]}
        activeAgentStatus={{ running: true, flow: 'manager', node: 'macro_quant', job_id: 'job_123' }}
      />
    )

    expect(screen.getByText('Macro Quant')).toBeInTheDocument()
    expect(screen.getByText('Macro Economist')).toBeInTheDocument()
    expect(screen.getByText('Strategic Allocator')).toBeInTheDocument()
    expect(screen.getByText('Equity Intel')).toBeInTheDocument()
    expect(screen.getByText('Archivist')).toBeInTheDocument()
    expect(screen.getByText('กำลังทำงาน')).toBeInTheDocument()
  })

  it('renders only compact (default) and full width mode buttons for kanban', () => {
    render(
      <ManagerRightPanel
        isOpen={true}
        onClose={vi.fn()}
        activeTab="kanban"
        onTabChange={vi.fn()}
        selectedJobId={null}
        onSelectJob={vi.fn()}
        detectedJobIds={[]}
      />
    )

    // Only compact and full buttons exist
    const compactBtn = screen.getByTitle('มุมมองกะทัดรัด (คอลัมน์เดียว)')
    const fullBtn = screen.getByTitle('มุมมองขยายเต็มจอ')
    expect(compactBtn).toBeInTheDocument()
    expect(fullBtn).toBeInTheDocument()
    expect(screen.queryByTitle('มุมมองกว้าง (4 คอลัมน์)')).not.toBeInTheDocument()

    // Default is compact (has bold font or active styling)
    expect(compactBtn).toHaveClass('font-bold')

    // Click full button
    fireEvent.click(fullBtn)
    expect(fullBtn).toHaveClass('font-bold')
  })
})
