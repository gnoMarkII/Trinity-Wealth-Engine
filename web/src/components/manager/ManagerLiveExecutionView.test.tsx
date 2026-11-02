import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import ManagerLiveExecutionView from './ManagerLiveExecutionView'
import { api } from '../../api/client'

vi.mock('../../api/client', () => ({
  api: {
    getJobStatus: vi.fn(),
    getJobOutputs: vi.fn(),
  },
}))

describe('ManagerLiveExecutionView', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getJobStatus).mockResolvedValue({
      job_id: 'job_macro_999',
      status: 'running',
      card_id: null,
      error_message: null,
      interrupt_payload: null,
      log_count: 5,
      current_node: 'macro_quant',
      created_at: Date.now(),
      updated_at: Date.now(),
    })
    vi.mocked(api.getJobOutputs).mockResolvedValue({
      job_id: 'job_macro_999',
      status: 'done',
      executive_summary: 'เศรษฐกิจมหภาคมีแนวโน้มขยายตัวดี',
      executive_summary_created_at: Date.now(),
      specialists: [],
      last_seq: 1,
      error_message: null,
    })
  })

  it('renders standby state when jobId is null', () => {
    render(
      <ManagerLiveExecutionView
        jobId={null}
        onTriggerQuickJob={vi.fn()}
        onSwitchToSwarm={vi.fn()}
      />
    )

    expect(screen.getByText('พร้อมติดตามความคืบหน้างานวิจัย')).toBeInTheDocument()
    expect(screen.getByText('สั่งอัปเดต Macro Report')).toBeInTheDocument()
    expect(screen.getByText('สั่งวิเคราะห์หุ้น NVDA')).toBeInTheDocument()
  })

  it('renders pipeline stages and live activity when jobId is provided', async () => {
    render(
      <ManagerLiveExecutionView
        jobId="job_macro_999"
        allJobIds={['job_macro_999']}
      />
    )

    expect(screen.getByText(/Macro Strategy Update/i)).toBeInTheDocument()
    expect(screen.getByText('ID: job_macro_999...')).toBeInTheDocument()
    expect(screen.getByText('ความคืบหน้ารวม')).toBeInTheDocument()

    // Pipeline stages (4 specialist stages)
    expect(screen.queryByText('Supervisor')).not.toBeInTheDocument()
    expect(screen.getByText('Macro Quant')).toBeInTheDocument()
    expect(screen.getByText('Macro Economist')).toBeInTheDocument()
    expect(screen.getByText('Strategic Allocator')).toBeInTheDocument()
    expect(screen.getByText('Archivist')).toBeInTheDocument()

    await waitFor(() => {
      expect(api.getJobStatus).toHaveBeenCalledWith('job_macro_999')
    })
  })
})
