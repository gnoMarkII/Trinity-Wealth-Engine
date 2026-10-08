import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MacroNotebookLMExport } from './MacroNotebookLMExport'
import { api, ApiError } from '../../../api/client'
import type { MacroNotebookLMExportResponseDTO, MacroNotebookLMExportStatusDTO } from '../../../api/types'

vi.mock('../../../api/client', () => {
  class MockApiError extends Error {
    status: number
    constructor(status: number, message: string) {
      super(message)
      this.status = status
    }
  }

  return {
    api: {
      getLatestMacroNotebookLMExport: vi.fn(),
      getMacroNotebookLMExport: vi.fn(),
      exportMacroToNotebookLM: vi.fn(),
      retryMacroNotebookLMExport: vi.fn(),
    },
    ApiError: MockApiError,
  }
})

describe('MacroNotebookLMExport', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders initial export button when no previous export exists', async () => {
    vi.mocked(api.getLatestMacroNotebookLMExport).mockRejectedValueOnce(
      new ApiError(404, 'Not found')
    )

    render(<MacroNotebookLMExport />)

    await waitFor(() => {
      expect(
        screen.getByRole('button', { name: /ส่งข้อมูล Macro ไป NotebookLM/i })
      ).toBeInTheDocument()
    })
  })

  it('triggers export API when initial button is clicked', async () => {
    vi.mocked(api.getLatestMacroNotebookLMExport).mockResolvedValueOnce(null)
    const mockExportResponse: MacroNotebookLMExportResponseDTO = {
      export_id: 'export_001',
      job_id: 'job_001',
      state: 'queued',
      stage: 'initialized',
      message: 'Export queued',
    }
    const mockStatusResponse: MacroNotebookLMExportStatusDTO = {
      export_id: 'export_001',
      job_id: 'job_001',
      mode: 'all_retained',
      state: 'queued',
      stage: 'initialized',
      snapshot_at: '2026-10-05T09:30:00Z',
      bundle_hash: 'hash123',
      notebooks: [],
      counts: {},
      source_results: [],
      coverage: {
        strategy_report_present: true,
        historical_reports_count: 1,
        catalog_notes_count: 5,
        indicator_series_count: 2,
        market_observables_cached: 13,
        market_observables_total: 13,
        thailand_hard_data_present: true,
        sector_rotation_present: true,
        news_events_count: 1,
      },
      warnings: [],
      can_retry: false,
    }
    vi.mocked(api.exportMacroToNotebookLM).mockResolvedValueOnce(mockExportResponse)
    vi.mocked(api.getMacroNotebookLMExport).mockResolvedValueOnce(mockStatusResponse)

    render(<MacroNotebookLMExport />)

    const btn = await screen.findByRole('button', { name: /ส่งข้อมูล Macro ไป NotebookLM/i })
    await userEvent.click(btn)

    expect(api.exportMacroToNotebookLM).toHaveBeenCalledWith({ mode: 'all_retained' })
    expect(screen.getByText(/กำลังส่งออก Macro/i)).toBeInTheDocument()
  })

  it('renders "เปิด NotebookLM เพื่อค้นคว้า" link when status is completed', async () => {
    const mockCompleted: MacroNotebookLMExportStatusDTO = {
      export_id: 'export_002',
      job_id: 'job_002',
      mode: 'all_retained',
      state: 'completed',
      stage: 'completed',
      snapshot_at: '2026-10-05T09:30:00Z',
      bundle_hash: 'hash1234567890',
      notebooks: [
        {
          notebook_id: 'nb_001',
          title: 'Macro Strategy Intelligence',
          url: 'https://notebooklm.google.com/notebook/nb_001',
          status: 'ready',
          source_count: 9,
        },
      ],
      counts: {},
      source_results: [
        { file_name: '00-research-guide.md', title: 'Guide', status: 'ok' },
        { file_name: '01-current-macro-report.md', title: 'Report', status: 'ok' },
      ],
      coverage: {
        strategy_report_present: true,
        historical_reports_count: 1,
        catalog_notes_count: 5,
        indicator_series_count: 2,
        market_observables_cached: 13,
        market_observables_total: 13,
        thailand_hard_data_present: true,
        sector_rotation_present: true,
        news_events_count: 1,
      },
      warnings: [],
      can_retry: false,
    }
    vi.mocked(api.getLatestMacroNotebookLMExport).mockResolvedValueOnce(mockCompleted)

    render(<MacroNotebookLMExport />)

    const link = await screen.findByRole('link', { name: /เปิด NotebookLM เพื่อค้นคว้า/i })
    expect(link).toBeInTheDocument()
    expect(link).toHaveAttribute('href', 'https://notebooklm.google.com/notebook/nb_001')
    expect(link).toHaveAttribute('target', '_blank')
    expect(link).toHaveAttribute('rel', 'noopener noreferrer')

    // Click details trigger button to view coverage checklist
    const detailsBtn = screen.getByRole('button', { name: /2\/2 แหล่ง/i })
    await userEvent.click(detailsBtn)

    expect(screen.getByText(/NotebookLM Research Bundle/i)).toBeInTheDocument()
    expect(screen.getByText(/00: คู่มือแนวทางการค้นคว้า/i)).toBeInTheDocument()
    expect(screen.getByText(/01: รายงานภาพรวมเศรษฐกิจ Macro ปัจจุบัน/i)).toBeInTheDocument()
  })

  it('renders retry button when export status is failed', async () => {
    const mockFailed: MacroNotebookLMExportStatusDTO = {
      export_id: 'export_003',
      job_id: 'job_003',
      mode: 'all_retained',
      state: 'failed',
      stage: 'failed',
      snapshot_at: '2026-10-05T09:30:00Z',
      bundle_hash: 'hash_err',
      notebooks: [],
      counts: {},
      source_results: [],
      coverage: {
        strategy_report_present: false,
        historical_reports_count: 0,
        catalog_notes_count: 0,
        indicator_series_count: 0,
        market_observables_cached: 0,
        market_observables_total: 13,
        thailand_hard_data_present: false,
        sector_rotation_present: false,
        news_events_count: 0,
      },
      warnings: [],
      error: 'MCP timeout on source upload',
      can_retry: true,
    }
    vi.mocked(api.getLatestMacroNotebookLMExport).mockResolvedValueOnce(mockFailed)
    vi.mocked(api.retryMacroNotebookLMExport).mockResolvedValueOnce({
      export_id: 'export_003',
      state: 'queued',
      stage: 'initialized',
      message: 'Retry queued',
    })
    vi.mocked(api.getMacroNotebookLMExport).mockResolvedValueOnce({
      ...mockFailed,
      state: 'queued',
    })

    render(<MacroNotebookLMExport />)

    expect(await screen.findByText(/ส่งออกไม่สำเร็จ/i)).toBeInTheDocument()
    const retryBtn = screen.getByRole('button', { name: /ลองใหม่/i })
    await userEvent.click(retryBtn)

    expect(api.retryMacroNotebookLMExport).toHaveBeenCalledWith('export_003')
  })
})
