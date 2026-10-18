import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { EquityNotebookLMExport } from './EquityNotebookLMExport'
import { api, ApiError } from '../../api/client'
import type {
  EquityNotebookLMExportResponseDTO,
  EquityNotebookLMExportStatusDTO,
} from '../../api/types'

vi.mock('../../api/client', () => {
  class MockApiError extends Error {
    status: number
    constructor(status: number, message: string) {
      super(message)
      this.status = status
    }
  }

  return {
    api: {
      getLatestEquityNotebookLMExport: vi.fn(),
      getEquityNotebookLMExport: vi.fn(),
      exportEquityToNotebookLM: vi.fn(),
      retryEquityNotebookLMExport: vi.fn(),
    },
    ApiError: MockApiError,
  }
})

describe('EquityNotebookLMExport', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders initial export button when no previous export exists', async () => {
    vi.mocked(api.getLatestEquityNotebookLMExport).mockRejectedValueOnce(
      new ApiError(404, 'Not found')
    )

    render(<EquityNotebookLMExport ticker="AAPL" market="US" />)

    await waitFor(() => {
      expect(
        screen.getByRole('button', { name: /ส่งข้อมูล Equity ไป NotebookLM/i })
      ).toBeInTheDocument()
    })
  })

  it('triggers export API when initial button is clicked', async () => {
    vi.mocked(api.getLatestEquityNotebookLMExport).mockResolvedValueOnce(null)
    const mockExportResponse: EquityNotebookLMExportResponseDTO = {
      export_id: 'eq_export_001',
      ticker: 'AAPL',
      market: 'US',
      job_id: 'job_001',
      state: 'queued',
      stage: 'bundle_prepared',
      message: 'Export queued',
    }
    const mockStatusResponse: EquityNotebookLMExportStatusDTO = {
      export_id: 'eq_export_001',
      ticker: 'AAPL',
      market: 'US',
      job_id: 'job_001',
      mode: 'all_retained',
      state: 'queued',
      stage: 'bundle_prepared',
      snapshot_at: '2026-10-16T12:00:00Z',
      bundle_hash: 'hash123456789012345',
      notebooks: [],
      counts: {},
      source_results: [],
      coverage: {
        has_latest_analysis: true,
        historical_analyses_count: 2,
        has_financials: true,
        has_sec_filings: true,
        has_insider_trades: true,
        earnings_calls_count: 1,
        news_count: 5,
        has_options_derivatives: true,
        has_portfolio_holding: true,
        user_notes_count: 1,
      },
      warnings: [],
      can_retry: false,
    }
    vi.mocked(api.exportEquityToNotebookLM).mockResolvedValueOnce(mockExportResponse)
    vi.mocked(api.getEquityNotebookLMExport).mockResolvedValueOnce(mockStatusResponse)

    render(<EquityNotebookLMExport ticker="AAPL" market="US" />)

    const btn = await screen.findByRole('button', {
      name: /ส่งข้อมูล Equity ไป NotebookLM/i,
    })
    await userEvent.click(btn)

    expect(api.exportEquityToNotebookLM).toHaveBeenCalledWith('AAPL', {
      market: 'US',
      mode: 'all_retained',
    })
    expect(screen.getByText(/กำลังส่งออก AAPL/i)).toBeInTheDocument()
  })

  it('renders "เปิด NotebookLM เพื่อค้นคว้า" link when status is completed', async () => {
    const mockCompleted: EquityNotebookLMExportStatusDTO = {
      export_id: 'eq_export_002',
      ticker: 'NVDA',
      market: 'US',
      job_id: 'job_002',
      mode: 'all_retained',
      state: 'completed',
      stage: 'completed',
      snapshot_at: '2026-10-16T12:00:00Z',
      bundle_hash: 'hashnvda9876543210',
      notebooks: [
        {
          notebook_id: 'nb_002',
          title: 'NVDA Equity Research',
          url: 'https://notebooklm.google.com/notebook/nb_002',
          status: 'ready',
          source_count: 10,
        },
      ],
      counts: { total_sources: 10 },
      source_results: [
        { file_name: '00-research-guide.md', title: 'Guide', status: 'ready' },
        { file_name: '01-equity-thesis.md', title: 'Thesis', status: 'ready' },
      ],
      coverage: {
        has_latest_analysis: true,
        historical_analyses_count: 1,
        has_financials: true,
        has_sec_filings: true,
        has_insider_trades: true,
        earnings_calls_count: 1,
        news_count: 3,
        has_options_derivatives: true,
        has_portfolio_holding: false,
        user_notes_count: 0,
      },
      warnings: [],
      can_retry: false,
    }

    vi.mocked(api.getLatestEquityNotebookLMExport).mockResolvedValueOnce(mockCompleted)

    render(<EquityNotebookLMExport ticker="NVDA" market="US" />)

    const link = await screen.findByRole('link', {
      name: /เปิด NotebookLM เพื่อค้นคว้า/i,
    })
    expect(link).toBeInTheDocument()
    expect(link).toHaveAttribute('href', 'https://notebooklm.google.com/notebook/nb_002')
    expect(link).toHaveAttribute('target', '_blank')
  })

  it('renders error badge and retry button when status is failed and can_retry is true', async () => {
    const mockFailed: EquityNotebookLMExportStatusDTO = {
      export_id: 'eq_export_003',
      ticker: 'TSLA',
      market: 'US',
      job_id: 'job_003',
      mode: 'all_retained',
      state: 'failed',
      stage: 'upload_sources',
      snapshot_at: '2026-10-16T12:00:00Z',
      bundle_hash: 'hashtsla111222333',
      notebooks: [],
      counts: {},
      source_results: [],
      coverage: {
        has_latest_analysis: true,
        historical_analyses_count: 0,
        has_financials: false,
        has_sec_filings: false,
        has_insider_trades: false,
        earnings_calls_count: 0,
        news_count: 0,
        has_options_derivatives: false,
        has_portfolio_holding: false,
        user_notes_count: 0,
      },
      warnings: ['No SEC filings cached'],
      error: 'Upload failed due to network timeout',
      can_retry: true,
    }

    vi.mocked(api.getLatestEquityNotebookLMExport).mockResolvedValueOnce(mockFailed)
    vi.mocked(api.retryEquityNotebookLMExport).mockResolvedValueOnce({
      export_id: 'eq_export_003',
      ticker: 'TSLA',
      market: 'US',
      job_id: 'job_retry_003',
      state: 'queued',
      stage: 'bundle_prepared',
      message: 'Retry queued',
    })
    vi.mocked(api.getEquityNotebookLMExport).mockResolvedValueOnce({
      ...mockFailed,
      state: 'queued',
      job_id: 'job_retry_003',
    })

    render(<EquityNotebookLMExport ticker="TSLA" market="US" />)

    const retryBtn = await screen.findByRole('button', { name: /ลองใหม่/i })
    expect(retryBtn).toBeInTheDocument()

    await userEvent.click(retryBtn)
    expect(api.retryEquityNotebookLMExport).toHaveBeenCalledWith('eq_export_003')
  })

  it('notifies user when export is blocked due to 409 conflict', async () => {
    vi.mocked(api.getLatestEquityNotebookLMExport).mockResolvedValueOnce(null)
    const onNotify = vi.fn()
    vi.mocked(api.exportEquityToNotebookLM).mockRejectedValueOnce(
      new ApiError(409, 'Baseline analysis required before export')
    )

    render(<EquityNotebookLMExport ticker="UNKNOWN" market="US" onNotify={onNotify} />)

    const btn = await screen.findByRole('button', {
      name: /ส่งข้อมูล Equity ไป NotebookLM/i,
    })
    await userEvent.click(btn)

    expect(onNotify).toHaveBeenCalledWith(
      expect.stringContaining('จำเป็นต้องมีบทวิเคราะห์พื้นฐานของ UNKNOWN ก่อน'),
      'error'
    )
  })

  it('opens and closes popover with document items and warnings', async () => {
    const mockWithWarnings: EquityNotebookLMExportStatusDTO = {
      export_id: 'eq_export_004',
      ticker: 'DELTA',
      market: 'TH',
      job_id: 'job_004',
      mode: 'all_retained',
      state: 'ready_with_warnings',
      stage: 'completed',
      snapshot_at: '2026-10-16T12:00:00Z',
      bundle_hash: 'hashdelta9999999999',
      notebooks: [
        {
          notebook_id: 'nb_004',
          title: 'DELTA Equity Research',
          url: 'https://notebooklm.google.com/notebook/nb_004',
          status: 'ready',
          source_count: 8,
        },
      ],
      counts: { total_sources: 10 },
      source_results: [
        { file_name: '00-research-guide.md', title: 'Guide', status: 'ready' },
        { file_name: '01-equity-thesis.md', title: 'Thesis', status: 'ready' },
      ],
      coverage: {
        has_latest_analysis: true,
        historical_analyses_count: 0,
        has_financials: true,
        has_sec_filings: false,
        has_insider_trades: false,
        earnings_calls_count: 0,
        news_count: 1,
        has_options_derivatives: false,
        has_portfolio_holding: false,
        user_notes_count: 0,
      },
      warnings: ['ตลาด TH: ไม่มีข้อมูล SEC EDGAR filings / Form 4'],
      can_retry: false,
    }

    vi.mocked(api.getLatestEquityNotebookLMExport).mockResolvedValueOnce(mockWithWarnings)

    render(<EquityNotebookLMExport ticker="DELTA" market="TH" />)

    const dropdownTrigger = await screen.findByRole('button', {
      name: /ดูรายละเอียดชุดข้อมูลที่ส่งออกและตัวเลือกอัปเดต/i,
    })
    expect(dropdownTrigger).toBeInTheDocument()

    await userEvent.click(dropdownTrigger)

    expect(
      screen.getByText(/NotebookLM Research Bundle: DELTA \(TH\)/i)
    ).toBeInTheDocument()
    expect(
      screen.getByText(/ข้อสังเกตความสมบูรณ์ของข้อมูล:/i)
    ).toBeInTheDocument()
    expect(
      screen.getByText(/ตลาด TH: ไม่มีข้อมูล SEC EDGAR filings \/ Form 4/i)
    ).toBeInTheDocument()
    expect(
      screen.getByText(/00: คู่มือแนวทางการค้นคว้า & Prompt ตัวอย่าง/i)
    ).toBeInTheDocument()
  })
})
