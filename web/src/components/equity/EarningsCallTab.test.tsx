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
    getPromiseLedger: vi.fn(),
    syncPromiseLedger: vi.fn(),
  },
}))

describe('EarningsCallTab', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getEarningsCalls).mockResolvedValue({ ticker: 'TSM', total_count: 0, items: [] })
    vi.mocked(api.getPromiseLedger).mockResolvedValue({
      ledger: {
        ticker: 'TSM',
        company: 'TSMC',
        framework_version: '0.1',
        run_date: '2026-10-20',
        calls: [],
        promises: [],
      },
      scores: {
        framework_version: '0.1',
        ticker: 'TSM',
        company: 'TSMC',
        run_date: '2026-10-20',
        window: { from: 'N/A', to: 'N/A', calls: 0, missing: [] },
        credibility: 0,
        grade: 'C',
        confidence: 'Low',
        delivery: 0,
        transparency: 100,
        n_promises: 0,
        n_resolved: 0,
        counts: {},
        bias: { label: 'Mixed', code: 'mixed', message: '', n: 0, min_n: 5 },
        milestone_slip_quarters: null,
        milestone_slip_n: 0,
        trend: { verdict: 'insufficient data', cohorts: [] },
      },
      from_cache: false,
    })
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

  it('renders Promise Check dashboard when promise ledger data is available', async () => {
    vi.mocked(api.getPromiseLedger).mockResolvedValue({
      ledger: {
        ticker: 'NVDA',
        company: 'NVIDIA Corporation',
        framework_version: '0.1',
        run_date: '2026-10-20',
        calls: [
          { quarter: 'FY2026Q2', date: '2026-08-20', status: 'ok', source: 'stockanalysis' },
        ],
        promises: [
          {
            id: 'P-NVDA-FY2026Q2-01',
            made_in: 'FY2026Q2',
            speaker: 'Colette Kress (CFO)',
            section: 'prepared',
            type: 'A_numeric',
            category: 'revenue',
            scope: 'company',
            quote: 'We expect revenue for the third quarter to be 32.5 billion dollars plus or minus 2 percent.',
            metric: 'Q3 Revenue guidance $32.5B',
            target_kind: 'point',
            target_point: 32.5,
            unit: 'USD bn',
            basis: 'GAAP',
            due: 'FY2026Q3',
            due_precision: 'explicit',
            status: 'Delivered',
            evidence_quote: 'Third-quarter revenue was a record 35.1 billion dollars.',
            delta_pct: 0.08,
            retired: false,
          },
        ],
      },
      scores: {
        framework_version: '0.1',
        ticker: 'NVDA',
        company: 'NVIDIA Corporation',
        run_date: '2026-10-20',
        window: { from: 'FY2025Q1', to: 'FY2026Q2', calls: 8, missing: [] },
        credibility: 94.5,
        grade: 'A',
        confidence: 'High',
        delivery: 96.0,
        transparency: 92.0,
        n_promises: 1,
        n_resolved: 1,
        counts: { Delivered: 1 },
        bias: {
          label: 'Sandbagger',
          code: 'sandbagger',
          message: 'Management consistently beats their guidance targets.',
          n: 1,
          min_n: 5,
        },
        milestone_slip_quarters: null,
        milestone_slip_n: 0,
        trend: {
          verdict: 'consistent',
          cohorts: [{ quarter: 'FY2026Q2', label: 'FY26 Q2', status: 'ok', made: 1, resolved: 1, delivered: 1, pending: 0, hit_rate: 1.0 }],
        },
      },
      from_cache: true,
    })

    render(
      <MemoryRouter>
        <EarningsCallTab ticker="NVDA" />
      </MemoryRouter>
    )

    expect(await screen.findByText(/Management Credibility Score/)).toBeInTheDocument()
    expect(screen.getAllByText(/94\.5/).length).toBeGreaterThan(0)
    expect(screen.getByText(/Sandbagger/)).toBeInTheDocument()
    expect(screen.getByText(/P-NVDA-FY2026Q2-01/)).toBeInTheDocument()
  })

  it('triggers 8-quarter sync when clicking sync button', async () => {
    vi.mocked(api.syncPromiseLedger).mockResolvedValue({
      ledger: {
        ticker: 'NVDA',
        company: 'NVIDIA Corporation',
        framework_version: '0.1',
        run_date: '2026-10-20',
        calls: [],
        promises: [
          {
            id: 'P-NVDA-01',
            made_in: 'FY2026Q2',
            speaker: 'Jensen Huang (CEO)',
            section: 'prepared',
            type: 'B_milestone',
            category: 'product',
            scope: 'company',
            quote: 'Blackwell production will ramp in Q4.',
            metric: 'Blackwell ramp in Q4',
            target_kind: 'milestone',
            due: 'FY2026Q4',
            due_precision: 'explicit',
            status: 'Pending',
            retired: false,
          },
        ],
      },
      scores: {
        framework_version: '0.1',
        ticker: 'NVDA',
        company: 'NVIDIA Corporation',
        run_date: '2026-10-20',
        window: { from: 'FY2025Q1', to: 'FY2026Q2', calls: 8, missing: [] },
        credibility: 85.0,
        grade: 'B',
        confidence: 'Medium',
        delivery: 85.0,
        transparency: 100.0,
        n_promises: 1,
        n_resolved: 0,
        counts: { Pending: 1 },
        bias: { label: 'Straight shooter', code: 'straight', message: '', n: 0, min_n: 5 },
        milestone_slip_quarters: null,
        milestone_slip_n: 0,
        trend: { verdict: 'insufficient data', cohorts: [] },
      },
      from_cache: false,
    })

    render(
      <MemoryRouter>
        <EarningsCallTab ticker="NVDA" />
      </MemoryRouter>
    )

    const syncButton = await screen.findByRole('button', { name: /ซิงค์ 8 ไตรมาสอัตโนมัติ/ })
    fireEvent.click(syncButton)

    await waitFor(() => {
      expect(api.syncPromiseLedger).toHaveBeenCalledWith('NVDA', true, 8)
    })
  })

  it('renders standout promises highlights card with kept and broken items', async () => {
    vi.mocked(api.getPromiseLedger).mockResolvedValue({
      ledger: {
        ticker: 'NFLX',
        company: 'Netflix, Inc.',
        framework_version: '0.1',
        run_date: '2026-10-20',
        calls: [],
        promises: [
          {
            id: 'P-NFLX-KEPT-01',
            made_in: '2025Q1',
            speaker: 'Ted Sarandos (Co-CEO)',
            section: 'prepared',
            type: 'A_numeric',
            category: 'financial',
            scope: 'company',
            quote: 'We forecast full year revenue growth of 15%.',
            metric: 'Full Year Revenue Growth',
            target_kind: 'point',
            target_point: 15.0,
            unit: '%',
            due: '2025Q4',
            due_precision: 'explicit',
            status: 'Delivered',
            evidence_quote: 'Full year revenue grew 16.2%, beating our 15% guide.',
            delta_pct: 0.08,
            retired: false,
          },
          {
            id: 'P-NFLX-BROKEN-01',
            made_in: '2024Q4',
            speaker: 'Spencer Neumann (CFO)',
            section: 'prepared',
            type: 'A_numeric',
            category: 'financial',
            scope: 'company',
            quote: 'Operating margin will reach 28% next quarter.',
            metric: 'Operating Margin Target',
            target_kind: 'point',
            target_point: 28.0,
            unit: '%',
            due: '2025Q1',
            due_precision: 'explicit',
            status: 'Missed',
            evidence_quote: 'Operating margin came in at 25.5% due to content timing.',
            delta_pct: -0.089,
            retired: false,
          },
        ],
      },
      scores: {
        framework_version: '0.1',
        ticker: 'NFLX',
        company: 'Netflix, Inc.',
        run_date: '2026-10-20',
        window: { from: '2024Q1', to: '2025Q4', calls: 8, missing: [] },
        credibility: 82.5,
        grade: 'B',
        confidence: 'High',
        delivery: 80.0,
        transparency: 90.0,
        n_promises: 2,
        n_resolved: 2,
        counts: { Delivered: 1, Missed: 1 },
        top_kept: ['P-NFLX-KEPT-01'],
        top_broken: ['P-NFLX-BROKEN-01'],
        bias: { label: 'Straight shooter', code: 'straight', message: '', n: 2, min_n: 5 },
        milestone_slip_quarters: null,
        milestone_slip_n: 0,
        trend: { verdict: 'insufficient data', cohorts: [] },
      },
      from_cache: true,
    })

    render(
      <MemoryRouter>
        <EarningsCallTab ticker="NFLX" />
      </MemoryRouter>
    )

    // Feature 1: Standout Promises Highlights
    expect(await screen.findByText(/คำสัญญาที่น่าสนใจที่สุด \(Standout Promises Highlights\)/)).toBeInTheDocument()
    expect(screen.getByText(/ทำได้ตามที่พูดไว้ \(Standout Kept Promises\)/)).toBeInTheDocument()
    expect(screen.getByText(/พลาดเป้าหรือทิ้งสัญญาเงียบๆ \(Broken or Dropped\)/)).toBeInTheDocument()
    expect(screen.getAllByText(/P-NFLX-KEPT-01/).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/P-NFLX-BROKEN-01/).length).toBeGreaterThan(0)
  })

  it('renders upcoming commitments watchlist for pending promises', async () => {
    vi.mocked(api.getPromiseLedger).mockResolvedValue({
      ledger: {
        ticker: 'MSFT',
        company: 'Microsoft Corporation',
        framework_version: '0.1',
        run_date: '2026-10-20',
        calls: [],
        promises: [
          {
            id: 'P-MSFT-PENDING-01',
            made_in: 'FY2026Q1',
            speaker: 'Satya Nadella (CEO)',
            section: 'prepared',
            type: 'B_milestone',
            category: 'cloud',
            scope: 'company',
            quote: 'Azure AI services capacity will double by Q3.',
            metric: 'Azure AI Capacity Expansion',
            target_kind: 'milestone',
            due: 'FY2026Q3',
            due_precision: 'explicit',
            status: 'Pending',
            retired: false,
          },
        ],
      },
      scores: {
        framework_version: '0.1',
        ticker: 'MSFT',
        company: 'Microsoft Corporation',
        run_date: '2026-10-20',
        window: { from: 'FY2025Q1', to: 'FY2026Q2', calls: 8, missing: [] },
        credibility: 90.0,
        grade: 'A',
        confidence: 'High',
        delivery: 90.0,
        transparency: 100.0,
        n_promises: 1,
        n_resolved: 0,
        counts: { Pending: 1 },
        bias: { label: 'Straight shooter', code: 'straight', message: '', n: 0, min_n: 5 },
        milestone_slip_quarters: null,
        milestone_slip_n: 0,
        trend: { verdict: 'insufficient data', cohorts: [] },
      },
      from_cache: true,
    })

    render(
      <MemoryRouter>
        <EarningsCallTab ticker="MSFT" />
      </MemoryRouter>
    )

    // Feature 2: Upcoming Commitments Watchlist
    expect(await screen.findByText(/สิ่งที่ต้องฟังในการประชุมครั้งถัดไป \(Upcoming Commitments Watchlist\)/)).toBeInTheDocument()
    expect(screen.getByText(/รอถึงกำหนด 1 รายการ/)).toBeInTheDocument()
    expect(screen.getAllByText(/Azure AI Capacity Expansion/).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/FY2026Q3/).length).toBeGreaterThan(0)
  })

  it('filters promise ledger table with text search and quarter selector', async () => {
    vi.mocked(api.getPromiseLedger).mockResolvedValue({
      ledger: {
        ticker: 'AAPL',
        company: 'Apple Inc.',
        framework_version: '0.1',
        run_date: '2026-10-20',
        calls: [],
        promises: [
          {
            id: 'P-AAPL-01',
            made_in: '2025Q1',
            speaker: 'Tim Cook (CEO)',
            section: 'prepared',
            type: 'A_numeric',
            category: 'financial',
            scope: 'company',
            quote: 'Gross margin between 45% and 46%.',
            metric: 'Gross Margin Range',
            target_kind: 'range',
            target_low: 45.0,
            target_high: 46.0,
            unit: '%',
            due: '2025Q2',
            due_precision: 'explicit',
            status: 'Delivered',
            retired: false,
          },
          {
            id: 'P-AAPL-02',
            made_in: '2025Q2',
            speaker: 'Luca Maestri (CFO)',
            section: 'prepared',
            type: 'A_numeric',
            category: 'financial',
            scope: 'company',
            quote: 'Services revenue double-digit growth.',
            metric: 'Services Revenue Growth',
            target_kind: 'direction',
            direction: 'up',
            due: '2025Q3',
            due_precision: 'explicit',
            status: 'Delivered',
            retired: false,
          },
        ],
      },
      scores: {
        framework_version: '0.1',
        ticker: 'AAPL',
        company: 'Apple Inc.',
        run_date: '2026-10-20',
        window: { from: '2024Q1', to: '2025Q2', calls: 8, missing: [] },
        credibility: 95.0,
        grade: 'A',
        confidence: 'High',
        delivery: 95.0,
        transparency: 100.0,
        n_promises: 2,
        n_resolved: 2,
        counts: { Delivered: 2 },
        bias: { label: 'Straight shooter', code: 'straight', message: '', n: 2, min_n: 5 },
        milestone_slip_quarters: null,
        milestone_slip_n: 0,
        trend: { verdict: 'consistent', cohorts: [] },
      },
      from_cache: true,
    })

    render(
      <MemoryRouter>
        <EarningsCallTab ticker="AAPL" />
      </MemoryRouter>
    )

    // Feature 3: Table Search and Quarter Filtering
    expect(await screen.findByPlaceholderText(/ค้นหาคำสัญญา, ชื่อผู้บริหาร, ตัวเลข หรือรหัสสัญญา.../)).toBeInTheDocument()
    expect(screen.getByText(/P-AAPL-01/)).toBeInTheDocument()
    expect(screen.getByText(/P-AAPL-02/)).toBeInTheDocument()

    // 1. Search text filter
    const searchInput = screen.getByPlaceholderText(/ค้นหาคำสัญญา, ชื่อผู้บริหาร, ตัวเลข หรือรหัสสัญญา.../)
    fireEvent.change(searchInput, { target: { value: 'Gross Margin' } })

    expect(screen.getByText(/P-AAPL-01/)).toBeInTheDocument()
    expect(screen.queryByText(/P-AAPL-02/)).not.toBeInTheDocument()

    // Clear search
    const clearBtn = screen.getByTitle(/ล้างคำค้นหา/)
    fireEvent.click(clearBtn)
    expect(screen.getByText(/P-AAPL-02/)).toBeInTheDocument()

    // 2. Quarter dropdown filter
    const quarterSelect = screen.getByLabelText(/กรองตามไตรมาส/)
    fireEvent.change(quarterSelect, { target: { value: '2025Q2' } })

    expect(screen.queryByText(/P-AAPL-01/)).not.toBeInTheDocument()
    expect(screen.getByText(/P-AAPL-02/)).toBeInTheDocument()
  })

  it('renders and toggles methodology accordion with 8-quarter transcript links', async () => {
    vi.mocked(api.getPromiseLedger).mockResolvedValue({
      ledger: {
        ticker: 'GOOGL',
        company: 'Alphabet Inc.',
        framework_version: '0.1',
        run_date: '2026-10-20',
        calls: [
          {
            quarter: 'FY2025Q4',
            date: '2026-01-30',
            status: 'ok',
            source: 'StockAnalysis',
            source_url: 'https://stockanalysis.com/stocks/googl/transcripts/2025-q4/',
            words: 12500,
          },
        ],
        promises: [
          {
            id: 'P-GOOGL-01',
            made_in: 'FY2025Q4',
            speaker: 'Sundar Pichai (CEO)',
            section: 'prepared',
            type: 'B_milestone',
            category: 'ai',
            scope: 'company',
            quote: 'Gemini 2.0 will reach enterprise general availability in Q1.',
            metric: 'Gemini 2.0 Enterprise GA',
            target_kind: 'milestone',
            due: 'FY2026Q1',
            due_precision: 'explicit',
            status: 'Pending',
            retired: false,
          },
        ],
      },
      scores: {
        framework_version: '0.1',
        ticker: 'GOOGL',
        company: 'Alphabet Inc.',
        run_date: '2026-10-20',
        window: { from: 'FY2025Q1', to: 'FY2025Q4', calls: 1, missing: [] },
        credibility: 90.0,
        grade: 'A',
        confidence: 'Low',
        delivery: 90.0,
        transparency: 100.0,
        n_promises: 1,
        n_resolved: 0,
        counts: { Pending: 1 },
        bias: { label: 'Straight shooter', code: 'straight', message: '', n: 0, min_n: 5 },
        milestone_slip_quarters: null,
        milestone_slip_n: 0,
        trend: { verdict: 'insufficient data', cohorts: [] },
      },
      from_cache: true,
    })

    render(
      <MemoryRouter>
        <EarningsCallTab ticker="GOOGL" />
      </MemoryRouter>
    )

    // Accordion toggle button should be present
    const toggleBtn = await screen.findByRole('button', { name: /วิธีการ เกณฑ์ให้คะแนน และบทถอดความที่ใช้/ })
    expect(toggleBtn).toBeInTheDocument()
    expect(toggleBtn).toHaveAttribute('aria-expanded', 'false')

    // Click to expand
    fireEvent.click(toggleBtn)
    expect(toggleBtn).toHaveAttribute('aria-expanded', 'true')

    // Check methodology pillars and transcript links
    expect(screen.getByText(/เกณฑ์คัดเลือกคำสัญญา 4 ข้อ/)).toBeInTheDocument()
    expect(screen.getByText(/เกณฑ์ส่วนต่างที่ยอมรับได้ \(Tolerance\)/)).toBeInTheDocument()
    expect(screen.getByText(/สูตรคะแนน Credibility Score \(80\/20\)/)).toBeInTheDocument()
    expect(screen.getByText(/บทถอดความที่ใช้ในการตรวจสอบ/)).toBeInTheDocument()
    expect(screen.getByText(/12,500 คำ/)).toBeInTheDocument()

    const readLink = screen.getByRole('link', { name: /เปิดอ่าน/ })
    expect(readLink).toHaveAttribute('href', 'https://stockanalysis.com/stocks/googl/transcripts/2025-q4/')
    expect(readLink).toHaveAttribute('target', '_blank')
  })
})


