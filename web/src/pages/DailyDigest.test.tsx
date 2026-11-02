import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import DailyDigest from './DailyDigest'
import { api } from '../api/client'

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>()
  return {
    ...actual,
    api: {
      getDailyDigest: vi.fn(),
      getDigestCandidates: vi.fn(),
      synthesizeDailyDigest: vi.fn(),
      getWeeklyDigest: vi.fn(),
      synthesizeWeeklyDigest: vi.fn(),
      updateDigestReadState: vi.fn(),
      appendJournal: vi.fn(),
    },
  }
})

describe('DailyDigest Page', () => {
  const mockDailyData = {
    has_digest: true,
    snapshot_age_hours: 1,
    is_snapshot_stale: false,
    fresh_pending_count: 2,
    available_dates: ['2026-10-25', '2026-10-24'],
    digest: {
      digest_id: 'daily-2026-10-25-morning',
      date: '2026-10-25',
      period: 'morning',
      generated_at: '2026-10-25T08:00:00Z',
      market_mood: 'risk_on' as const,
      market_mood_summary: 'ตลาดโลกเปิดรับความเสี่ยงต่อเนื่อง Bond yield ปรับตัวลง',
      transmission_chain: 'US Yields Fall -> THB Strong -> Fund Inflow to SET',
      top_portfolio_focus: 'DELTA และ CPALL มีโอกาสรับอานิสงส์เชิงบวก',
      selected_video_ids: ['vid1'],
      items: [
        {
          source_candidate_id: 'cand_1',
          thai_headline: 'เฟดส่งสัญญาณตรึงดอกเบี้ย ตลาดหุ้นทั่วโลกตอบรับเชิงบวก',
          what_happened: 'ธนาคารกลางสหรัฐคงดอกเบี้ยตามคาด',
          why_it_matters: 'ต้นทุนทางการเงินคงที่ ส่งผลดีต่อพอร์ตหุ้นเทค',
          thesis_signal: 'thesis_reinforcer' as const,
          impact_type: 'direct_holding' as const,
          affected_tickers: ['DELTA'],
          affected_sectors: ['Technology'],
          macro_impact_score: 8,
          asset_impact_score: 7,
          time_horizon: 'short_term' as const,
          youtube_corroboration: 'รายการ Money Coach ยืนยันกระแสเงินเข้าเทคฯ',
          source: 'Bloomberg',
          link: 'https://bloomberg.com/news/1',
          links: ['https://bloomberg.com/news/1'],
          published_at: '2026-10-25T07:00:00Z',
          age_hours: 1,
          freshness_tier: 'ultra_fresh',
          freshness_label: '1 ชม. ที่แล้ว',
          portfolio_badges: [
            {
              symbol: 'DELTA',
              company_name: 'Delta Electronics',
              impact_type: 'direct_holding' as const,
              weight_pct: 7.5,
              sector: 'Technology',
            },
          ],
          is_read: false,
          is_new: true,
        },
      ],
      youtube_highlights: [
        {
          video_id: 'vid1',
          thai_title: 'เจาะลึกทิศทางเศรษฐกิจไตรมาส 4',
          channel: 'Money Coach',
          key_takeaways: ['กระแสเงินทุนเริ่มไหลกลับตลาดเกิดใหม่', 'โฟกัสหุ้นกลุ่มอิเล็กทรอนิกส์'],
          mentioned_tickers: ['DELTA'],
          alignment_with_news: 'confirmed' as const,
          alignment_note: 'สอดคล้องกับข่าวเฟดคงดอกเบี้ย',
        },
      ],
    },
  }

  const mockCandidatesData = {
    last_fetched_at: '2026-10-25T07:30:00Z',
    stale_dropped_count: 5,
    news_candidates: [
      {
        candidate_id: 'cand_1',
        title: 'Fed signals rate pause',
        summary: 'Interest rates kept steady',
        source: 'Bloomberg',
        links: ['https://bloomberg.com/news/1'],
        sources_count: 1,
        published_at: '2026-10-25T07:00:00Z',
        age_hours: 1,
        freshness_tier: 'ultra_fresh',
        freshness_label: '1 ชม. ที่แล้ว',
        is_stale: false,
        matched_tickers: ['DELTA'],
        portfolio_badges: [
          {
            symbol: 'DELTA',
            company_name: 'Delta Electronics',
            impact_type: 'direct_holding' as const,
            weight_pct: 7.5,
            sector: 'Technology',
          },
        ],
      },
      {
        candidate_id: 'cand_2',
        title: 'General market summary with no tickers',
        summary: 'General macro trends',
        source: 'Reuters',
        links: ['https://reuters.com/news/2'],
        sources_count: 1,
        published_at: '2026-10-25T06:00:00Z',
        age_hours: 2,
        freshness_tier: 'ultra_fresh',
        freshness_label: '2 ชม. ที่แล้ว',
        is_stale: false,
        matched_tickers: [],
        portfolio_badges: [],
      },
    ],
    youtube_candidates: [
      {
        video_id: 'vid1',
        title: 'เจาะลึกทิศทางเศรษฐกิจไตรมาส 4',
        channel: 'Money Coach',
        link: 'https://youtube.com/watch?v=vid1',
        published_at: '2026-10-25T06:00:00Z',
        age_hours: 2,
        freshness_tier: 'ultra_fresh',
        freshness_label: '2 ชม. ที่แล้ว',
        thumbnail: '',
        is_fetched: true,
        cached_summary_snippet: 'กระแสเงินทุนเริ่มไหลกลับ',
        cached_tickers: ['DELTA'],
      },
    ],
  }

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getDailyDigest).mockResolvedValue(mockDailyData as any)
    vi.mocked(api.getDigestCandidates).mockResolvedValue(mockCandidatesData as any)
    vi.mocked(api.updateDigestReadState).mockResolvedValue({ ok: true, read_item_ids: ['cand_1'] })
  })

  it('renders daily digest market pulse and news cards correctly', async () => {
    render(
      <MemoryRouter>
        <DailyDigest />
      </MemoryRouter>
    )

    // Wait for content to render
    await waitFor(() => {
      expect(screen.getByText(/Daily Market Intelligence Digest/i)).toBeInTheDocument()
    })

    // Check Market Pulse
    expect(screen.getByText(/30-Sec Market Pulse/i)).toBeInTheDocument()
    expect(screen.getByText(/RISK-ON/i)).toBeInTheDocument()
    expect(screen.getByText(/DELTA และ CPALL มีโอกาสรับอานิสงส์เชิงบวก/i)).toBeInTheDocument()

    // Check News Card Headline
    expect(screen.getByText(/เฟดส่งสัญญาณตรึงดอกเบี้ย ตลาดหุ้นทั่วโลกตอบรับเชิงบวก/i)).toBeInTheDocument()
    expect(screen.getByText(/THESIS REINFORCER/i)).toBeInTheDocument()
    expect(screen.getByText(/ต้นทุนทางการเงินคงที่ ส่งผลดีต่อพอร์ตหุ้นเทค/i)).toBeInTheDocument()

    // Check YouTube Highlight
    expect(screen.getByText(/เจาะลึกทิศทางเศรษฐกิจไตรมาส 4/i)).toBeInTheDocument()
    expect(screen.getAllByText(/Money Coach/i).length).toBeGreaterThan(0)
  })

  it('allows toggling read state on news item', async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter>
        <DailyDigest />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(screen.getByText(/เฟดส่งสัญญาณตรึงดอกเบี้ย/i)).toBeInTheDocument()
    })

    // Toggle read state button
    const readBtn = screen.getByTitle(/ทำเครื่องหมายว่าอ่านแล้ว/i)
    await user.click(readBtn)

    expect(api.updateDigestReadState).toHaveBeenCalledWith({
      item_ids: ['cand_1'],
      mark_as_read: true,
    })
  })

  it('opens Pre-LLM Selection Gate and displays candidates', async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter>
        <DailyDigest />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(screen.getByText(/Daily Market Intelligence Digest/i)).toBeInTheDocument()
    })

    const toggleBtn = screen.getByRole('button', { name: /เลือกข่าว\/คลิป/i })
    await user.click(toggleBtn)

    expect(screen.getByText(/Interactive Pre-LLM Selection Gate/i)).toBeInTheDocument()
    expect(screen.getByText(/เลือกคลิป YouTube/i)).toBeInTheDocument()
    expect(screen.getAllByText(/หัวข้อข่าวสด/i).length).toBeGreaterThan(0)
  })

  it('switches to Weekly Matrix tab, fetches 7-day candidates and renders weekly developments', async () => {
    const user = userEvent.setup()
    const mockWeeklyData = {
      has_digest: true,
      snapshot_age_hours: 12,
      is_snapshot_stale: false,
      digest: {
        digest_id: 'weekly-2026-W43',
        from_date: '2026-10-18',
        to_date: '2026-10-25',
        generated_at: '2026-10-25T10:00:00Z',
        executive_themes: ['Global Liquidity Easing', 'Semiconductor Supply Chain Resilience'],
        portfolio_health_summary: 'พอร์ตภาพรวมแข็งแกร่ง หุ้นเทคได้รับแรงหนุนจาก Bond yield ที่ชะลอตัว',
        portfolio_matrix: [
          {
            ticker: 'DELTA',
            thesis_status: 'strong_hold',
            thesis_status_label: 'Strong Hold',
            weekly_summary: 'คำสั่งซื้อ AI Server เติบโตต่อเนื่อง ทิศทางกำไรมีแนวโน้มขยายตัว',
          },
        ],
        consensus_narratives: ['ตลาดคาดการณ์เฟดลดดอกเบี้ย 2 ครั้งในปีนี้'],
        contrarian_and_pitches: ['ระวังแรงขายทำกำไรกลุ่มอิเล็กทรอนิกส์ระยะสั้น'],
        week_ahead_checklist: ['ติดตามตัวเลขเงินเฟ้อสหรัฐฯ Core PCE'],
        items: [
          {
            source_candidate_id: 'cand_w1',
            thai_headline: 'รายงานสรุปภาพรวมห่วงโซ่อุปทานชิปโลกประจำสัปดาห์',
            what_happened: 'กำลังการผลิตชิป AI ตึงตัวต่อเนื่อง ส่งผลให้ราคาปรับตัวขึ้น',
            why_it_matters: 'ส่งผลบวกต่อมาร์จิ้นของ DELTA ตลอดไตรมาส',
            thesis_signal: 'thesis_reinforcer',
            impact_type: 'direct_holding',
            affected_tickers: ['DELTA'],
            affected_sectors: ['Technology'],
            macro_impact_score: 9,
            asset_impact_score: 8,
            time_horizon: 'medium_term',
            source: 'Reuters',
            link: 'https://reuters.com/news/w1',
            links: ['https://reuters.com/news/w1'],
            published_at: '2026-10-24T12:00:00Z',
            age_hours: 24,
            freshness_tier: 'fresh',
            freshness_label: '1 วันที่แล้ว',
            portfolio_badges: [
              {
                symbol: 'DELTA',
                company_name: 'Delta Electronics',
                impact_type: 'direct_holding',
                weight_pct: 7.5,
                sector: 'Technology',
              },
            ],
            is_read: false,
            is_new: false,
          },
        ],
      },
    }

    vi.mocked(api.getWeeklyDigest).mockResolvedValue(mockWeeklyData as any)

    render(
      <MemoryRouter>
        <DailyDigest />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(screen.getByText(/Daily Market Intelligence Digest/i)).toBeInTheDocument()
    })

    // Switch to Weekly Matrix tab
    const weeklyTabBtn = screen.getByRole('button', { name: /Weekly Matrix/i })
    await user.click(weeklyTabBtn)

    // Should fetch weekly digest and 7-day / 168h candidates
    expect(api.getWeeklyDigest).toHaveBeenCalled()
    expect(api.getDigestCandidates).toHaveBeenCalledWith(7, false, 168)

    // Should display weekly executive themes and portfolio matrix
    await waitFor(() => {
      expect(screen.getByText(/Global Liquidity Easing/i)).toBeInTheDocument()
    })
    expect(screen.getByText(/Semiconductor Supply Chain Resilience/i)).toBeInTheDocument()
    expect(screen.getByText(/คำสั่งซื้อ AI Server เติบโตต่อเนื่อง/i)).toBeInTheDocument()

    // Should display weekly key news development
    expect(screen.getByText(/รายงานสรุปภาพรวมห่วงโซ่อุปทานชิปโลกประจำสัปดาห์/i)).toBeInTheDocument()
  })

  it('renders historical date dropdown and allows selecting previous digest date', async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter>
        <DailyDigest />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(screen.getByText(/Daily Market Intelligence Digest/i)).toBeInTheDocument()
    })

    // Check date selector dropdown is rendered
    const dateSelect = screen.getByLabelText(/เลือกวันที่สรุปข่าว/i)
    expect(dateSelect).toBeInTheDocument()
    expect(screen.getByText(/2026-10-25 \(ล่าสุด\)/i)).toBeInTheDocument()
    expect(screen.getByText(/2026-10-24/i)).toBeInTheDocument()

    // Select previous date
    await user.selectOptions(dateSelect, '2026-10-24')
    expect(api.getDailyDigest).toHaveBeenCalledWith('2026-10-24')
  })

  it('provides quick selection buttons in Selection Gate (portfolio only, select all, deselect all)', async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter>
        <DailyDigest />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(screen.getByText(/Daily Market Intelligence Digest/i)).toBeInTheDocument()
    })

    // Open Selection Gate
    const toggleBtn = screen.getByRole('button', { name: /เลือกข่าว\/คลิป/i })
    await user.click(toggleBtn)

    // Check action buttons exist
    const portfolioOnlyBtn = screen.getByRole('button', { name: /เฉพาะกระทบพอร์ต/i })
    const selectAllNewsBtn = screen.getAllByRole('button', { name: /เลือกทั้งหมด/i })[1]! // News is second column
    const deselectAllNewsBtn = screen.getAllByRole('button', { name: /ยกเลิกทั้งหมด/i })[1]!

    expect(portfolioOnlyBtn).toBeInTheDocument()
    expect(selectAllNewsBtn).toBeInTheDocument()
    expect(deselectAllNewsBtn).toBeInTheDocument()

    // Click "🎯 เฉพาะกระทบพอร์ต"
    await user.click(portfolioOnlyBtn)

    // Ready count should reflect 1 selected news (cand_1 has DELTA badge, cand_2 does not)
    expect(screen.getByText(/1 หัวข้อข่าว/i)).toBeInTheDocument()

    // Click "เลือกทั้งหมด"
    await user.click(selectAllNewsBtn)
    expect(screen.getByText(/2 หัวข้อข่าว/i)).toBeInTheDocument()

    // Click "ยกเลิกทั้งหมด"
    await user.click(deselectAllNewsBtn)
    expect(screen.getByText(/0 หัวข้อข่าว/i)).toBeInTheDocument()
  })
})

