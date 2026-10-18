import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { EquityNews } from './EquityNews'
import { api } from '../../api/client'

vi.mock('../../api/client', () => ({
  api: {
    getEquityNews: vi.fn(),
    getEquityNewsDiscovery: vi.fn(),
  },
}))

describe('EquityNews', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getEquityNewsDiscovery).mockResolvedValue(null as any)
  })

  it('renders loading state initially', () => {
    vi.mocked(api.getEquityNews).mockImplementation(() => new Promise(() => {}))
    render(<EquityNews ticker="AAPL" />)
    expect(screen.getByText('กำลังโหลดข่าวสาร...')).toBeInTheDocument()
  })

  it('renders empty 404 state when no news found and discovery is empty', async () => {
    vi.mocked(api.getEquityNews).mockRejectedValue({ status: 404, message: '404 Not Found' })
    vi.mocked(api.getEquityNewsDiscovery).mockResolvedValue({
      query_symbol: 'AAPL',
      items: [],
      status: 'ok',
      source: 'rss',
      as_of_date: '2026-10-17',
      fetched_at: Date.now(),
      limitations: [],
    })
    render(<EquityNews ticker="AAPL" />)

    await waitFor(() => {
      expect(screen.getByText('ยังไม่มีข้อมูลข่าวสำหรับหุ้นตัวนี้ในระบบ')).toBeInTheDocument()
      expect(screen.getByText(/สั่งงานผ่านผู้จัดการ \(Manager Agent\)/)).toBeInTheDocument()
    })
  })

  it('renders error state on API failure when no discovery news exists', async () => {
    vi.mocked(api.getEquityNews).mockRejectedValue(new Error('Network error'))
    vi.mocked(api.getEquityNewsDiscovery).mockResolvedValue(null as any)
    render(<EquityNews ticker="AAPL" />)

    await waitFor(() => {
      expect(screen.getByText('เกิดข้อผิดพลาด')).toBeInTheDocument()
      expect(screen.getByText('Network error')).toBeInTheDocument()
    })
  })

  it('renders news items and summary successfully', async () => {
    vi.mocked(api.getEquityNews).mockResolvedValue({
      ticker: 'AAPL',
      market: 'US',
      last_updated: '2026-08-05 22:10:59',
      news_date: '2026-08-05',
      items: [
        {
          title: 'Apple Intelligence News',
          source: 'Reuters',
          link: 'https://example.com/news',
          published_at: '2026-08-05T20:00:00Z',
          age_hours: 2,
          freshness_reason: 'Fresh',
          is_stale: false,
          sources_count: 2,
          summary: 'Apple is rolling out new AI features across iOS devices.',
        },
      ],
    })

    render(<EquityNews ticker="AAPL" />)

    await waitFor(() => {
      expect(screen.getByText('Apple Intelligence News')).toBeInTheDocument()
      expect(screen.getByText('Apple is rolling out new AI features across iOS devices.')).toBeInTheDocument()
      expect(screen.getByText('ที่มา: Reuters')).toBeInTheDocument()
      expect(screen.getByText('2 sources')).toBeInTheDocument()
      expect(screen.getByText('อ่านต่อบน Reuters')).toBeInTheDocument()
    })
  })

  it('renders discovery candidates even if Vault news is 404', async () => {
    vi.mocked(api.getEquityNews).mockRejectedValue({ status: 404, message: '404 Not Found' })
    vi.mocked(api.getEquityNewsDiscovery).mockResolvedValue({
      query_symbol: 'NFLX',
      items: [
        {
          headline: 'Netflix Subscriber Growth Tops Estimates in Q3',
          publisher: 'Bloomberg',
          source_type: 'RSS',
          article_url: 'https://bloomberg.com/nflx',
          published_at: '2026-10-17 12:00:00',
          discovered_at: Date.now(),
          is_stale: false,
        },
      ],
      status: 'ok',
      source: 'rss',
      as_of_date: '2026-10-17',
      fetched_at: Date.now(),
      limitations: [],
    })

    render(<EquityNews ticker="NFLX" />)

    await waitFor(() => {
      expect(screen.getByText('RSS News Discovery Candidates')).toBeInTheDocument()
      expect(screen.getByText('Netflix Subscriber Growth Tops Estimates in Q3')).toBeInTheDocument()
      expect(screen.getByText('Bloomberg')).toBeInTheDocument()
      expect(screen.getByText('ดูบทความต้นฉบับ ↗')).toBeInTheDocument()
      // Should NOT show the 404 empty state
      expect(screen.queryByText('ยังไม่มีข้อมูลข่าวสำหรับหุ้นตัวนี้ในระบบ')).not.toBeInTheDocument()
    })
  })
})
