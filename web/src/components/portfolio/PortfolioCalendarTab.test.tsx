import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import PortfolioCalendarTab from './PortfolioCalendarTab'
import { api } from '../../api/client'

vi.mock('../../api/client', () => ({
  api: {
    getPortfolioCalendar: vi.fn(),
  },
}))

describe('PortfolioCalendarTab', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders loading skeleton initially', () => {
    vi.mocked(api.getPortfolioCalendar).mockReturnValue(new Promise(() => {}))
    const { container } = render(
      <MemoryRouter>
        <PortfolioCalendarTab />
      </MemoryRouter>
    )
    expect(container.querySelector('.animate-pulse')).toBeInTheDocument()
  })

  it('renders error message when API call fails', async () => {
    vi.mocked(api.getPortfolioCalendar).mockRejectedValue(new Error('Network error'))
    render(
      <MemoryRouter>
        <PortfolioCalendarTab />
      </MemoryRouter>
    )
    await waitFor(() => {
      expect(screen.getByText('เกิดข้อผิดพลาดในการโหลดปฏิทิน')).toBeInTheDocument()
      expect(screen.getByText('Network error')).toBeInTheDocument()
    })
  })

  it('renders calendar title and event chips on API success', async () => {
    const now = new Date()
    // Guarantee targetDate is within current month grid regardless of day-of-month
    const targetDate = new Date(now.getFullYear(), now.getMonth(), 15)
    const year = targetDate.getFullYear()
    const monthStr = String(targetDate.getMonth() + 1).padStart(2, '0')
    const dayStr = String(targetDate.getDate()).padStart(2, '0')
    const eventDateStr = `${year}-${monthStr}-${dayStr}`

    const mockData = {
      generated_at: '2026-08-06T00:00:00Z',
      events: [
        {
          ticker: 'AAPL',
          company_name: 'Apple Inc.',
          event_type: 'earnings' as const,
          event_date: eventDateStr,
          days_until: 5,
          bucket: 'holding' as const,
          eps_estimate: 1.98,
        },
      ],
      tickers_fetched: 1,
      tickers_failed: [],
    }


    vi.mocked(api.getPortfolioCalendar).mockResolvedValue(mockData)

    render(
      <MemoryRouter>
        <PortfolioCalendarTab />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(screen.getByText('📅 Corporate Events Calendar')).toBeInTheDocument()
      expect(screen.getAllByText('AAPL')[0]).toBeInTheDocument()
      expect(screen.getAllByText('อีก 5d')[0]).toBeInTheDocument()
    })

  })
})
