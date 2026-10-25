import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { EventStudyTab } from './EventStudyTab'
import { api } from '../../api/client'
import type { EventStudyResponseDTO } from '../../api/types'

vi.mock('../../api/client', () => ({
  api: {
    getEventStudyPresets: vi.fn(),
    postEventStudy: vi.fn(),
  },
}))

describe('EventStudyTab', () => {
  const mockStudyResponse: EventStudyResponseDTO = {
    ticker: 'AAPL',
    event_name: 'Apple iPhone Launch Events',
    horizons: [5, 10, 20, 60],
    events: [
      {
        event_id: 'evt_1_2023-09-12',
        event_date: '2023-09-12',
        entry_date: '2023-09-13',
        entry_price: 175.5,
        description: 'iPhone 15 Series / USB-C',
        forward_returns: { '5': 3.25, '10': 5.12, '20': -1.2, '60': 8.4 },
        daily_trajectory: [
          { day_offset: 1, date: '2023-09-13', close: 176.0, cumulative_return_pct: 0.28 },
          { day_offset: 5, date: '2023-09-19', close: 181.2, cumulative_return_pct: 3.25 },
          { day_offset: 20, date: '2023-10-10', close: 173.4, cumulative_return_pct: -1.2 },
        ],
        is_win_by_horizon: { '5': true, '10': true, '20': false, '60': true },
      },
    ],
    horizon_stats: {
      '5': {
        horizon_days: 5,
        sample_size: 1,
        hit_rate_pct: 100.0,
        avg_return_pct: 3.25,
        median_return_pct: 3.25,
        best_return_pct: 3.25,
        worst_return_pct: 3.25,
        avg_win_pct: 3.25,
        avg_loss_pct: null,
        risk_reward_ratio: null,
        win_count: 1,
        loss_count: 0,
      },
      '20': {
        horizon_days: 20,
        sample_size: 1,
        hit_rate_pct: 0.0,
        avg_return_pct: -1.2,
        median_return_pct: -1.2,
        best_return_pct: -1.2,
        worst_return_pct: -1.2,
        avg_win_pct: null,
        avg_loss_pct: -1.2,
        risk_reward_ratio: null,
        win_count: 0,
        loss_count: 1,
      },
    },
    average_trajectory: [
      { day_offset: 1, avg_return_pct: 0.28, sample_count: 1 },
      { day_offset: 5, avg_return_pct: 3.25, sample_count: 1 },
    ],
    warnings: [
      {
        code: 'UNRELIABLE_SAMPLE_SIZE',
        message: 'Sample size is small (N=1 < 5). Historical patterns may lack statistical significance.',
      },
    ],
  }

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.getEventStudyPresets).mockResolvedValue([
      {
        preset_id: 'earnings',
        name: 'Earnings Announcements (PEAD)',
        category: 'earnings',
        description: 'Quarterly earnings releases',
        recommended_ticker: '',
      },
      {
        preset_id: 'apple_iphone',
        name: 'Apple iPhone Launch Events',
        category: 'product_launch',
        description: 'Historical announcement dates for iPhone keynotes',
        recommended_ticker: 'AAPL',
      },
      {
        preset_id: 'custom',
        name: 'Custom Event Dates',
        category: 'custom',
        description: 'User-defined event dates',
        recommended_ticker: '',
      },
    ])
    vi.mocked(api.postEventStudy).mockResolvedValue(mockStudyResponse)
  })

  it('renders presets dropdown and loads event study report', async () => {
    render(<EventStudyTab ticker="AAPL" market="US" />)

    // Check header
    expect(screen.getByText('Event Study Engine')).toBeInTheDocument()

    // Wait for study data to load
    await waitFor(() => {
      expect(screen.getByText('Apple iPhone Launch Events')).toBeInTheDocument()
    })

    // Verify 4 Horizon summary cards rendered
    expect(screen.getByText('ขอบเขต 5 วันทำการ')).toBeInTheDocument()
    expect(screen.getAllByText('+3.25%').length).toBeGreaterThan(0)

    // Verify sample size warning is displayed
    expect(screen.getByText(/UNRELIABLE_SAMPLE_SIZE/i)).toBeInTheDocument()

    // Verify statistics table header
    expect(screen.getByText('สถิติการกระจายตัวของผลตอบแทน (Statistical Distribution Summary)')).toBeInTheDocument()

    // Verify event instances ledger
    expect(screen.getByText('รายละเอียดเหตุการณ์ในอดีต (Event Instances Ledger)')).toBeInTheDocument()
    expect(screen.getByText('iPhone 15 Series / USB-C')).toBeInTheDocument()
  })

  it('switches active horizon when clicking horizon card', async () => {
    render(<EventStudyTab ticker="AAPL" market="US" />)

    await waitFor(() => {
      expect(screen.getByText('Apple iPhone Launch Events')).toBeInTheDocument()
    })

    const card5d = screen.getByText('ขอบเขต 5 วันทำการ').closest('button')
    expect(card5d).toBeInTheDocument()

    fireEvent.click(card5d!)

    // Header updates to 5 days
    expect(screen.getByText(/เส้นทางผลตอบแทนสะสม 5 วัน/i)).toBeInTheDocument()
  })

  it('handles custom date input mode', async () => {
    render(<EventStudyTab ticker="AAPL" market="US" />)

    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument()
    })

    const select = screen.getByRole('combobox')
    fireEvent.change(select, { target: { value: 'custom' } })

    expect(screen.getByPlaceholderText(/ชื่อเหตุการณ์/i)).toBeInTheDocument()
    expect(screen.getByPlaceholderText(/2024-01-15/i)).toBeInTheDocument()
  })
})
