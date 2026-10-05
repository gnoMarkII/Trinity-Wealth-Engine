import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../../../api/client'
import { SectorRotationDashboard } from './SectorRotationDashboard'

const coldResponse = (timeframe: 'daily' | 'weekly', tail: number) => ({
  capability_status: 'enabled' as const,
  refresh_state: 'running' as const,
  retry_after_seconds: 2,
  error_code: null,
  last_attempt_at: null,
  expected_session: '2026-10-01',
  freshness: 'unknown' as const,
  missing_sessions: 0,
  served_at: '2026-10-02T00:00:00Z',
  timeframe,
  tail,
  summary: null,
  snapshot: null,
})

const warmResponse = (timeframe: 'daily' | 'weekly', tail: number) => ({
  capability_status: 'enabled' as const,
  refresh_state: 'idle' as const,
  retry_after_seconds: null,
  error_code: null,
  last_attempt_at: '2026-10-02T10:00:00Z',
  expected_session: '2026-10-02',
  freshness: 'fresh' as const,
  missing_sessions: 0,
  served_at: '2026-10-02T10:15:00Z',
  timeframe,
  tail,
  summary: {
    summary_version: '1.0.0',
    timeframe,
    rotation_as_of: '2026-10-02',
    ranked_by_excess_3m: [],
    sector_breadth_3m: {
      outperforming: 5,
      valid_sectors: 10,
      expected_sectors: 11,
      status: 'complete' as const,
      as_of: '2026-10-02',
    },
    quadrant_members: { Leading: ['XLK'], Weakening: [], Lagging: ['XLE'], Improving: [] },
    periods_in_quadrant: { XLK: 3, XLE: 2 },
    elapsed_days_in_quadrant: { XLK: 14, XLE: 7 },
    momentum_delta: { XLK: 0.5, XLE: -0.2 },
    heading_deg: { XLK: 45.0, XLE: 210.0 },
  },
  snapshot: {
    snapshot_id: `sr_${timeframe}_active_revision`,
    input_digest: 'digest_123',
    as_of_date: '2026-10-02',
    coverage: { XLK: 252, XLE: 252 },
    available_sectors: 2,
    benchmark_status: 'available',
    rows: [
      {
        ticker: 'XLK',
        name: 'Technology',
        status: 'available',
        reason: null,
        price_as_of: '2026-10-02',
        rotation_as_of: '2026-10-02',
        relative_strength: 105.2,
        relative_trend: 104.5,
        relative_momentum: 102.1,
        quadrant: 'Leading',
        momentum_direction: 'rising',
        quadrant_changed_at: '2026-09-18',
        returns_pct: { '1W_absolute_pct': 2.5, '1W_excess_pp': 1.2, '1M_absolute_pct': 2.5, '1M_excess_pp': 1.2 },
        return_metrics: {
          '1W': {
            absolute_return_pct: 2.5,
            excess_return_pp: 1.2,
            relative_return_pct: 1.18,
            start_date: '2026-09-25',
            end_date: '2026-10-02',
            expected_sessions: 6,
            valid_sessions: 6,
            status: 'available',
            freshness: 'fresh',
            reason: null,
          },
          '1M': {
            absolute_return_pct: 2.5,
            excess_return_pp: 1.2,
            relative_return_pct: 1.18,
            start_date: '2026-09-02',
            end_date: '2026-10-02',
            expected_sessions: 22,
            valid_sessions: 22,
            status: 'available',
            freshness: 'fresh',
            reason: null,
          },
        },
        history: [{ as_of: '2026-10-02', relative_trend: 104.5, relative_momentum: 102.1, quadrant: 'Leading', status: 'available' }],
        relative_price_history: [{ as_of: '2026-10-02', sector_spy_rebased_100: 105.2, status: 'available' }],
        quadrant_transitions: [],
      },
      {
        ticker: 'XLE',
        name: 'Energy',
        status: 'partial',
        reason: 'missing_observations',
        price_as_of: '2026-10-02',
        rotation_as_of: '2026-10-02',
        relative_strength: 92.4,
        relative_trend: 96.0,
        relative_momentum: 94.0,
        quadrant: 'Lagging',
        momentum_direction: 'falling',
        quadrant_changed_at: null,
        returns_pct: { '1W_absolute_pct': null, '1W_excess_pp': null },
        return_metrics: {},
        history: [{ as_of: '2026-10-02', relative_trend: 96.0, relative_momentum: 94.0, quadrant: 'Lagging', status: 'available' }],
        relative_price_history: [],
        quadrant_transitions: [],
      },
    ],
  },
})

describe('SectorRotationDashboard', () => {
  afterEach(() => vi.restoreAllMocks())

  it('shows a cold refresh state, changes timeframe, and requests refresh explicitly', async () => {
    const getSectorRotation = vi.spyOn(api, 'getSectorRotation')
      .mockImplementation(async (timeframe = 'weekly', tail = 12) => coldResponse(timeframe, tail))
    const refreshSectorRotation = vi.spyOn(api, 'refreshSectorRotation')
      .mockImplementation(async (timeframe = 'weekly', tail = 12) => coldResponse(timeframe, tail))
    const user = userEvent.setup()

    render(<SectorRotationDashboard />)

    expect(await screen.findByText(/กำลังดึงข้อมูลย้อนหลังและคำนวณ/)).toBeInTheDocument()
    expect(getSectorRotation).toHaveBeenCalledWith('weekly', 12)

    await user.click(screen.getByRole('button', { name: /รายวัน \(Daily\)/ }))
    expect(getSectorRotation).toHaveBeenLastCalledWith('daily', 20)

    await user.click(screen.getByRole('button', { name: /Refresh/ }))
    expect(refreshSectorRotation).toHaveBeenCalledWith('daily', 20)
  })

  it('renders warm snapshot with quadrant map, formatted metrics, and handles missing observations with em-dash', async () => {
    vi.spyOn(api, 'getSectorRotation')
      .mockImplementation(async (timeframe = 'weekly', tail = 12) => warmResponse(timeframe, tail) as any)

    render(<SectorRotationDashboard />)

    // Verify quadrant cards render
    expect(await screen.findByText(/US Sector Rotation/i)).toBeInTheDocument()
    expect(screen.getAllByText('XLK').length).toBeGreaterThan(0)
    expect(screen.getAllByText('XLE').length).toBeGreaterThan(0)

    // Formatted metric: +1.20 pp excess
    expect(screen.getByText(/\+1\.20 pp/)).toBeInTheDocument()

    // Partial/missing metric on Energy renders '—' safely without crashing
    const dashes = screen.getAllByText('—')
    expect(dashes.length).toBeGreaterThan(0)
  })

  it('prevents stale responses from overriding newer requests when toggling timeframes quickly', async () => {
    let resolveWeekly: ((value: any) => void) | null = null
    const weeklyPromise = new Promise((resolve) => {
      resolveWeekly = resolve
    })

    vi.spyOn(api, 'getSectorRotation').mockImplementation(async (timeframe = 'weekly', tail = 12) => {
      if (timeframe === 'weekly') {
        return weeklyPromise as any
      }
      return warmResponse('daily', tail) as any
    })

    const user = userEvent.setup()
    render(<SectorRotationDashboard />)

    // Quickly switch to daily while weekly is still pending
    const dailyBtn = screen.getByRole('button', { name: /รายวัน \(Daily\)/ })
    await user.click(dailyBtn)

    // Daily renders first
    expect(await screen.findByText(/Technology/)).toBeInTheDocument()

    // Now resolve the slow weekly response
    if (resolveWeekly) {
      (resolveWeekly as (val: any) => void)(warmResponse('weekly', 12))
    }

    // Give microtasks time to execute
    await new Promise((r) => setTimeout(r, 50))

    // The active view must still be daily (not overridden by stale weekly response)
    expect(dailyBtn).toHaveClass('bg-white')
    expect(dailyBtn).toHaveClass('text-slate-900')
  })
})
