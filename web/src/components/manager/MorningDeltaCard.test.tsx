import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { MorningDeltaCard } from './MorningDeltaCard'
import type { MorningReviewPacketDTO } from '../../api/types'
import { api } from '../../api/client'
import { act } from '@testing-library/react'

const createDeferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej })
  return { promise, resolve, reject }
}

const mockPacket: MorningReviewPacketDTO = {
  review_date: '2026-11-02',
  status: 'REVIEW_REQUIRED',
  portfolio_delta: {
    total_nav: 1250000.0,
    daily_pnl_value: 8500.0,
    daily_pnl_pct: 0.68,
    cash_weight_pct: 14.5,
    bucket_drift_detected: false,
  },
  macro_delta: {
    yield_spread_10y_2y_bps: 16.0,
    yield_spread_24h_change_bps: 2.5,
    ofr_stress_index: -0.42,
    macro_regime_label: 'Late Cycle Expansion',
  },
  thesis_exceptions: [
    {
      exception_id: 'EXC-001',
      claim_id: 'TH-NVDA-v1',
      ticker: 'NVDA',
      detected_at_utc: '2026-11-02T10:00:00Z',
      exception_type: 'FALSIFIER_BREACHED',
      metric_name: 'gross_margin',
      observed_value: '68.0%',
    },
  ],
  pending_decisions: [
    {
      action: 'Acknowledge NVDA Margin Breach',
      urgency: 'HIGH',
    },
  ],
  packet_sha256: 'deadbeef1234567890abcdef',
  signoff_info: null,
}

describe('MorningDeltaCard Component', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('renders the 4 quadrants and REVIEW REQUIRED badge correctly', () => {
    render(<MorningDeltaCard initialPacket={mockPacket} />)

    // Check Header & Status
    expect(screen.getByText('Morning PM Review (Delta-Only)')).toBeInTheDocument()
    expect(screen.getByTestId('status-review-required')).toBeInTheDocument()

    // Quadrant 1: Portfolio
    expect(screen.getByText('📈 Portfolio Delta')).toBeInTheDocument()
    expect(screen.getByText('+0.68%')).toBeInTheDocument()
    expect(screen.getByText('฿1,250,000')).toBeInTheDocument()

    // Quadrant 2: Macro
    expect(screen.getByText('🌐 Macro Regime')).toBeInTheDocument()
    expect(screen.getAllByText('Late Cycle Expansion').length).toBeGreaterThan(0)
    expect(screen.getByText('16.0 bps')).toBeInTheDocument()

    // Quadrant 3: Thesis Exceptions
    expect(screen.getByText('⚠️ Thesis Watchdog')).toBeInTheDocument()
    expect(screen.getAllByText('1 ALERTS').length).toBeGreaterThan(0)
    expect(screen.getByText('$NVDA: FALSIFIER_BREACHED')).toBeInTheDocument()

    // Quadrant 4: Actions
    expect(screen.getByText('📋 PM Actions')).toBeInTheDocument()
    expect(screen.getByText('Acknowledge NVDA Margin Breach')).toBeInTheDocument()
  })

  it('renders SIGNED OFF badge when packet is signed off', () => {
    const signedOffPacket: MorningReviewPacketDTO = {
      ...mockPacket,
      status: 'SIGNED_OFF',
      signoff_info: {
        signoff_id: 'SIG-001',
        review_date: '2026-11-02',
        status: 'SIGNED_OFF',
        signed_off_by: 'CHINO_PM',
        signed_off_at_utc: '2026-11-02T16:00:00Z',
        acknowledged_exceptions_count: 1,
        notes: 'Verified allocations.',
        packet_sha256: 'deadbeef1234567890abcdef',
      },
    }

    render(<MorningDeltaCard initialPacket={signedOffPacket} />)
    expect(screen.getByTestId('status-signed-off')).toBeInTheDocument()
    expect(screen.getByText('SIGNED OFF (CHINO_PM)')).toBeInTheDocument()
  })

  it('opens sign-off modal and calls signOffMorningReview API upon confirmation', async () => {
    const signOffSpy = vi.spyOn(api, 'signOffMorningReview').mockResolvedValueOnce({
      signoff_id: 'SIG-NEW',
      review_date: '2026-11-02',
      status: 'SIGNED_OFF',
      signed_off_by: 'HUMAN_PM',
      signed_off_at_utc: '2026-11-02T17:00:00Z',
      acknowledged_exceptions_count: 1,
      notes: 'Approval note.',
      packet_sha256: 'deadbeef1234567890abcdef',
    })

    const onSignOffSuccess = vi.fn()
    render(<MorningDeltaCard initialPacket={mockPacket} onSignOffSuccess={onSignOffSuccess} />)

    // Click Sign-off button
    fireEvent.click(screen.getByText('Sign-off & Acknowledge'))

    // Modal opens
    expect(screen.getByText('Acknowledge & Sign-off')).toBeInTheDocument()

    // Type notes and click confirm
    const textarea = screen.getByPlaceholderText(/Acknowledged NVDA margin drift/i)
    fireEvent.change(textarea, { target: { value: 'Approval note.' } })

    fireEvent.click(screen.getByTestId('exception-ack-checkbox'))
    fireEvent.click(screen.getByText('Confirm Sign-off'))

    await waitFor(() => {
      expect(signOffSpy).toHaveBeenCalledWith({
        portfolio_id: 'default',
        review_date: '2026-11-02',
        signed_off_by: 'HUMAN_PM',
        notes: 'Approval note.',
        packet_sha256: 'deadbeef1234567890abcdef',
        verified_sample_ids: [],
        acknowledged_exception_ids: ['EXC-001'],
      })
      expect(onSignOffSuccess).toHaveBeenCalled()
    })
  })

  it('renders review sampling checklist in sign-off modal and supports Verify All', async () => {
    const samplingPacket: MorningReviewPacketDTO = {
      ...mockPacket,
      review_sample: [
        {
          sample_id: 'SMP-BREACH-NVDA',
          symbol: 'NVDA',
          sample_type: 'BREACH',
          headline: 'Thesis Exception (Gross Margin)',
          target_value: 'Observed 68% (Exp: 72%)',
          urgency: 'CRITICAL',
        },
        {
          sample_id: 'SMP-MOVER-AAPL',
          symbol: 'AAPL',
          sample_type: 'TOP_MOVER',
          headline: 'Top Daily PnL Mover',
          target_value: 'Daily movement: +3.2%',
          urgency: 'NORMAL',
        },
        {
          sample_id: 'SMP-ROTATION-MSFT',
          symbol: 'MSFT',
          sample_type: 'ROTATION',
          headline: 'Calendar Rotation Review',
          target_value: 'Periodic holding review',
          urgency: 'NORMAL',
        },
      ],
    }

    const signOffSpy = vi.spyOn(api, 'signOffMorningReview').mockResolvedValueOnce({
      signoff_id: 'SIG-SMP',
      review_date: '2026-11-02',
      status: 'SIGNED_OFF',
      signed_off_by: 'HUMAN_PM',
      signed_off_at_utc: '2026-11-02T17:00:00Z',
      acknowledged_exceptions_count: 1,
      notes: null,
      packet_sha256: 'deadbeef1234567890abcdef',
      verified_sample_ids: ['SMP-BREACH-NVDA', 'SMP-MOVER-AAPL', 'SMP-ROTATION-MSFT'],
    })

    render(<MorningDeltaCard initialPacket={samplingPacket} />)

    // Open Modal
    fireEvent.click(screen.getByText('Sign-off & Acknowledge'))

    // Checklist visible
    expect(screen.getByText('🔍 Institutional Sample Checks')).toBeInTheDocument()
    expect(screen.getByTestId('sample-item-SMP-BREACH-NVDA')).toBeInTheDocument()
    expect(screen.getByTestId('sample-item-SMP-MOVER-AAPL')).toBeInTheDocument()
    expect(screen.getByTestId('sample-item-SMP-ROTATION-MSFT')).toBeInTheDocument()
    expect(screen.getByText('0/3')).toBeInTheDocument()

    // Click Verify All
    fireEvent.click(screen.getByText('Verify All'))
    expect(screen.getByText('3/3')).toBeInTheDocument()
    expect(screen.getByText('Confirm Sign-off (3/3)')).toBeInTheDocument()

    // Confirm Sign-off
    fireEvent.click(screen.getByTestId('exception-ack-checkbox'))
    fireEvent.click(screen.getByText('Confirm Sign-off (3/3)'))

    await waitFor(() => {
      expect(signOffSpy).toHaveBeenCalledWith({
        portfolio_id: 'default',
        review_date: '2026-11-02',
        signed_off_by: 'HUMAN_PM',
        notes: undefined,
        packet_sha256: 'deadbeef1234567890abcdef',
        verified_sample_ids: ['SMP-BREACH-NVDA', 'SMP-MOVER-AAPL', 'SMP-ROTATION-MSFT'],
        acknowledged_exception_ids: ['EXC-001'],
      })
    })
  })

  it('toggles 4 quadrants tray between expanded and collapsed when button is clicked', () => {
    render(<MorningDeltaCard initialPacket={mockPacket} defaultExpanded={false} />)
    const toggleBtn = screen.getByTestId('toggle-quadrants-btn')
    const tray = screen.getByTestId('quadrants-tray')

    // Default is collapsed (hidden class applied)
    expect(toggleBtn).toHaveTextContent('4 Quadrants ▾')
    expect(tray).toHaveClass('hidden')

    // Click to expand
    fireEvent.click(toggleBtn)
    expect(toggleBtn).toHaveTextContent('ย่อ ▴')
    expect(tray).toHaveClass('block')

    // Click again to collapse
    fireEvent.click(toggleBtn)
    expect(toggleBtn).toHaveTextContent('4 Quadrants ▾')
    expect(tray).toHaveClass('hidden')
  })

  it('clears the previous portfolio packet while the new portfolio loads', async () => {
    const a = createDeferred<MorningReviewPacketDTO>()
    const b = createDeferred<MorningReviewPacketDTO>()
    vi.spyOn(api, 'getMorningReviewToday').mockReturnValueOnce(a.promise).mockReturnValueOnce(b.promise)

    const { rerender } = render(<MorningDeltaCard portfolioId="portfolio-a" />)
    const packetA = {
      ...mockPacket,
      portfolio_delta: { ...mockPacket.portfolio_delta, total_nav: 111 },
    }
    await act(async () => { a.resolve(packetA) })
    expect(screen.getByText(/111/)).toBeInTheDocument()

    rerender(<MorningDeltaCard portfolioId="portfolio-b" />)
    expect(screen.getByTestId('morning-delta-card-loading')).toBeInTheDocument()
    expect(screen.queryByText(/111/)).not.toBeInTheDocument()
  })

  it('ignores an older A-B-A response after the latest request has resolved', async () => {
    const a1 = createDeferred<MorningReviewPacketDTO>()
    const b = createDeferred<MorningReviewPacketDTO>()
    const a2 = createDeferred<MorningReviewPacketDTO>()
    vi.spyOn(api, 'getMorningReviewToday')
      .mockReturnValueOnce(a1.promise)
      .mockReturnValueOnce(b.promise)
      .mockReturnValueOnce(a2.promise)

    const { rerender } = render(<MorningDeltaCard portfolioId="portfolio-a" />)
    rerender(<MorningDeltaCard portfolioId="portfolio-b" />)
    rerender(<MorningDeltaCard portfolioId="portfolio-a" />)

    const packetA2 = {
      ...mockPacket,
      portfolio_delta: { ...mockPacket.portfolio_delta, total_nav: 333 },
    }
    await act(async () => { a2.resolve(packetA2) })
    expect(screen.getByText(/333/)).toBeInTheDocument()

    await act(async () => { a1.resolve({
      ...mockPacket,
      portfolio_delta: { ...mockPacket.portfolio_delta, total_nav: 111 },
    }) })
    expect(screen.getByText(/333/)).toBeInTheDocument()
    expect(screen.queryByText(/111/)).not.toBeInTheDocument()
  })

  it('renders unavailable daily P&L as N/A instead of zero', () => {
    const unavailablePacket: MorningReviewPacketDTO = {
      ...mockPacket,
      portfolio_delta: {
        ...mockPacket.portfolio_delta,
        daily_pnl_value: null,
        daily_pnl_pct: null,
      },
    }
    render(<MorningDeltaCard initialPacket={unavailablePacket} />)
    expect(screen.getByTestId('daily-pnl-value')).toHaveTextContent('N/A')
    expect(screen.getByTestId('daily-pnl-percent')).toHaveTextContent('N/A')
  })

  it('renders late-run warning and quarantined holdings alerts when present in packet', () => {
    const alertPacket: MorningReviewPacketDTO = {
      ...mockPacket,
      run_quality: 'LATE_RUN_STALE',
      drift_minutes: 240,
      quarantined_holdings: [
        {
          symbol: 'BROKEN',
          reason: 'Holding BROKEN has missing price',
          guardrail_code: 'MISSING_AS_OF_PRICE',
          current_value_thb: 0,
        },
      ],
    }

    render(<MorningDeltaCard initialPacket={alertPacket} />)
    expect(screen.getByTestId('late-run-warning')).toBeInTheDocument()
    expect(screen.getByText(/Late Run/)).toBeInTheDocument()
    expect(screen.getByTestId('quarantined-holdings-alert')).toBeInTheDocument()
    expect(screen.getByText(/\$BROKEN/)).toBeInTheDocument()
  })

  it('renders factor exposure and beta badges when present in portfolio_delta', () => {
    const factorPacket: MorningReviewPacketDTO = {
      ...mockPacket,
      portfolio_delta: {
        ...mockPacket.portfolio_delta,
        net_exposure_pct: 75.0,
        long_exposure_pct: 85.0,
        short_exposure_pct: 10.0,
        short_market_value: -25000,
        portfolio_beta: 1.15,
        beta_benchmark: 'SPY',
      },
    }

    render(<MorningDeltaCard initialPacket={factorPacket} />)
    expect(screen.getByTestId('factor-exposure-beta-badge')).toBeInTheDocument()
    expect(screen.getByText('75% Net')).toBeInTheDocument()
    expect(screen.getByText('(85% L / 10% S)')).toBeInTheDocument()
    expect(screen.getByTestId('portfolio-beta-badge')).toBeInTheDocument()
    expect(screen.getByText('1.15')).toBeInTheDocument()
    expect(screen.getByText('(SPY)')).toBeInTheDocument()
  })

  it('renders Chapter 6 Point-in-Time Macro Vintage Walk table with anti-lookahead badge', () => {
    const vintagePacket: MorningReviewPacketDTO = {
      ...mockPacket,
      macro_delta: {
        ...mockPacket.macro_delta,
        vintage_walk: [
          {
            series_id: 'SYN_PAYROLL',
            observation_date: '2026-04-01',
            cutoff_date: '2026-06-10',
            selected_vintage_date: '2026-06-05',
            asof_value: 148,
            latest_vintage_date: '2026-07-03',
            latest_value: 151,
            revision_after_cutoff: 3,
            unit: 'thousands',
            seasonal_adjustment: 'SA',
            status: 'REVIEW_REQUIRED',
          },
          {
            series_id: 'SYN_PAYROLL',
            observation_date: '2026-05-01',
            cutoff_date: '2026-06-10',
            selected_vintage_date: '2026-06-05',
            asof_value: 120,
            latest_vintage_date: '2026-07-03',
            latest_value: 126,
            revision_after_cutoff: 6,
            unit: 'thousands',
            seasonal_adjustment: 'SA',
            status: 'REVIEW_REQUIRED',
          },
        ],
      },
    }

    render(<MorningDeltaCard initialPacket={vintagePacket} />)
    expect(screen.getByTestId('macro-vintage-walk-section')).toBeInTheDocument()
    expect(screen.getByText('📈 Historical Vintage Walk')).toBeInTheDocument()
    expect(screen.getByText('🛡️ No Look-Ahead')).toBeInTheDocument()
    expect(screen.getAllByText('SYN_PAYROLL').length).toBe(2)
    expect(screen.getByText('148')).toBeInTheDocument()
    expect(screen.getByText('151')).toBeInTheDocument()
    expect(screen.getByText('+3')).toBeInTheDocument()
    expect(screen.getByText('120')).toBeInTheDocument()
    expect(screen.getByText('126')).toBeInTheDocument()
    expect(screen.getByText('+6')).toBeInTheDocument()
  })

  it('renders missed-run warning and routine fresh badges properly', () => {
    const missedPacket: MorningReviewPacketDTO = {
      ...mockPacket,
      run_quality: 'MISSED_RUN',
      drift_minutes: 180,
    }
    const { rerender } = render(<MorningDeltaCard initialPacket={missedPacket} />)
    expect(screen.getByTestId('missed-run-warning')).toBeInTheDocument()
    expect(screen.getByText(/MISSED_RUN: Scheduled routine was missed/i)).toBeInTheDocument()

    const freshPacket: MorningReviewPacketDTO = {
      ...mockPacket,
      run_quality: 'ON_TIME',
      drift_minutes: 2,
    }
    rerender(<MorningDeltaCard initialPacket={freshPacket} />)
    expect(screen.getByTestId('routine-fresh-badge')).toBeInTheDocument()
    expect(screen.getByText(/Routine Fresh/i)).toBeInTheDocument()
  })

  it('renders catalyst exception with date change and allows PM to record materiality assessment', async () => {
    const reviewSpy = vi.spyOn(api, 'reviewCatalystException').mockResolvedValueOnce({
      exception_id: 'EXC-CAT-01',
      catalyst_id: 'CAT-01',
      entity_id: 'GOOGL',
      classification: 'DATE_CHANGED',
      source_id: 'SYN-IR-001',
      observed_fact: 'Earnings call rescheduled',
      prior_date: '2026-10-24',
      new_date: '2026-10-29',
      analyst_field: 'materiality_assessment',
      human_assessment: 'Non-material schedule shift',
      status: 'ACKNOWLEDGED',
      detected_utc: '2026-11-05T06:00:00Z',
    })

    const catalystPacket: MorningReviewPacketDTO = {
      ...mockPacket,
      thesis_exceptions: [
        {
          exception_id: 'EXC-CAT-01',
          catalyst_id: 'CAT-01',
          entity_id: 'GOOGL',
          classification: 'DATE_CHANGED',
          source_id: 'SYN-IR-001',
          observed_fact: 'Earnings call rescheduled by 5 days',
          prior_date: '2026-10-24',
          new_date: '2026-10-29',
          analyst_field: 'materiality_assessment',
          human_assessment: null,
          status: 'REVIEW_REQUIRED',
          detected_utc: '2026-11-05T06:00:00Z',
        } as any,
      ],
    }

    render(<MorningDeltaCard initialPacket={catalystPacket} />)

    expect(screen.getByTestId('catalyst-exception-EXC-CAT-01')).toBeInTheDocument()
    expect(screen.getByText('Earnings call rescheduled by 5 days')).toBeInTheDocument()
    expect(screen.getByText('2026-10-24')).toBeInTheDocument()
    expect(screen.getByText('2026-10-29')).toBeInTheDocument()

    // Open assessment input
    fireEvent.click(screen.getByTestId('open-assessment-btn-EXC-CAT-01'))
    const input = screen.getByTestId('assessment-input-EXC-CAT-01')
    fireEvent.change(input, { target: { value: 'Non-material schedule shift' } })

    // Save assessment
    fireEvent.click(screen.getByTestId('save-assessment-btn-EXC-CAT-01'))

    await waitFor(() => {
      expect(reviewSpy).toHaveBeenCalledWith('EXC-CAT-01', 'Non-material schedule shift')
    })
  })
})
