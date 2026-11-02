import { render, screen, fireEvent } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import HarnessTelemetryWidget from './HarnessTelemetryWidget'

afterEach(() => {
  vi.restoreAllMocks()
})

describe('HarnessTelemetryWidget', () => {
  it('renders healthy telemetry stats when data is loaded', async () => {
    vi.spyOn(api, 'getHarnessTelemetry').mockResolvedValue({
      total_events: 5,
      resolved_rate: 1.0,
      issue_counts: { SUPPORTING_DATA_MISMATCH: 2 },
      top_recurring_issues: [['SUPPORTING_DATA_MISMATCH', 2]],
    })

    render(<HarnessTelemetryWidget />)

    expect(await screen.findByText('Harness AI Health')).toBeInTheDocument()
    expect(screen.getByText('100% Clean')).toBeInTheDocument()
    expect(screen.getByText('5')).toBeInTheDocument()
    expect(screen.getByText('Recurring Patterns')).toBeInTheDocument()
  })

  it('renders Nominal status when 0 events recorded', async () => {
    vi.spyOn(api, 'getHarnessTelemetry').mockResolvedValue({
      total_events: 0,
      resolved_rate: 0.0,
      issue_counts: {},
      top_recurring_issues: [],
    })

    render(<HarnessTelemetryWidget />)

    expect(await screen.findByText('Harness AI Health')).toBeInTheDocument()
    expect(screen.getByText('Nominal')).toBeInTheDocument()
    expect(screen.getByText('0')).toBeInTheDocument()
  })

  it('opens detailed telemetry modal on click and closes it', async () => {
    vi.spyOn(api, 'getHarnessTelemetry').mockResolvedValue({
      total_events: 8,
      resolved_rate: 0.875,
      issue_counts: { SUPPORTING_DATA_MISMATCH: 1 },
      top_recurring_issues: [['SUPPORTING_DATA_MISMATCH', 1]],
    })

    render(<HarnessTelemetryWidget />)

    expect(await screen.findByText('Harness AI Health')).toBeInTheDocument()
    const detailBtn = screen.getByText('สถิติเชิงลึก (Telemetry)')
    fireEvent.click(detailBtn)

    expect(screen.getByText('AI Harness & Learning Telemetry')).toBeInTheDocument()
    expect(screen.getByText('ความถี่ของข้อบกพร่อง (Pareto Analysis)')).toBeInTheDocument()

    const closeBtn = screen.getByText('ปิดหน้าต่าง')
    fireEvent.click(closeBtn)

    expect(screen.queryByText('AI Harness & Learning Telemetry')).not.toBeInTheDocument()
  })
})
