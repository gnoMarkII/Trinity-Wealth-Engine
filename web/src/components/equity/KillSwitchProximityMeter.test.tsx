import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import KillSwitchProximityMeter from './KillSwitchProximityMeter'

describe('KillSwitchProximityMeter', () => {
  const mockFalsifiers = [
    {
      falsifier_id: 'NFLX_MARGIN_FLOOR',
      metric_name: 'Operating Margin (EBIT %)',
      condition: 'EBIT margin falls below 27.2%',
      threshold_value: 27.2,
      source_ref: 'json-pointer:///fixed_parameters/base_ebit_margin_pct',
      narrative_explanation: 'หากอัตรากำไรลดลงต่ำกว่า 27.2% แสดงว่าสูญเสีย Pricing Power',
    },
    {
      falsifier_id: 'NFLX_STOP_LOSS_BREAK',
      metric_name: 'Structural Support Breakdown',
      condition: 'Weekly close below $63.14',
      threshold_value: 63.14,
      source_ref: 'json-pointer:///tactical_setup/invalidation_stop_loss',
      narrative_explanation: 'หากราคาหลุดแนวรับสำคัญที่ $63.14 จะเข้าสู่แนวโน้มขาลง',
    },
  ]

  it('renders safety buffers, thresholds, and humanized chips', () => {
    render(
      <KillSwitchProximityMeter
        falsifiers={mockFalsifiers}
        currentOperatingMargin={29.68}
        currentPrice={67.05}
        currency="$"
      />
    )

    expect(screen.getByText('Thesis Falsifiers & Invalidation Criteria')).toBeInTheDocument()
    expect(screen.getByText('2 Kill-Switches Active')).toBeInTheDocument()

    // EBIT Margin Buffer (29.68 - 27.2 = +2.48%)
    expect(screen.getByText('+2.48% Buffer')).toBeInTheDocument()
    expect(screen.getByText('29.68%')).toBeInTheDocument()
    expect(screen.getByText('< 27.2%')).toBeInTheDocument()

    // Stop Loss Distance ($67.05 - $63.14 = $3.91)
    expect(screen.getByText('+$3.91 (6.2% buffer)')).toBeInTheDocument()
    expect(screen.getByText('$67.05')).toBeInTheDocument()
    expect(screen.getByText('< $63.14')).toBeInTheDocument()

    // Humanized audit chips
    expect(screen.getByText('📄 GAAP 10-Q Filing')).toBeInTheDocument()
    expect(screen.getByText('📈 Weekly Pivot S1 (Technical)')).toBeInTheDocument()
  })

  it('renders nothing when falsifiers list is empty', () => {
    const { container } = render(<KillSwitchProximityMeter falsifiers={[]} />)
    expect(container.firstChild).toBeNull()
  })
})
