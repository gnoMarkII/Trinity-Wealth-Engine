import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import PriceSpectrumGauge from './PriceSpectrumGauge'

describe('PriceSpectrumGauge', () => {
  it('renders correctly with given price, target, and buy zone', () => {
    render(
      <PriceSpectrumGauge
        currentPrice={67.05}
        targetPrice12m={77.36}
        upside12mPct={15.4}
        intrinsicValue={71.40}
        buyZoneMin={63.64}
        buyZoneMax={66.67}
        isInBuyZone={false}
        stopLoss={63.14}
        currency="$"
        actionStance="HOLD_WAIT"
        stanceReason="Business Conviction ปานกลาง — ถือหรือรอความชัดเจน"
      />
    )

    // Current Price Big Number Callout
    expect(screen.getByText('Current Close Price')).toBeInTheDocument()
    expect(screen.getByText('$67.05')).toBeInTheDocument()

    // 12M Target Price & Upside
    expect(screen.getByText('12M Target Price (Reverse DCF)')).toBeInTheDocument()
    expect(screen.getByText('$77.36')).toBeInTheDocument()
    expect(screen.getByText('+15.4% Upside')).toBeInTheDocument()

    // Action Stance
    expect(screen.getByText('HOLD WAIT')).toBeInTheDocument()

    // Spectrum Markers
    expect(screen.getByText(/Stop \$63.14/)).toBeInTheDocument()
    expect(screen.getByText(/Buy \$63.64/)).toBeInTheDocument()
    expect(screen.getByText(/Max \$66.67/)).toBeInTheDocument()
    expect(screen.getByText(/Target \$77.36/)).toBeInTheDocument()
  })

  it('renders nothing when currentPrice is 0 or NaN', () => {
    const { container } = render(<PriceSpectrumGauge currentPrice={0} />)
    expect(container.firstChild).toBeNull()
  })
})
