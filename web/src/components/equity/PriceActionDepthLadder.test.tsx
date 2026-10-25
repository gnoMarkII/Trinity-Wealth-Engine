import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import PriceActionDepthLadder from './PriceActionDepthLadder'

describe('PriceActionDepthLadder', () => {
  it('renders resistance, current price, buy zone and support accurately', () => {
    render(
      <PriceActionDepthLadder
        currentPrice={67.05}
        keyResistance={77.36}
        keySupport={63.14}
        buyZoneMin={63.64}
        buyZoneMax={66.67}
        stopLoss={63.14}
        currency="$"
      />
    )

    expect(screen.getByTestId('price-action-depth-ladder')).toBeInTheDocument()
    expect(screen.getByText('$67.05')).toBeInTheDocument()
    expect(screen.getByText('$77.36')).toBeInTheDocument()
    expect(screen.getByText('$63.14')).toBeInTheDocument()
    expect(screen.getByText('$63.64 - $66.67')).toBeInTheDocument()
  })
})
