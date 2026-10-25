import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import ExpectationGapCard from './ExpectationGapCard'

describe('ExpectationGapCard', () => {
  it('renders comparative growth and margin statistics correctly', () => {
    render(
      <ExpectationGapCard
        marketImpliedRevenueGrowth={10.39}
        actualRevenueGrowth={15.85}
        marketImpliedOperatingMargin={23.77}
        actualOperatingMargin={29.68}
        solverStatus="converged"
        pvOf5yFcf={67.26e9}
        pvOfTerminalValue={236.94e9}
        enterpriseValue={304.20e9}
        currency="$"
      />
    )

    expect(screen.getByText('Reverse DCF: Market Expectation vs Reality Gap')).toBeInTheDocument()
    expect(screen.getByText('Solver: converged')).toBeInTheDocument()

    // Revenue Growth
    expect(screen.getByText('10.39%')).toBeInTheDocument()
    expect(screen.getByText('15.85%')).toBeInTheDocument()
    expect(screen.getByText('+5.46% Beat')).toBeInTheDocument()

    // Operating Margin
    expect(screen.getByText('23.77%')).toBeInTheDocument()
    expect(screen.getByText('29.68%')).toBeInTheDocument()
    expect(screen.getByText('+5.91% Beat')).toBeInTheDocument()

    // Enterprise Value Breakdown
    expect(screen.getByText(/Enterprise Value Breakdown: \$304.20B/)).toBeInTheDocument()
  })

  it('renders nothing when growth and margin data are both absent', () => {
    const { container } = render(<ExpectationGapCard />)
    expect(container.firstChild).toBeNull()
  })
})
