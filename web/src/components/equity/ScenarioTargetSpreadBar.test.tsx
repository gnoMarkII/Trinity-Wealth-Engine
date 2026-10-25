import { render, screen, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import ScenarioTargetSpreadBar, { type ScenarioItem } from './ScenarioTargetSpreadBar'

describe('ScenarioTargetSpreadBar', () => {
  const mockScenarios: ScenarioItem[] = [
    {
      key: 'bear',
      label: 'Bear Case',
      color: 'bg-rose-500',
      textCol: 'text-rose-700',
      bgCol: 'bg-rose-50 border-rose-200',
      data: {
        target_price: 62.5,
        upside_pct: -6.8,
        margin_of_safety_pct: -7.2,
        implied_exit_ebitda_multiple: 16.5,
      },
    },
    {
      key: 'base',
      label: 'Base Case',
      color: 'bg-sky-500',
      textCol: 'text-sky-700',
      bgCol: 'bg-sky-50 border-sky-200',
      data: {
        target_price: 77.36,
        upside_pct: 15.4,
        margin_of_safety_pct: 13.3,
        implied_exit_ebitda_multiple: 19.8,
      },
    },
  ]

  it('renders scenarios and handles selection callback', () => {
    const handleSelect = vi.fn()
    render(
      <ScenarioTargetSpreadBar
        scenarios={mockScenarios}
        selectedScenarioKey="base"
        onSelectScenario={handleSelect}
        currentPrice={67.05}
        currency="$"
      />
    )

    expect(screen.getByTestId('scenario-target-spread-bar')).toBeInTheDocument()
    expect(screen.getByText('$62.50')).toBeInTheDocument()
    expect(screen.getByText('$77.36')).toBeInTheDocument()

    fireEvent.click(screen.getByText('Bear Case'))
    expect(handleSelect).toHaveBeenCalledWith('bear')
  })
})
