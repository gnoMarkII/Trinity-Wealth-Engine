import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import TacticalStageCycleBar from './TacticalStageCycleBar'

describe('TacticalStageCycleBar', () => {
  it('highlights stage 4 when stock is in markdown', () => {
    render(<TacticalStageCycleBar currentStage="STAGE_4_MARKDOWN" />)

    expect(screen.getByTestId('tactical-stage-cycle-bar')).toBeInTheDocument()
    expect(screen.getByText('STAGE 4 MARKDOWN')).toBeInTheDocument()
    const stage4Step = screen.getByTestId('stage-step-4')
    expect(stage4Step).toHaveClass('bg-rose-600')
  })

  it('highlights stage 2 when stock is in markup', () => {
    render(<TacticalStageCycleBar currentStage="STAGE_2_MARKUP" />)

    const stage2Step = screen.getByTestId('stage-step-2')
    expect(stage2Step).toHaveClass('bg-emerald-600')
  })
})
