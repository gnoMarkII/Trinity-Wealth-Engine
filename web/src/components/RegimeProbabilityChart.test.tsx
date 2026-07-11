import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import RegimeProbabilityChart from './RegimeProbabilityChart'

describe('RegimeProbabilityChart', () => {
  it('เรียง regime ตามลำดับมาตรฐานก่อน แล้วต่อด้วยชื่อที่ไม่รู้จัก', () => {
    render(
      <RegimeProbabilityChart
        probabilities={{ Recession: 0.1, Goldilocks: 0.4, CustomRegime: 0.5 }}
      />
    )
    const labels = screen.getAllByText(/Goldilocks|Recession|CustomRegime/).map((el) => el.textContent)
    expect(labels).toEqual(['Goldilocks', 'Recession', 'CustomRegime'])
  })

  it('แปลงความน่าจะเป็นเป็นเปอร์เซ็นต์ปัดเศษ', () => {
    render(<RegimeProbabilityChart probabilities={{ Goldilocks: 0.456 }} />)
    expect(screen.getByText('46%')).toBeInTheDocument()
  })

  it('probabilities ว่าง → ไม่ crash', () => {
    const { container } = render(<RegimeProbabilityChart probabilities={{}} />)
    expect(container.firstChild).not.toBeNull()
  })
})
