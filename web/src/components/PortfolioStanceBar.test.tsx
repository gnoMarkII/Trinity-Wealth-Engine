import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { AssetAllocationDTO } from '../api/types'
import PortfolioStanceBar from './PortfolioStanceBar'

function makeAllocation(stance: string): AssetAllocationDTO {
  return {
    asset_class: 'Equities',
    asset_bucket: null,
    stance,
    confidence: 'medium',
    rationale: '',
    supporting_data: [],
    why_not_high: '',
    allocation_delta: '',
    invalidation_conditions: [],
    warnings: [],
  }
}

describe('PortfolioStanceBar', () => {
  it('นับจำนวน allocation ต่อหมวด stance ถูกต้อง', () => {
    render(
      <PortfolioStanceBar
        allocations={[
          makeAllocation('Overweight'),
          makeAllocation('OVERWEIGHT equities'),
          makeAllocation('Underweight'),
          makeAllocation('Neutral'),
        ]}
      />
    )
    expect(screen.getByText('Overweight (2)')).toBeInTheDocument()
    expect(screen.getByText('Underweight (1)')).toBeInTheDocument()
    expect(screen.getByText('Neutral (1)')).toBeInTheDocument()
  })

  it('allocations ว่าง → legend ยังแสดงศูนย์ครบทุกหมวด ไม่ crash', () => {
    render(<PortfolioStanceBar allocations={[]} />)
    expect(screen.getByText('Overweight (0)')).toBeInTheDocument()
    expect(screen.getByText('Neutral (0)')).toBeInTheDocument()
    expect(screen.getByText('Underweight (0)')).toBeInTheDocument()
  })
})
