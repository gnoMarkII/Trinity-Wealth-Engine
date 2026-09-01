import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { DataQualityFlagsCard } from './DataQualityFlagsCard'

describe('DataQualityFlagsCard', () => {
  const mockFlags = [
    'hardcoded_us_risk_free:dcf',
    'eps_proxy_base_growth:dcf',
    'hardcoded_cost_of_debt:dcf',
  ]

  it('renders consolidated flags and summary chips in compact panel', () => {
    render(<DataQualityFlagsCard flags={mockFlags} />)

    expect(screen.getByText('Data Quality & Model Assumptions')).toBeInTheDocument()
    expect(screen.getByText('3')).toBeInTheDocument()
    expect(screen.getByText(/1 Fallback/)).toBeInTheDocument()
    expect(screen.getByText(/1 Methodology/)).toBeInTheDocument()
    expect(screen.getByText(/1 Note/)).toBeInTheDocument()
  })

  it('toggles expand and collapse view on click', () => {
    render(<DataQualityFlagsCard flags={mockFlags} />)

    const toggleBtn = screen.getByRole('button', { name: /ย่อแถบ/ })
    fireEvent.click(toggleBtn)

    expect(screen.getByRole('button', { name: /ขยายดูรายละเอียด/ })).toBeInTheDocument()
  })

  it('returns null when flags are empty', () => {
    const { container } = render(<DataQualityFlagsCard flags={[]} />)
    expect(container.firstChild).toBeNull()
  })
})
