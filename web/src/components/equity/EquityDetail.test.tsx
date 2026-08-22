import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { EquityDetail } from './EquityDetail'
import { mockEquityDetailAAPL } from '../../mocks/equity'
describe('EquityDetail', () => {
  it('renders loading state', () => {
    render(<EquityDetail status="loading" />)
    expect(screen.getByText('กำลังโหลดข้อมูล...')).toBeInTheDocument()
  })

  it('renders not-found state', () => {
    render(<EquityDetail status="not-found" />)
    expect(screen.getByText('ไม่พบข้อมูล')).toBeInTheDocument()
    expect(screen.getByText(/ยังไม่มีการวิเคราะห์สำหรับหุ้นตัวนี้/)).toBeInTheDocument()
  })

  it('renders error state with default message', () => {
    render(<EquityDetail status="error" />)
    expect(screen.getByText('เกิดข้อผิดพลาด')).toBeInTheDocument()
    expect(screen.getByText('ไม่สามารถโหลดข้อมูลได้')).toBeInTheDocument()
  })

  it('renders error state with custom message', () => {
    render(<EquityDetail status="error" errorMessage="Network Error" />)
    expect(screen.getByText('Network Error')).toBeInTheDocument()
  })

  it('renders detail view successfully', () => {
    render(<EquityDetail status="success" data={mockEquityDetailAAPL} />)
    
    expect(screen.getByText('AAPL')).toBeInTheDocument()
    expect(screen.getByText('(US)')).toBeInTheDocument()
    expect(screen.getByText('Apple Inc.')).toBeInTheDocument()
    
    expect(screen.getByText('Base Case Summary')).toBeInTheDocument()
    expect(screen.getByText(mockEquityDetailAAPL.base_case_summary)).toBeInTheDocument()
    
    expect(screen.getByText('bullish')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '📈 Chart' })).toBeInTheDocument()
  })
})

