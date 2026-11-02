import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ManagerResearchJobCard from './ManagerResearchJobCard'

describe('ManagerResearchJobCard', () => {
  it('renders compact active running indicator with pulse beacon and timer', () => {
    render(
      <ManagerResearchJobCard
        jobId="job_nvda_123"
        ticker="NVDA"
        jobType="equity_intel"
        status="queued"
      />
    )

    expect(screen.getByText(/AI กำลังวิเคราะห์หุ้น NVDA/i)).toBeInTheDocument()
    expect(screen.getByText(/อยู่ในคิว/i)).toBeInTheDocument()
    expect(screen.getByText(/ขั้นตอน:/i)).toBeInTheDocument()
    expect(screen.getByText(/0s/i)).toBeInTheDocument()
  })

  it('renders completed status and report view correctly', () => {
    render(
      <ManagerResearchJobCard
        jobId="job_macro_456"
        ticker="MACRO"
        jobType="macro_intel"
        status="completed"
      />
    )

    expect(screen.getByText(/รายงานผลการวิเคราะห์สภาวะเศรษฐกิจมหภาค/i)).toBeInTheDocument()
    expect(screen.getByText(/วิจัยเสร็จสมบูรณ์/i)).toBeInTheDocument()
    expect(screen.getByText(/job_macro_456/i)).toBeInTheDocument()
    expect(screen.getByText(/คัดลอก/i)).toBeInTheDocument()
  })
})
