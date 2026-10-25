import { render, screen, fireEvent } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import GoalFundingTensionSimulator from './GoalFundingTensionSimulator'

describe('GoalFundingTensionSimulator', () => {
  const mockLevers = {
    save_more_monthly_thb: 28938.92,
    total_monthly_savings_needed_thb: 48938.92,
    extend_timeline_months: 85,
    total_months_needed: 145,
    downsized_target_thb: 4878199.11,
    required_risk_cagr_pct: 17.9,
  }

  it('renders collapsed by default and expands on toggle click', () => {
    render(
      <GoalFundingTensionSimulator
        goalName="พอร์ตเพื่อการเกษียณ"
        targetAmountThb={6000000}
        currentAmountThb={1200000}
        monthlySavingsThb={20000}
        deadlineDaysLeft={1800}
        requiredReturnPct={17.9}
        tensionLevel="CRITICAL_TENSION"
        adjustmentLevers={mockLevers}
        expectedAnnualReturn={7.0}
      />
    )

    // Header banner is visible
    expect(screen.getByTestId('goal-funding-tension-simulator')).toBeInTheDocument()
    expect(screen.getByText(/เป้าหมายมีความตึงเครียดสูง/i)).toBeInTheDocument()
    expect(screen.getByText(/Risk Gap: \+10\.9% ต่อปี/i)).toBeInTheDocument()
    expect(screen.getAllByText(/จำลอง 4 คานงัด/i).length).toBeGreaterThanOrEqual(1)

    // 4 levers are collapsed initially
    expect(screen.queryByText(/1\. ออมเพิ่ม/i)).not.toBeInTheDocument()

    // Click header to expand
    fireEvent.click(screen.getByText(/เป้าหมายมีความตึงเครียดสูง/i))

    // Now 4 lever cards and preview drawer are visible
    expect(screen.getByText(/1\. ออมเพิ่ม/i)).toBeInTheDocument()
    expect(screen.getByText(/2\. ยืดเวลา/i)).toBeInTheDocument()
    expect(screen.getByText(/3\. ปรับเป้าหมาย/i)).toBeInTheDocument()
    expect(screen.getByText(/4\. รับเสี่ยงเพิ่ม/i)).toBeInTheDocument()
    expect(screen.getByText(/คานงัดที่ 1: ออมเพิ่มรายเดือน/i)).toBeInTheDocument()

    // Click collapse button
    fireEvent.click(screen.getByText(/หุบแบบจำลอง 4 คานงัด/i))
    expect(screen.queryByText(/1\. ออมเพิ่ม/i)).not.toBeInTheDocument()
  })

  it('allows clicking different levers when expanded to dynamically update the simulation preview', () => {
    render(
      <GoalFundingTensionSimulator
        goalName="พอร์ตเพื่อการเกษียณ"
        targetAmountThb={6000000}
        currentAmountThb={1200000}
        monthlySavingsThb={20000}
        deadlineDaysLeft={1800}
        requiredReturnPct={17.9}
        tensionLevel="CRITICAL_TENSION"
        adjustmentLevers={mockLevers}
        expectedAnnualReturn={7.0}
        defaultExpanded={true}
      />
    )

    // Checks 4 lever cards are visible immediately
    expect(screen.getByText(/1\. ออมเพิ่ม/i)).toBeInTheDocument()
    expect(screen.getByText(/2\. ยืดเวลา/i)).toBeInTheDocument()

    // Click Lever 2: Extend Timeline
    fireEvent.click(screen.getByText(/2\. ยืดเวลา/i))
    expect(screen.getByText(/คานงัดที่ 2: ขยายระยะเวลา/i)).toBeInTheDocument()
    expect(screen.getAllByText(/เวลาทบต้น/i).length).toBeGreaterThanOrEqual(1)

    // Click Lever 3: Downsize Target
    fireEvent.click(screen.getByText(/3\. ปรับเป้าหมาย/i))
    expect(screen.getByText(/คานงัดที่ 3: ปรับขนาดเป้าหมาย/i)).toBeInTheDocument()

    // Click Lever 4: Accept Risk
    fireEvent.click(screen.getByText(/4\. รับเสี่ยงเพิ่ม/i))
    expect(screen.getByText(/คานงัดที่ 4: รับความเสี่ยงเพิ่มขึ้น/i)).toBeInTheDocument()
    expect(screen.getByText(/⚠️ เสี่ยงสูงพิเศษ/i)).toBeInTheDocument()
  })
})
