import { render, screen, fireEvent } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { ActualGoalItemDTO } from '../../api/types'
import PortfolioGoalsTab from './PortfolioGoalsTab'

describe('PortfolioGoalsTab', () => {
  const mockGoals: ActualGoalItemDTO[] = [
    {
      name: 'Emergency Fund',
      goal_type: 'cash_target',
      portfolio_id: 'default',
      target_amount_thb: 300000,
      current_amount_thb: 150000,
      progress_pct: 50.0,
      bucket_id: 'CASH',
      deadline: '2027-12-31',
      deadline_days_left: 440,
      notes: 'เงินสำรอง 6 เดือน',
      priority_tier: 'needs',
      monthly_savings_thb: 5000,
      required_return_pct: 0.0,
      tension_level: 'BALANCED',
    },
    {
      name: 'Retirement Growth',
      goal_type: 'bucket_target',
      portfolio_id: 'default',
      target_amount_thb: 1000000,
      current_amount_thb: 600000,
      progress_pct: 60.0,
      bucket_id: 'GROWTH',
      deadline: null,
      deadline_days_left: null,
      notes: null,
      priority_tier: 'wants',
      monthly_savings_thb: 0,
      required_return_pct: null,
      tension_level: 'INSUFFICIENT_DATA',
    },
    {
      name: 'Dream Vacation Villa',
      goal_type: 'nav_target',
      portfolio_id: 'default',
      target_amount_thb: 5000000,
      current_amount_thb: 500000,
      progress_pct: 10.0,
      bucket_id: null,
      deadline: '2028-12-31',
      deadline_days_left: 700,
      notes: 'วิลล่าพักผ่อนริมทะเล',
      priority_tier: 'wishes',
      monthly_savings_thb: 20000,
      required_return_pct: 28.5,
      tension_level: 'CRITICAL_TENSION',
      adjustment_levers: {
        save_more_monthly_thb: 35000,
        total_monthly_savings_needed_thb: 55000,
        extend_timeline_months: 24,
        total_months_needed: 47,
        downsized_target_thb: 2800000,
        required_risk_cagr_pct: 28.5,
      },
    },
  ]



  const mockPolicy = {
    purpose_buckets: [
      {
        bucket_id: 'CASH',
        name: 'Liquidity Reserve',
        role: 'Emergency Cash',
        target_percent: 20,
      },
      {
        bucket_id: 'GROWTH',
        name: 'Long-term Core',
        role: 'Capital Appreciation',
        target_percent: 60,
      },
    ],
  }

  const mockBucketSummaries = [
    {
      bucket_id: 'CASH',
      name: 'Liquidity Reserve',
      target_percent: 20,
      actual_value_thb: 150000,
      actual_percent: 15.0,
      variance: -5.0,
      color: '#06B6D4',
    },
    {
      bucket_id: 'GROWTH',
      name: 'Long-term Core',
      target_percent: 60,
      actual_value_thb: 600000,
      actual_percent: 60.0,
      variance: 0.0,
      color: '#3B82F6',
    },
  ]

  it('renders goals list and summary details', () => {
    render(
      <PortfolioGoalsTab
        goals={mockGoals}
        generatedAt="2026-10-10T12:00:00Z"
        selectedPortfolioId="default"
      />
    )

    expect(screen.getByText(/Portfolio Goals \(3 เป้าหมาย\)/i)).toBeInTheDocument()
    expect(screen.getByText('Emergency Fund')).toBeInTheDocument()
    expect(screen.getByText('Retirement Growth')).toBeInTheDocument()
    expect(screen.getByText('Dream Vacation Villa')).toBeInTheDocument()
    expect(screen.getByText('50.0%')).toBeInTheDocument()
    expect(screen.getByText('60.0%')).toBeInTheDocument()
    expect(screen.getByText('10.0%')).toBeInTheDocument()

    // Priority Tier badges
    expect(screen.getByText(/Needs \(ล้มเหลวไม่ได้\)/i)).toBeInTheDocument()
    expect(screen.getByText(/Wants \(สำคัญมาก\)/i)).toBeInTheDocument()
    expect(screen.getByText(/Wishes \(เสี่ยงสูงได้\)/i)).toBeInTheDocument()

    // Required Return & Tension 4 Levers
    expect(screen.getByText(/0.0% \(ออมครบตามแผน\)/i)).toBeInTheDocument()
    expect(screen.getAllByText(/28.5% ต่อปี/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText(/เป้าหมายมีความตึงเครียดสูง/i)).toBeInTheDocument()
    // Collapsed by default; click to expand simulator
    fireEvent.click(screen.getByText(/เป้าหมายมีความตึงเครียดสูง/i))
    expect(screen.getAllByText(/1. ออมเพิ่ม/i).length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText(/2. ยืดเวลา/i).length).toBeGreaterThanOrEqual(1)
  })


  it('renders purpose bucket attribution and funding share when policy and summaries are provided', () => {
    render(
      <PortfolioGoalsTab
        goals={mockGoals}
        generatedAt="2026-10-10T12:00:00Z"
        selectedPortfolioId="default"
        allocationPolicy={mockPolicy}
        bucketSummaries={mockBucketSummaries}
      />
    )

    expect(screen.getByText('Liquidity Reserve')).toBeInTheDocument()
    expect(screen.getByText('Emergency Cash')).toBeInTheDocument()
    expect(screen.getByText('Long-term Core')).toBeInTheDocument()
    expect(screen.getByText('Capital Appreciation')).toBeInTheDocument()

    expect(screen.getByText(/สัดส่วนในพอร์ต: 15.0%/i)).toBeInTheDocument()
    expect(screen.getByText(/ทุนสนับสนุนเป้าหมาย: 50.0%/i)).toBeInTheDocument()
    expect(screen.getByText(/สัดส่วนในพอร์ต: 60.0%/i)).toBeInTheDocument()
    expect(screen.getByText(/ทุนสนับสนุนเป้าหมาย: 60.0%/i)).toBeInTheDocument()
  })
})
