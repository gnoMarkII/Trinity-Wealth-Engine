import { render, screen, fireEvent } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import PortfolioOverviewTab from './PortfolioOverviewTab'

describe('PortfolioOverviewTab', () => {
  const targets = [
    { bucket_id: 'CORE', name: 'Core Growth', target_percent: 60, color: null },
    { bucket_id: 'DIV', name: 'Dividend Shield', target_percent: 40, color: null },
  ]

  const summaries = [
    {
      bucket_id: 'CORE',
      name: 'Core Growth',
      target_percent: 60,
      actual_value_thb: 600000,
      actual_percent: 60,
      variance: 0,
      color: null,
    },
    {
      bucket_id: 'DIV',
      name: 'Dividend Shield',
      target_percent: 40,
      actual_value_thb: 400000,
      actual_percent: 40,
      variance: 0,
      color: null,
    },
  ]

  it('renders warning badge when warning is provided', () => {
    render(
      <PortfolioOverviewTab
        portfolioId="default"
        targets={targets}
        summaries={summaries}
        warning="ผลรวมสัดส่วนเป้าหมายเท่ากับ 100% (ปกติ)"
        onSelectBucket={vi.fn()}
      />
    )
    expect(screen.getByText(/ผลรวมสัดส่วนเป้าหมายเท่ากับ 100% \(ปกติ\)/i)).toBeInTheDocument()
  })

  it('renders concentric donut chart and strategy buckets table', () => {
    render(
      <PortfolioOverviewTab portfolioId="default" targets={targets} summaries={summaries} warning={null} onSelectBucket={vi.fn()} />
    )
    expect(screen.getByText('Allocation Rings (Target vs Actual)')).toBeInTheDocument()
    expect(screen.getByText('Strategy Buckets Breakdown')).toBeInTheDocument()
    expect(screen.getAllByText('Core Growth').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Dividend Shield').length).toBeGreaterThanOrEqual(1)
  })

  it('triggers onSelectBucket callback when bucket row in table is clicked', () => {
    const onSelectMock = vi.fn()
    render(
      <PortfolioOverviewTab portfolioId="default" targets={targets} summaries={summaries} warning={null} onSelectBucket={onSelectMock} />
    )

    const coreElements = screen.getAllByText('Core Growth')
    const row = coreElements[0]?.closest('tr')
    expect(row).toBeDefined()
    if (row) fireEvent.click(row)


    expect(onSelectMock).toHaveBeenCalledWith('CORE')
  })

  it('renders applied policy banner and provenance badges when policy is provided', () => {
    const policy = {
      draft_id: 'draft-99',
      allocation_basis: 'nav_percent',
      applied_at_iso: '2026-10-10T12:00:00Z',
      purpose_buckets: [
        {
          bucket_id: 'CORE',
          name: 'Core Growth',
          role: 'Long-term Growth',
          target_percent: 60,
          cash_inflow_routing_pct: '70.0',
          minimum_amount: '500000',
        },
      ],
    }
    const origin = {
      applied_draft_id: 'draft-99',
      command_id: 'cmd-1',
      essence_ref: 'ess-snap-12345678',
      axis_ref: 'axis-snap-12345678',
      context_ref: 'ctx-1',
      is_modified_since_apply: true,
    }

    render(
      <PortfolioOverviewTab
        portfolioId="default"
        targets={targets}
        summaries={summaries}
        warning={null}
        allocationPolicy={policy}
        allocationPlanOrigin={origin}
        onSelectBucket={vi.fn()}
      />
    )

    expect(screen.getByText(/นโยบายกลยุทธ์ที่ใช้งานอยู่/i)).toBeInTheDocument()
    expect(screen.getByText(/มีการแก้ไข Targets เพิ่มเติม/i)).toBeInTheDocument()
    expect(screen.getByText(/Essence: ess-snap-12345/i)).toBeInTheDocument()
    expect(screen.getByText(/Axis: axis-snap-1234/i)).toBeInTheDocument()
  })


  it('renders strategy buckets breakdown without monthly inflow routing or funding gap table', () => {
    render(
      <PortfolioOverviewTab
        portfolioId="default"
        targets={targets}
        summaries={summaries}
        warning={null}
        onSelectBucket={vi.fn()}
      />
    )

    expect(screen.queryByText(/แผนจัดสรรเงินเข้าใหม่รายเดือน/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/สถานะเงินทุนและเงินสำรอง/i)).not.toBeInTheDocument()
    expect(screen.getByText(/Strategy Buckets Breakdown/i)).toBeInTheDocument()
  })

  it('renders rebalance review alert when variance exceeds threshold', () => {
    const highVarSummaries = [
      {
        bucket_id: 'CORE',
        name: 'Core Growth',
        target_percent: 60,
        actual_value_thb: 700000,
        actual_percent: 70,
        variance: 10,
        color: null,
      },
      {
        bucket_id: 'DIV',
        name: 'Dividend Shield',
        target_percent: 40,
        actual_value_thb: 300000,
        actual_percent: 30,
        variance: -10,
        color: null,
      },
    ]

    render(
      <PortfolioOverviewTab
        portfolioId="default"
        targets={targets}
        summaries={highVarSummaries}
        warning={null}
        onSelectBucket={vi.fn()}
      />
    )

    expect(screen.getByText(/แจ้งเตือนการปรับสมดุลพอร์ต/i)).toBeInTheDocument()
    expect(screen.getByText(/Overweight/i)).toBeInTheDocument()
    expect(screen.getByText(/Underweight/i)).toBeInTheDocument()
  })

  it('renders AI bucket auto-assign button with badge when unassigned holdings exist', () => {
    const holdings = [
      {
        symbol: 'NVDA',
        asset_type: 'Stock',
        units: 10,
        current_price_thb: 4000,
        avg_cost_thb: 3500,
        market_value_thb: 40000,
        unrealized_pl_thb: 5000,
        unrealized_pl_percent: 14.28,
        cost_basis_thb: 35000,
        weight_percent: 4,
        bucket_id: null, // Unassigned
      },
      {
        symbol: 'CASH',
        asset_type: 'Cash',
        units: 960000,
        current_price_thb: 1,
        avg_cost_thb: 1,
        market_value_thb: 960000,
        unrealized_pl_thb: 0,
        unrealized_pl_percent: 0,
        cost_basis_thb: 960000,
        weight_percent: 96,
        bucket_id: 'CORE',
      },
    ] as any

    render(
      <PortfolioOverviewTab
        portfolioId="default"
        targets={targets}
        summaries={summaries}
        warning={null}
        holdings={holdings}
        onSelectBucket={vi.fn()}
        onSuccess={vi.fn()}
      />
    )

    const aiBtn = screen.getByRole('button', { name: /จัดสินทรัพย์เข้า Bucket ด้วย AI/i })
    expect(aiBtn).toBeInTheDocument()
    // Badge should show 1 unassigned
    expect(screen.getByText('1')).toBeInTheDocument()

    // Clicking opens the modal
    fireEvent.click(aiBtn)
    expect(screen.getByText('จัดกลุ่มสินทรัพย์อัตโนมัติด้วย AI')).toBeInTheDocument()
  })

  it('renders skeleton shimmer when loading is true and summaries are empty', () => {
    const { container } = render(
      <PortfolioOverviewTab
        portfolioId="default"
        targets={[]}
        summaries={[]}
        warning={null}
        loading={true}
        onSelectBucket={vi.fn()}
      />
    )
    expect(container.querySelectorAll('.animate-pulse').length).toBeGreaterThanOrEqual(1)
    expect(screen.queryByText(/ยังไม่มีข้อมูลสัดส่วนพอร์ต/i)).not.toBeInTheDocument()
  })
})

