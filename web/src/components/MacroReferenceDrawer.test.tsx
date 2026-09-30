import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { AssetAllocationDTO, MacroDashboardDTO } from '../api/types'
import MacroReferenceDrawer from './MacroReferenceDrawer'

function makeAllocation(overrides: Partial<AssetAllocationDTO> = {}): AssetAllocationDTO {
  return {
    asset_class: 'Equities',
    asset_bucket: null,
    stance: 'Neutral',
    confidence: 'medium',
    rationale: '',
    supporting_data: [],
    why_not_high: '',
    allocation_delta: '',
    invalidation_conditions: [],
    warnings: [],
    ...overrides,
  }
}

function makeDashboard(overrides: Partial<MacroDashboardDTO> = {}): MacroDashboardDTO {
  return {
    evaluated_at: '2026-07-10',
    overall_regime: 'Reflation',
    time_horizon: '3-6 เดือน',
    key_assumptions: [],
    regime_probabilities: {},
    regime_evidence: [],
    warnings: [],
    ...overrides,
  }
}

describe('MacroReferenceDrawer', () => {
  it('isOpen=false → ไม่ render', () => {
    const { container } = render(
      <MacroReferenceDrawer data={makeDashboard()} isOpen={false} onClose={() => {}} />
    )
    expect(container).toBeEmptyDOMElement()
  })

  it('รวม observable_refs จากทุกส่วนแบบ dedupe — id ซ้ำนับครั้งเดียว', () => {
    const data = makeDashboard({
      asset_allocation: [makeAllocation({ observable_refs: ['obs_vix', 'obs_us10y'] })],
      pair_trades: [],
      regime_evidence: [
        {
          dimension: 'Volatility',
          signal: 'calm',
          evidence: '',
          conflict: '',
          confidence: 'high',
          observable_refs: ['obs_vix'], // ซ้ำกับ allocation
        },
      ],
    })
    render(<MacroReferenceDrawer data={data} isOpen onClose={() => {}} />)
    expect(screen.getByRole('button', { name: /ตัวชี้วัดเศรษฐกิจ \(2\)/ })).toBeInTheDocument()
  })

  it('สลับแท็บไปรายงานและโมเดลคำนวณ — แยก .md เป็น report และ .py เป็น quant engine', async () => {
    const data = makeDashboard({ source_files: ['Global_Macro_Snapshot_2026.md', 'valuation.py'] })
    render(<MacroReferenceDrawer data={data} isOpen onClose={() => {}} />)

    await userEvent.click(screen.getByRole('button', { name: /รายงานและโมเดลคำนวณ \(2\)/ }))
    expect(screen.getByText(/Global Macro Snapshot/)).toBeInTheDocument()
    expect(screen.getByText(/Valuation Engine/)).toBeInTheDocument()
  })

  it('แท็บหลักฐานรายสินทรัพย์แสดง stance ของแต่ละ allocation', async () => {
    const data = makeDashboard({
      asset_allocation: [makeAllocation({ asset_class: 'Gold', stance: 'Overweight' })],
    })
    render(<MacroReferenceDrawer data={data} isOpen onClose={() => {}} />)

    await userEvent.click(screen.getByRole('button', { name: /หลักฐานอ้างอิงรายสินทรัพย์/ }))
    expect(screen.getByText('Gold')).toBeInTheDocument()
    expect(screen.getByText('Overweight')).toBeInTheDocument()
  })

  it('รวม observable_refs จาก thailand_market_stance ในการนับ dedupe', () => {
    const data = makeDashboard({
      asset_allocation: [makeAllocation({ observable_refs: ['obs_vix'] })],
      thailand_market_stance: {
        observable_refs: ['obs_th_pe_ratio', 'obs_vix'], // obs_vix ซ้ำ, obs_th_pe_ratio เพิ่มใหม่
      },
    })
    render(<MacroReferenceDrawer data={data} isOpen onClose={() => {}} />)
    expect(screen.getByRole('button', { name: /ตัวชี้วัดเศรษฐกิจ \(2\)/ })).toBeInTheDocument()
  })

  it('สลับแท็บไปทะเบียน Observables และกรองตามภูมิภาค', async () => {
    const data = makeDashboard({
      observable_registry: {
        obs_us10y: {
          observable_id: 'obs_us10y',
          canonical_name: 'US 10Y Yield',
          region: 'United States',
          provider: 'US Treasury',
          source_type: 'official',
          status: 'verified',
          value: 4.25,
          unit: '%',
          observed_at: '2026-07-10',
        },
        obs_th_pe: {
          observable_id: 'obs_th_pe',
          canonical_name: 'SET Trailing P/E',
          region: 'Thailand',
          provider: 'SET',
          source_type: 'official',
          status: 'verified',
          value: 15.2,
          unit: 'x',
          observed_at: '2026-07-10',
        },
      },
    })
    render(<MacroReferenceDrawer data={data} isOpen onClose={() => {}} />)

    await userEvent.click(screen.getByRole('button', { name: /ทะเบียน Observables \(2\)/ }))
    expect(screen.getByText('US 10Y Yield')).toBeInTheDocument()
    expect(screen.getByText('SET Trailing P/E')).toBeInTheDocument()

    // Filter to Thailand
    await userEvent.click(screen.getByRole('button', { name: 'Thailand' }))
    expect(screen.queryByText('US 10Y Yield')).not.toBeInTheDocument()
    expect(screen.getByText('SET Trailing P/E')).toBeInTheDocument()
  })

  it('กด Escape → เรียก onClose (ผ่าน useFocusTrap)', async () => {
    const onClose = vi.fn()
    render(<MacroReferenceDrawer data={makeDashboard()} isOpen onClose={onClose} />)
    await userEvent.keyboard('{Escape}')
    expect(onClose).toHaveBeenCalledOnce()
  })

  it('เปิดแล้วโฟกัสถูกดึงเข้า drawer (ปุ่มแรกใน dialog)', () => {
    render(<MacroReferenceDrawer data={makeDashboard()} isOpen onClose={() => {}} />)
    const dialog = screen.getByRole('dialog')
    expect(dialog.contains(document.activeElement)).toBe(true)
  })
})
