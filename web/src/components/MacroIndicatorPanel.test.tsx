import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from '../api/client'
import type { MacroIndicatorDTO, MacroIndicatorSeriesDTO } from '../api/types'
import MacroIndicatorPanel from './MacroIndicatorPanel'

function makeIndicator(overrides: Partial<MacroIndicatorDTO> = {}): MacroIndicatorDTO {
  return {
    indicator_id: 'obs_us10y',
    series_key: 'us10y',
    label: 'US 10Y Yield',
    value: 4.2,
    display_value: '4.20%',
    unit: '%',
    observed_at: '2026-07-10',
    provider: 'FRED',
    source_file: 'Global_Macro_Snapshot.md',
    is_valid: true,
    stale_reason: '',
    chart_available: true,
    ...overrides,
  }
}

function makeSeries(pointCount: number): MacroIndicatorSeriesDTO {
  return {
    indicator_id: 'obs_us10y',
    series_key: 'us10y',
    label: 'US 10Y Yield',
    unit: '%',
    range: '3m',
    points: Array.from({ length: pointCount }, (_, i) => ({
      observed_at: `2026-07-0${i + 1}`,
      value: 4 + i * 0.1,
    })),
  }
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('MacroIndicatorPanel', () => {
  it('indicators ว่าง → ไม่ render อะไรเลย', () => {
    const { container } = render(<MacroIndicatorPanel indicators={[]} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('โหลด series ด้วย range เริ่มต้น 3m แล้ว render กราฟเมื่อมี ≥2 จุด', async () => {
    const spy = vi.spyOn(api, 'getMacroIndicatorSeries').mockResolvedValue(makeSeries(5))
    render(<MacroIndicatorPanel indicators={[makeIndicator()]} />)

    expect(spy).toHaveBeenCalledWith('obs_us10y', '3m')
    expect(await screen.findByRole('img', { name: 'กราฟ US 10Y Yield' })).toBeInTheDocument()
  })

  it('เปลี่ยน range → refetch ด้วย range ใหม่ + aria-pressed ย้ายตาม', async () => {
    const spy = vi.spyOn(api, 'getMacroIndicatorSeries').mockResolvedValue(makeSeries(3))
    render(<MacroIndicatorPanel indicators={[makeIndicator()]} />)
    await screen.findByRole('img', { name: /กราฟ/ })

    await userEvent.click(screen.getByRole('button', { name: '1M' }))
    expect(spy).toHaveBeenLastCalledWith('obs_us10y', '1m')
    expect(screen.getByRole('button', { name: '1M' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: '3M' })).toHaveAttribute('aria-pressed', 'false')
  })

  it('chart_available=false → ไม่ fetch และแสดงข้อความไม่มีข้อมูลกราฟ', () => {
    const spy = vi.spyOn(api, 'getMacroIndicatorSeries').mockResolvedValue(makeSeries(0))
    render(<MacroIndicatorPanel indicators={[makeIndicator({ chart_available: false })]} />)

    expect(spy).not.toHaveBeenCalled()
    expect(screen.getByText(/ยังไม่มีข้อมูลเชิงตัวเลขสำหรับสร้างกราฟ/)).toBeInTheDocument()
  })

  it('series มีจุดเดียว → บอกว่ารอ snapshot ครบ 2 จุดก่อน', async () => {
    vi.spyOn(api, 'getMacroIndicatorSeries').mockResolvedValue(makeSeries(1))
    render(<MacroIndicatorPanel indicators={[makeIndicator()]} />)

    expect(await screen.findByText(/อย่างน้อย 2 จุด/)).toBeInTheDocument()
  })

  it('โหลด series ล้มเหลว → แสดง error message', async () => {
    vi.spyOn(api, 'getMacroIndicatorSeries').mockRejectedValue(new Error('network'))
    render(<MacroIndicatorPanel indicators={[makeIndicator()]} />)

    expect(await screen.findByText('ไม่สามารถโหลดข้อมูลกราฟได้')).toBeInTheDocument()
  })

  it('เปลี่ยน indicator ผ่าน dropdown → fetch ตัวใหม่', async () => {
    const spy = vi.spyOn(api, 'getMacroIndicatorSeries').mockResolvedValue(makeSeries(3))
    render(
      <MacroIndicatorPanel
        indicators={[
          makeIndicator(),
          makeIndicator({ indicator_id: 'obs_vix', series_key: 'vix', label: 'VIX Index' }),
        ]}
      />
    )
    await screen.findByRole('img', { name: /กราฟ/ })

    await userEvent.selectOptions(screen.getByLabelText(/เลือกตัวชี้วัด/), 'obs_vix')
    expect(spy).toHaveBeenLastCalledWith('obs_vix', '3m')
  })
})
