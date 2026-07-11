import { describe, expect, it } from 'vitest'
import type { MacroDashboardDTO, MacroIndicatorDTO } from '../api/types'
import { enrichIndicator, enrichSourceFile, extractIndicatorValue } from './macroReferences'

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

function makeIndicator(overrides: Partial<MacroIndicatorDTO> = {}): MacroIndicatorDTO {
  return {
    indicator_id: 'obs_us10y',
    series_key: 'us10y',
    label: 'US 10Y Treasury Yield',
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

describe('enrichIndicator', () => {
  it('id ที่อยู่ในตาราง KNOWN_INDICATORS ได้ชื่อ/หมวดเต็ม', () => {
    const meta = enrichIndicator('obs_us10y')
    expect(meta.name).toContain('US 10-Year Treasury Yield')
    expect(meta.category).toBe('Rates & Liquidity')
  })

  it('id ไม่รู้จักแต่เข้า pattern (vix/hyg/oil) ได้หมวดถูกต้อง', () => {
    expect(enrichIndicator('obs_something_vix_20990101').category).toBe('Volatility')
    expect(enrichIndicator('obs_ratio_hyg_x').category).toBe('Credit')
    expect(enrichIndicator('obs_crude_price').category).toBe('Commodities & Energy')
  })

  it('id ไม่เข้า pattern ไหนเลย → ชื่อ generic จาก id ไม่ crash', () => {
    const meta = enrichIndicator('obs_mystery_metric')
    expect(meta.name).toBe('MYSTERY METRIC')
    expect(meta.category).toBe('Macro & Equities')
  })

  it('ข้อมูลจาก dashboard_indicators (backend DTO) ชนะตาราง hardcode เสมอ', () => {
    const dashboard = makeDashboard({
      dashboard_indicators: [
        makeIndicator({ indicator_id: 'obs_us10y', label: 'ชื่อจริงจาก backend', display_value: '4.55%', provider: 'FRED Live' }),
      ],
    })
    const meta = enrichIndicator('obs_us10y', dashboard)
    expect(meta.name).toBe('ชื่อจริงจาก backend')
    expect(meta.extractedValue).toBe('4.55%')
    expect(meta.sourceProvider).toBe('FRED Live')
  })

  it('id ที่ไม่อยู่ใน dashboard_indicators ยัง fallback ไป pattern ตามเดิม', () => {
    const dashboard = makeDashboard({ dashboard_indicators: [makeIndicator()] })
    const meta = enrichIndicator('obs_global_vix_x', dashboard)
    expect(meta.category).toBe('Volatility')
  })
})

describe('extractIndicatorValue', () => {
  it('ดึงค่า VIX จากข้อความ evidence', () => {
    const dashboard = makeDashboard({
      regime_evidence: [
        { dimension: 'Volatility', signal: 'calm', evidence: 'VIX Index = 15.2 ต่ำกว่าค่าเฉลี่ย', conflict: '', confidence: 'high' },
      ],
    })
    expect(extractIndicatorValue('obs_vix', dashboard)).toBe('15.2 pts')
  })

  it('ดึงค่า HYG/LQD จาก supporting_data ของ asset allocation', () => {
    const dashboard = makeDashboard({
      asset_allocation: [
        {
          asset_class: 'Credit',
          asset_bucket: null,
          stance: 'Neutral',
          confidence: 'medium',
          rationale: '',
          supporting_data: ['HYG/LQD Ratio = 0.87 ทรงตัว'],
          why_not_high: '',
          allocation_delta: '',
          invalidation_conditions: [],
          warnings: [],
        },
      ],
    })
    expect(extractIndicatorValue('obs_ratio_hyg_lqd', dashboard)).toBe('0.87')
  })

  it('ไม่มีข้อความไหน match → คืน null', () => {
    expect(extractIndicatorValue('obs_vix', makeDashboard())).toBeNull()
  })
})

describe('enrichSourceFile', () => {
  it('valuation.py ได้ title/คำอธิบายเฉพาะ', () => {
    const meta = enrichSourceFile('valuation.py')
    expect(meta.type).toBe('quant_engine')
    expect(meta.title).toContain('Valuation Engine')
  })

  it('ไฟล์ .py ที่มีคำว่า regime ได้ title โมเดล regime', () => {
    const meta = enrichSourceFile('regime_model.py')
    expect(meta.type).toBe('quant_engine')
    expect(meta.title).toContain('Regime Probability Engine')
  })

  it('ไฟล์ .py อื่นๆ ยังเป็น quant_engine พร้อมคำอธิบาย generic', () => {
    const meta = enrichSourceFile('mystery_tool.py')
    expect(meta.type).toBe('quant_engine')
    expect(meta.filename).toBe('mystery_tool.py')
  })

  it('ไฟล์ .md จับคู่ snapshot ประเภท Global/Country ถูกต้อง', () => {
    expect(enrichSourceFile('Global_Macro_Snapshot_2026-07-09.md').title).toContain('Global Macro Snapshot')
    expect(enrichSourceFile('Country_Macro_Snapshot_TH.md').title).toContain('Country Macro Snapshot')
  })

  it('ไฟล์ .md อื่นแปลง underscore เป็นช่องว่างและตัดนามสกุล', () => {
    const meta = enrichSourceFile('Weekly_Notes.md')
    expect(meta.type).toBe('report')
    expect(meta.title).toBe('Weekly Notes')
  })
})
