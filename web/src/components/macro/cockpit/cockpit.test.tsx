import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi } from 'vitest'
import { SourceProvenanceBadge } from './SourceProvenanceBadge'
import { MacroRegionTabs } from './MacroRegionTabs'
import { YieldCurveChart } from './YieldCurveChart'
import { OfrStressBar } from './OfrStressBar'
import { DivergingFlowBar } from './DivergingFlowBar'
import { BreadthStackedBar } from './BreadthStackedBar'
import { FedBotRatePair } from './FedBotRatePair'
import { ThailandMacroSection } from './ThailandMacroSection'

describe('Macro Cockpit Components', () => {
  it('renders SourceProvenanceBadge for all 4 origins', () => {
    const { rerender } = render(
      <SourceProvenanceBadge origin="provider" sourceName="Settrade" observedAt="2026-09-26" />
    )
    expect(screen.getByText('ข้อมูลจากผู้ให้บริการ')).toBeInTheDocument()
    expect(screen.getByText(/Settrade/)).toBeInTheDocument()
    expect(screen.getByText('2026-09-26')).toBeInTheDocument()

    rerender(<SourceProvenanceBadge origin="deterministic" sourceName="Python Math" />)
    expect(screen.getByText('ระบบคำนวณ')).toBeInTheDocument()

    rerender(<SourceProvenanceBadge origin="ai" evaluatedAt="2026-09-27" />)
    expect(screen.getByText('AI วิเคราะห์')).toBeInTheDocument()
    expect(screen.getByText('2026-09-27')).toBeInTheDocument()

    rerender(<SourceProvenanceBadge origin="external" sourceName="TradingView" />)
    expect(screen.getByText('กราฟภายนอก')).toBeInTheDocument()
  })

  it('handles MacroRegionTabs tab switching', async () => {
    const onChange = vi.fn()
    render(<MacroRegionTabs activeTab="us" onChange={onChange} />)

    const aiBtn = screen.getByRole('button', { name: /บทวิเคราะห์ AI/i })
    await userEvent.click(aiBtn)
    expect(onChange).toHaveBeenCalledWith('ai')

    const thBtn = screen.getByRole('button', { name: /ประเทศไทย \(TH\)/i })
    await userEvent.click(thBtn)
    expect(onChange).toHaveBeenCalledWith('th')

    const crossBtn = screen.getByRole('button', { name: /ความเชื่อมโยง US–TH/i })
    await userEvent.click(crossBtn)
    expect(onChange).toHaveBeenCalledWith('cross-border')
  })

  it('renders YieldCurveChart with normal and inverted warnings', () => {
    const normalData = {
      observation_date: '2026-09-26',
      yields: [
        { maturity: '2 Yr', yield_percent: 4.1 },
        { maturity: '10 Yr', yield_percent: 4.3 },
      ],
      spread_10y_2y_bps: 20,
      fetched_at: 1727337600,
      source: 'US Treasury',
      unit: '%',
      is_stale: false,
    }

    const { rerender } = render(<YieldCurveChart data={normalData} />)
    expect(screen.getByText('US Treasury Yield Curve (Cross-Section)')).toBeInTheDocument()
    expect(screen.getByText('NORMAL (ชันปกติ)')).toBeInTheDocument()
    expect(screen.getByText('+20 bps')).toBeInTheDocument()

    const invertedData = {
      ...normalData,
      spread_10y_2y_bps: -45,
    }
    rerender(<YieldCurveChart data={invertedData} />)
    expect(screen.getByText('INVERTED (กลับหัว)')).toBeInTheDocument()
    expect(screen.getByText('-45 bps')).toBeInTheDocument()
  })

  it('renders OfrStressBar with 5 categories around zero baseline', () => {
    render(
      <OfrStressBar
        fsiValue={-1.5}
        asOfDate="2026-09-24"
        regime="Below Average Stress"
        categories={[
          { label: 'Credit', value: -0.8 },
          { label: 'Funding', value: 0.3 },
        ]}
      />
    )
    expect(screen.getByText('US Financial Stress Index (OFR FSI)')).toBeInTheDocument()
    expect(screen.getByText('-1.50')).toBeInTheDocument()
    expect(screen.getByText('Credit')).toBeInTheDocument()
    expect(screen.getByText('Funding')).toBeInTheDocument()
    expect(screen.getByText('+0.30 σ')).toBeInTheDocument()
  })

  it('renders DivergingFlowBar without 100% share distortion', () => {
    const flowData = {
      market: 'SET',
      as_of: '2026-09-26',
      total_value: 40000000000,
      investors: [
        { investor_type: 'Foreign Investors', buy_value: 15000000000, sell_value: 18000000000, net_value: -3000000000 },
        { investor_type: 'Retail Investors', buy_value: 12000000000, sell_value: 10000000000, net_value: 2000000000 },
      ],
      source: 'Settrade',
      is_stale: false,
    }

    render(<DivergingFlowBar flow={flowData} />)
    expect(screen.getByText('SET 4-Investor Flow (Diverging Net Flow)')).toBeInTheDocument()
    expect(screen.getByText('Foreign Investors')).toBeInTheDocument()
    expect(screen.getByText('-3,000.0 ล้านบาท')).toBeInTheDocument()
    expect(screen.getByText('+2,000.0 ล้านบาท')).toBeInTheDocument()
  })

  it('renders BreadthStackedBar with 100% distribution', () => {
    const breadthData = {
      market: 'SET',
      as_of: '2026-09-26',
      gainers: 300,
      losers: 150,
      unchanged: 50,
      source: 'Settrade',
      is_stale: false,
    }

    render(<BreadthStackedBar breadth={breadthData} />)
    expect(screen.getByText('SET Market Breadth (100% Stacked Distribution)')).toBeInTheDocument()
    expect(screen.getByText('2.00x')).toBeInTheDocument()
    expect(screen.getByText('Bullish Breadth')).toBeInTheDocument()
    expect(screen.getByText('300')).toBeInTheDocument()
    expect(screen.getByText('150')).toBeInTheDocument()
  })

  it('renders FedBotRatePair without recalculating spread in UI', () => {
    const ratesData = {
      as_of_date: '2026-09-26',
      rates: [
        { country: 'US', rate_value: 5.25, rate_type: 'Fed Funds Target Range' },
        { country: 'TH', rate_value: 2.25, rate_type: 'BOT 1-Day Repo' },
      ],
      spreads_vs_bot_repo: { US: 300 },
    }

    render(<FedBotRatePair ratesData={ratesData} />)
    expect(screen.getByText('Fed Funds vs BoT 1D Repo (Policy Rate Spread)')).toBeInTheDocument()
    expect(screen.getByText('5.25%')).toBeInTheDocument()
    expect(screen.getByText('2.25%')).toBeInTheDocument()
    expect(screen.getByText('+300 bps')).toBeInTheDocument()
  })

  it('renders ThailandMacroSection with Thai Market Observables and Microstructure metrics', () => {
    const mockAiData: any = {
      evaluated_at: '2026-09-27T15:30:00Z',
      overall_regime: 'Reflation',
      time_horizon: '3-6 Months',
      thailand_market_stance: {
        investor_flow: { foreign_net_mb: -3386.63 },
        market_breadth: { advance_decline_ratio: 1.08, sentiment: 'neutral' },
        valuation: { pe_ratio: 16.0 },
        physical_gold: { bar_sell_thb: 67850 },
        policy_spread_bps: 275.0,
      },
      asset_allocation: [],
      warnings: [],
    }

    render(
      <ThailandMacroSection
        flow={null}
        gold={null}
        valuation={null}
        breadth={null}
        aiData={mockAiData}
      />
    )

    // Verify Pure Market Observables Header
    expect(screen.getByText(/สภาวะตลาดทุนและสภาพคล่องไทย \(Thai Market Microstructure\)/)).toBeInTheDocument()
    // Verify Microstructure metric labels and values
    expect(screen.getByText('Foreign Net Flow')).toBeInTheDocument()
    expect(screen.getByText('-3,386.63 ลบ.')).toBeInTheDocument()
    expect(screen.getByText('ต่างชาติขายสุทธิ')).toBeInTheDocument()
    expect(screen.getByText('1.08x')).toBeInTheDocument()
    expect(screen.getByText('16.00x')).toBeInTheDocument()
    expect(screen.getByText('67,850 ฿')).toBeInTheDocument()
    // Verify spread formatted properly
    expect(screen.getByText('+275 bps')).toBeInTheDocument()
    // Verify that AI narrative is NOT in ThailandMacroSection
    expect(screen.queryByText(/AI วิเคราะห์สภาวะตลาดทุนและค่าเงินบาท/)).not.toBeInTheDocument()
  })

  it('renders ThailandMacroSection without policy spread displaying fallback text and never +275 bps', () => {
    const mockAiDataWithoutSpread: any = {
      evaluated_at: '2026-09-27T15:30:00Z',
      overall_regime: 'Reflation',
      thailand_market_stance: {
        investor_flow: { foreign_net_mb: -100.0 },
        policy_spread_bps: null, // missing/null rate spread
      },
      asset_allocation: [],
      warnings: [],
    }

    render(
      <ThailandMacroSection
        flow={null}
        gold={null}
        valuation={null}
        breadth={null}
        aiData={mockAiDataWithoutSpread}
      />
    )

    expect(screen.getByText('ไม่มีข้อมูลส่วนต่างดอกเบี้ย')).toBeInTheDocument()
    expect(screen.queryByText('+275 bps')).not.toBeInTheDocument()
  })

  it('renders ThailandMacroSection with negative policy spread formatted with minus sign', () => {
    const mockAiDataNegSpread: any = {
      evaluated_at: '2026-09-27T15:30:00Z',
      overall_regime: 'Reflation',
      thailand_market_stance: {
        policy_spread_bps: -50.0,
      },
      asset_allocation: [],
      warnings: [],
    }

    render(
      <ThailandMacroSection
        flow={null}
        gold={null}
        valuation={null}
        breadth={null}
        aiData={mockAiDataNegSpread}
      />
    )

    expect(screen.getByText('−50 bps')).toBeInTheDocument()
    expect(screen.queryByText('+275 bps')).not.toBeInTheDocument()
  })

  it('renders ThailandMacroSection with Thai Sovereign Fiscal Health metrics from MOF', () => {
    const mockAiDataFiscal: any = {
      evaluated_at: '2026-10-03T10:00:00Z',
      overall_regime: 'Unknown',
      regional_assessments: {
        Thailand: {
          state: 'Unknown',
          confidence: 0.0,
          data_gaps: [
            'Thailand Growth (สศช. Real GDP)',
            'Thailand Inflation (สนค. Headline CPI)',
            'Thailand Monetary: อัตราดอกเบี้ยนโยบายพร้อมใช้งาน (2.50%) แต่ขาด Headline CPI เพื่อคำนวณอัตราดอกเบี้ยจริง (Real Policy Rate)',
          ],
          fiscal_health: {
            debt_to_gdp_pct: 63.85,
            public_debt_million_thb: 11850000,
            statutory_limit_pct: 70.0,
            status: 'within_ceiling',
          },
        },
      },
    }

    render(
      <ThailandMacroSection
        flow={null}
        gold={null}
        valuation={null}
        breadth={null}
        aiData={mockAiDataFiscal}
      />
    )

    // Fiscal Health panel & metrics
    expect(screen.getByText(/ความยั่งยืนทางการคลังและหนี้สาธารณะ \(Thai Sovereign Fiscal Health\)/)).toBeInTheDocument()
    expect(screen.getByText('63.85%')).toBeInTheDocument()
    expect(screen.getByText('11.85 ล้านล้านบาท')).toBeInTheDocument()
    expect(screen.getByText(/ปกติ \(ต่ำกว่าเพดาน 70%\)/)).toBeInTheDocument()

    // Dynamic structured gap reasons
    expect(screen.getByText('สศช. (NESDC)')).toBeInTheDocument()
    expect(screen.getByText('สนค. พาณิชย์ (TPSO/MOC)')).toBeInTheDocument()
    expect(screen.getByText('ธนาคารแห่งประเทศไทย (BOT)')).toBeInTheDocument()
    expect(screen.getByText('ดอกเบี้ยนโยบายพร้อม แต่ขาด Headline CPI เพื่อคำนวณ Real Rate')).toBeInTheDocument()
  })
})

