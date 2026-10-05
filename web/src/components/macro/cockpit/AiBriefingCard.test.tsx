import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi } from 'vitest'
import { AiBriefingCard } from './AiBriefingCard'
import type { MacroDashboardDTO } from '../../../api/types'

describe('AiBriefingCard', () => {
  const mockAiData: MacroDashboardDTO = {
    evaluated_at: new Date(Date.now() - 3600000).toISOString(), // 1 hr ago
    overall_regime: 'Stagflation',
    time_horizon: '3-6 Months',
    conviction_level: 'High',
    conviction_rationale:
      'เงินเฟ้อภาคบริการยังคงทรงตัวในระดับสูง ขณะที่ตลาดแรงงานเริ่มมีสัญญาณชะลอตัวอย่างเห็นได้ชัด',
    quant_narrative_alignment: 'Aligned (90%)',
    focus_themes: ['Yield Curve Steepener', 'Defensive Quality', 'Energy Squeeze'],
    asset_allocation: [
      {
        asset_class: 'US Short Treasuries',
        region: 'US',
        stance: 'Overweight',
        confidence: 'high',
        rationale: 'ผลตอบแทนหน้าตั๋วยังสูงและปลอดความเสี่ยง',
        warnings: [],
      },
      {
        asset_class: 'High-Beta Equities',
        region: 'Global',
        stance: 'Underweight',
        confidence: 'medium',
        rationale: 'ความเสี่ยง valuation ตึงตัว',
        warnings: [],
      },
    ],
    pair_trades: [],
    risk_scenarios: [],
    warnings: [
      { code: 'W01', message: 'จับตาราคาน้ำมันดิบ WTI ที่อาจดีดตัวทะลุ $85' },
    ],
    dashboard_indicators: [],
    report_references: [],
    evaluated_sources: [],
    source_files: [],
  } as unknown as MacroDashboardDTO

  it('renders loading skeleton when aiLoading is true', () => {
    const { container } = render(<AiBriefingCard aiData={null} aiLoading={true} />)
    expect(container.querySelector('.animate-pulse')).toBeInTheDocument()
  })

  it('renders invitation card when aiData is null', async () => {
    const onUpdate = vi.fn()
    render(<AiBriefingCard aiData={null} onUpdateMacro={onUpdate} />)
    expect(screen.getByText(/AI Executive Briefing/)).toBeInTheDocument()
    expect(screen.getByText(/ยังไม่มีรายงานบทวิเคราะห์/)).toBeInTheDocument()

    const btn = screen.getByRole('button', { name: /เริ่มวิเคราะห์ภาวะเศรษฐกิจ/i })
    await userEvent.click(btn)
    expect(onUpdate).toHaveBeenCalledTimes(1)
  })

  it('renders complete executive briefing with regime, stance chips, themes, and warnings', async () => {
    const onNavigate = vi.fn()
    const onUpdate = vi.fn()

    render(
      <AiBriefingCard
        aiData={mockAiData}
        onUpdateMacro={onUpdate}
        onNavigateToAiTab={onNavigate}
      />
    )

    // Regime and Conviction
    expect(screen.getByText(/สภาวะเศรษฐกิจหลัก: Stagflation/)).toBeInTheDocument()
    expect(screen.getByText(/Conviction: High/i)).toBeInTheDocument()
    expect(screen.getByText(/กรอบเวลา: 3-6 Months/)).toBeInTheDocument()

    // Themes
    expect(screen.getByText(/#Yield Curve Steepener/)).toBeInTheDocument()
    expect(screen.getByText(/#Defensive Quality/)).toBeInTheDocument()

    // Rationale
    expect(screen.getByText(/เงินเฟ้อภาคบริการยังคงทรงตัวในระดับสูง/)).toBeInTheDocument()

    // Stance chips
    expect(screen.getByText('US Short Treasuries')).toBeInTheDocument()
    expect(screen.getByText('High-Beta Equities')).toBeInTheDocument()

    // Top Warning
    expect(screen.getByText(/จับตาราคาน้ำมันดิบ WTI/)).toBeInTheDocument()

    // Navigation link
    const linkBtn = screen.getByRole('button', { name: /ดูพอร์ตเต็ม →/i })
    await userEvent.click(linkBtn)
    expect(onNavigate).toHaveBeenCalledTimes(1)

    // Update button
    const updateBtn = screen.getByRole('button', { name: /อัปเดตบทวิเคราะห์/i })
    await userEvent.click(updateBtn)
    expect(onUpdate).toHaveBeenCalledTimes(1)
  })
})
