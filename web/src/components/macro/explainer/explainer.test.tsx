import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ContextTooltip } from './ContextTooltip'
import { TrafficLightSubtitle } from './TrafficLightSubtitle'
import { SafetyShieldBanner } from './SafetyShieldBanner'
import { IllustrativeAllocationPie } from './IllustrativeAllocationPie'
import { MacroGlossaryDrawer, GLOSSARY_TERMS } from './MacroGlossaryDrawer'

describe('Explainer UI Components', () => {
  describe('ContextTooltip', () => {
    it('renders trigger button and toggles content on click', async () => {
      const user = userEvent.setup()
      render(
        <ContextTooltip
          title="Yield Curve"
          whatIsIt="กราฟเปรียบเทียบผลตอบแทนพันธบัตร"
          howToRead="ดูว่าชันหรือกลับหัว"
          impactOnMoney="บอกว่าควรระวังหุ้นหรือไม่"
        />
      )

      // Initial state: tooltip popup not shown
      expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()

      // Click trigger
      const button = screen.getByRole('button', { name: /คำอธิบาย: Yield Curve/ })
      await user.click(button)

      // Tooltip is now visible with 3-bullet points
      expect(screen.getByRole('tooltip')).toBeInTheDocument()
      expect(screen.getByText('Yield Curve')).toBeInTheDocument()
      expect(screen.getByText('กราฟเปรียบเทียบผลตอบแทนพันธบัตร')).toBeInTheDocument()
      expect(screen.getByText('ดูว่าชันหรือกลับหัว')).toBeInTheDocument()
      expect(screen.getByText('บอกว่าควรระวังหุ้นหรือไม่')).toBeInTheDocument()

      // Click close button inside tooltip
      const closeBtn = screen.getByText('✕')
      await user.click(closeBtn)
      expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
    })
  })

  describe('TrafficLightSubtitle', () => {
    it('renders subtitle with correct status badge styling', () => {
      render(
        <TrafficLightSubtitle
          status="positive"
          text="เส้นชันปกติ: ตลาดยังไม่เห็นสัญญาณวิกฤต"
        />
      )
      expect(screen.getByText('เส้นชันปกติ: ตลาดยังไม่เห็นสัญญาณวิกฤต')).toBeInTheDocument()
      expect(screen.getByText('🟢')).toBeInTheDocument()
    })

    it('renders warning status', () => {
      render(
        <TrafficLightSubtitle
          status="warning"
          text="Valuation แพงเกินไป"
        />
      )
      expect(screen.getByText('Valuation แพงเกินไป')).toBeInTheDocument()
      expect(screen.getByText('⚠️')).toBeInTheDocument()
    })
  })

  describe('SafetyShieldBanner', () => {
    it('renders TAA vs SAA shield banner', () => {
      render(<SafetyShieldBanner type="taa" />)
      expect(screen.getByText(/Tactical Asset Allocation/i)).toBeInTheDocument()
      expect(screen.getByText(/ไม่ใช่การเทขายล้างพอร์ตหลักระยะยาว/i)).toBeInTheDocument()
    })

    it('renders FX Hedge shield banner', () => {
      render(<SafetyShieldBanner type="fx_hedge" />)
      expect(screen.getByText(/FX Hedging/i)).toBeInTheDocument()
      expect(screen.getByText(/Hedge FX/i)).toBeInTheDocument()
    })

    it('renders Invalidation Trigger banner', () => {
      render(<SafetyShieldBanner type="invalidation" />)
      expect(screen.getByText(/Invalidation Triggers/i)).toBeInTheDocument()
      expect(screen.getByText(/เบรกเกอร์ตัดไฟ/i)).toBeInTheDocument()
    })
  })

  describe('IllustrativeAllocationPie', () => {
    it('calculates total 100 THB and renders asset split chips', () => {
      render(
        <IllustrativeAllocationPie
          allocations={[
            { asset_class: 'US Equities', stance: 'underweight', allocation_delta: '-5%' },
            { asset_class: 'SET Equities', stance: 'overweight', allocation_delta: '+3%' },
            { asset_class: 'US Treasuries', stance: 'overweight', allocation_delta: '+2%' },
            { asset_class: 'Gold', stance: 'neutral', allocation_delta: '0%' },
          ]}
        />
      )

      expect(screen.getByText(/ตัวอย่างเงิน 100 บาท/i)).toBeInTheDocument()
      expect(screen.getByText(/หุ้นสหรัฐฯ/i)).toBeInTheDocument()
      expect(screen.getByText(/หุ้นไทย/i)).toBeInTheDocument()
      expect(screen.getByText(/พันธบัตรรัฐบาล/i)).toBeInTheDocument()
      expect(screen.getByText(/ทองคำ/i)).toBeInTheDocument()
      expect(screen.getByText(/เงินสด \/ กองทุนตลาดเงิน/i)).toBeInTheDocument()
    })
  })

  describe('MacroGlossaryDrawer', () => {
    it('returns null when isOpen is false', () => {
      const { container } = render(
        <MacroGlossaryDrawer isOpen={false} onClose={() => {}} />
      )
      expect(container).toBeEmptyDOMElement()
    })

    it('renders all 25 terms and filters by search text', async () => {
      const user = userEvent.setup()
      const onClose = vi.fn()
      render(<MacroGlossaryDrawer isOpen={true} onClose={onClose} />)

      // Header is rendered
      expect(screen.getByText('คู่มือคำศัพท์ Macro ภาษาคน')).toBeInTheDocument()
      expect(screen.getByText(`แสดง ${GLOSSARY_TERMS.length} จาก ${GLOSSARY_TERMS.length} คำศัพท์`)).toBeInTheDocument()

      // Search for ERP
      const searchInput = screen.getByPlaceholderText(/ค้นหาคำศัพท์/i)
      await user.type(searchInput, 'ERP')

      expect(screen.getByText('Equity Risk Premium (ERP)')).toBeInTheDocument()
      expect(screen.queryByText('Goldilocks Economy')).not.toBeInTheDocument()

      // Close button
      const closeButtons = screen.getAllByRole('button', { name: /ปิด/i })
      expect(closeButtons.length).toBeGreaterThan(0)
      await user.click(closeButtons[0]!)
      expect(onClose).toHaveBeenCalled()
    })
  })
})
