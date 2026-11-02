import React from 'react'

export type SafetyShieldType = 'taa' | 'fx_hedge' | 'invalidation'

export interface SafetyShieldBannerProps {
  type: SafetyShieldType
  customMessage?: string
  className?: string
}

const SHIELD_CONFIG: Record<SafetyShieldType, { title: string; defaultMessage: string; bg: string; border: string; text: string; icon: string }> = {
  taa: {
    title: 'คำแนะนำการปรับทัพตามรอบ (Tactical Asset Allocation)',
    defaultMessage: 'สัดส่วนและมุมมองในหน้านี้เป็นการปรับเอียงน้ำหนักพอร์ตตามรอบเศรษฐกิจ 3–6 เดือน สำหรับเงินก้อนใหม่หรือเงินส่วนที่มีความยืดหยุ่น ไม่ใช่การเทขายล้างพอร์ตหลักระยะยาว (Core Portfolio) ทิ้ง',
    bg: 'bg-indigo-50/70',
    border: 'border-indigo-200/80',
    text: 'text-indigo-950',
    icon: '🛡️',
  },
  fx_hedge: {
    title: 'คำแนะนำความเสี่ยงค่าเงินสำหรับคนไทย (FX Hedging)',
    defaultMessage: 'สำหรับคนไทยที่ลงทุนสินทรัพย์ต่างประเทศ ในช่วงที่ค่าเงินบาทผันผวน แนะนำเลือกลงทุนผ่านกองทุนที่มีนโยบายป้องกันความเสี่ยงอัตราแลกเปลี่ยน (Hedge FX) เพื่อไม่ให้กำไรจากหุ้นถูกค่าเงินบาทกลืนหายไป',
    bg: 'bg-teal-50/70',
    border: 'border-teal-200/80',
    text: 'text-teal-950',
    icon: '💵',
  },
  invalidation: {
    title: 'เบรกเกอร์ตัดไฟ (จุดที่ต้องยอมรับว่าสมมติฐานผิด / Invalidation Triggers)',
    defaultMessage: 'หาก US 10Y Yield พุ่งทะลุ 5.25% หรือ VIX Index พุ่งเกิน 25.0 จุด แปลว่าตลาดหลุดกรอบและเข้าสู่ภาวะตื่นตระหนก ให้ระงับการซื้อสินทรัพย์เสี่ยงทันทีและเพิ่มเงินสดสำรองเป็น 40%',
    bg: 'bg-rose-50/70',
    border: 'border-rose-200/80',
    text: 'text-rose-950',
    icon: '🚨',
  },
}

export const SafetyShieldBanner: React.FC<SafetyShieldBannerProps> = ({
  type,
  customMessage,
  className = '',
}) => {
  const cfg = SHIELD_CONFIG[type]
  const message = customMessage || cfg.defaultMessage

  return (
    <div
      className={`rounded-xl border p-3 text-xs leading-relaxed flex items-start gap-2.5 ${cfg.bg} ${cfg.border} ${cfg.text} ${className}`}
    >
      <span className="text-base shrink-0 mt-0.5">{cfg.icon}</span>
      <div className="space-y-0.5">
        <span className="font-bold text-[11px] uppercase tracking-wider block">{cfg.title}</span>
        <p className="text-[11px] leading-relaxed opacity-90">{message}</p>
      </div>
    </div>
  )
}

