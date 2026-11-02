import React from 'react'

export interface AllocationItem {
  asset_class: string
  stance: string
  allocation_delta?: string | null
  asset_bucket?: string | null
}

export interface IllustrativeAllocationPieProps {
  allocations?: AllocationItem[]
  className?: string
}

interface ComputedSlice {
  label: string
  amount: number
  color: string
  textColor: string
  bgLight: string
  border: string
  icon: string
  note: string
}

export const IllustrativeAllocationPie: React.FC<IllustrativeAllocationPieProps> = ({
  allocations = [],
  className = '',
}) => {
  // Compute illustrative 100 THB split based on stances
  // Base split:
  // US Equities: 30
  // Thai Equities: 15
  // Fixed Income: 20
  // Gold/Commodities: 15
  // Cash/Liquidity: 20
  let usEq = 30
  let thEq = 15
  let fi = 20
  let gold = 15
  let cash = 20

  allocations.forEach((a) => {
    const name = (a.asset_class || '').toLowerCase()
    const stance = (a.stance || '').toLowerCase()
    const delta = a.allocation_delta || ''

    let shift = 0
    if (delta.includes('+') || stance === 'overweight') shift = 3
    else if (delta.includes('-') || stance === 'underweight') shift = -3

    if (name.includes('us equities') || name.includes('s&p')) {
      usEq = Math.max(10, Math.min(50, usEq + shift))
    } else if (name.includes('set') || name.includes('thai')) {
      thEq = Math.max(5, Math.min(30, thEq + shift))
    } else if (name.includes('treasur') || name.includes('bond') || name.includes('fixed')) {
      fi = Math.max(10, Math.min(35, fi + shift))
    } else if (name.includes('gold')) {
      gold = Math.max(5, Math.min(25, gold + shift))
    }
  })

  // Balance with cash so sum is exactly 100
  cash = 100 - (usEq + thEq + fi + gold)
  if (cash < 10) {
    const diff = 10 - cash
    usEq -= Math.ceil(diff / 2)
    fi -= Math.floor(diff / 2)
    cash = 10
  }

  const slices: ComputedSlice[] = [
    {
      label: 'หุ้นสหรัฐฯ (S&P 500)',
      amount: usEq,
      color: 'bg-blue-500',
      textColor: 'text-blue-900',
      bgLight: 'bg-blue-50',
      border: 'border-blue-200',
      icon: '🇺🇸',
      note: 'ถือตัวใหญ่กำไรแกร่ง ไม่ไล่ราคา',
    },
    {
      label: 'หุ้นไทย (SET)',
      amount: thEq,
      color: 'bg-emerald-500',
      textColor: 'text-emerald-900',
      bgLight: 'bg-emerald-50',
      border: 'border-emerald-200',
      icon: '🇹🇭',
      note: 'Valuation ถูก ทยอยแบ่งสะสม',
    },
    {
      label: 'พันธบัตรรัฐบาล',
      amount: fi,
      color: 'bg-indigo-500',
      textColor: 'text-indigo-900',
      bgLight: 'bg-indigo-50',
      border: 'border-indigo-200',
      icon: '🏛️',
      note: 'รับผลตอบแทนดอกเบี้ยสูง',
    },
    {
      label: 'ทองคำ',
      amount: gold,
      color: 'bg-amber-500',
      textColor: 'text-amber-900',
      bgLight: 'bg-amber-50',
      border: 'border-amber-200',
      icon: '🪙',
      note: 'ประกันความเสี่ยงภูมิรัฐศาสตร์',
    },
    {
      label: 'เงินสด / กองทุนตลาดเงิน',
      amount: cash,
      color: 'bg-teal-500',
      textColor: 'text-teal-900',
      bgLight: 'bg-teal-50',
      border: 'border-teal-200',
      icon: '💵',
      note: 'สำรองสภาพคล่อง ดักช้อนจังหวะย่อ',
    },
  ]

  return (
    <div className={`rounded-xl border border-sky-100 bg-white/90 p-4 shadow-xs space-y-3 ${className}`}>
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1 border-b border-zinc-100 pb-2">
        <div className="flex items-center gap-1.5 font-bold text-zinc-900 text-xs sm:text-[13px]">
          <span>💰</span>
          <span>ตัวอย่างเงิน 100 บาท วันนี้ควรแบ่งยังไง (Illustrative 100-Baht Split)</span>
        </div>
        <span className="text-[10px] text-zinc-400 font-medium">
          คำนวณตาม Tactical Tilt ล่าสุดของ AI
        </span>
      </div>

      {/* Visual Multi-Segment Bar */}
      <div className="h-4 w-full rounded-full overflow-hidden flex bg-zinc-100 p-0.5">
        {slices.map((s, idx) => (
          <div
            key={idx}
            className={`h-full ${s.color} transition-all duration-300 first:rounded-l-full last:rounded-r-full`}
            style={{ width: `${s.amount}%` }}
            title={`${s.label}: ${s.amount} บาท (${s.amount}%)`}
          />
        ))}
      </div>

      {/* Chip Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2 pt-1">
        {slices.map((s, idx) => (
          <div
            key={idx}
            className={`rounded-lg border p-2 text-xs flex flex-col justify-between ${s.bgLight} ${s.border}`}
          >
            <div>
              <div className="flex items-center justify-between text-[11px] font-semibold text-zinc-700">
                <span className="truncate flex items-center gap-1">
                  <span>{s.icon}</span>
                  <span className="truncate">{s.label}</span>
                </span>
              </div>
              <div className="mt-1 text-base font-extrabold text-zinc-900">
                {s.amount} <span className="text-[11px] font-normal text-zinc-500">บาท</span>
              </div>
            </div>
            <p className="mt-1 text-[10px] text-zinc-500 line-clamp-1" title={s.note}>
              {s.note}
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}
