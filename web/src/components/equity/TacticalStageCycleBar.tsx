import React from 'react'
import { ContextTooltip } from './explainer'

export interface TacticalStageCycleBarProps {
  currentStage?: string | null
  className?: string
}

export const TacticalStageCycleBar: React.FC<TacticalStageCycleBarProps> = ({
  currentStage = 'UNKNOWN',
  className = '',
}) => {
  const stage = (currentStage || 'UNKNOWN').toUpperCase()

  const isStage1 = stage.includes('STAGE_1') || stage.includes('ACCUMULATION')
  const isStage2 = stage.includes('STAGE_2') || stage.includes('MARKUP')
  const isStage3 = stage.includes('STAGE_3') || stage.includes('DISTRIBUTION')
  const isStage4 = stage.includes('STAGE_4') || stage.includes('MARKDOWN')

  const stages = [
    {
      num: 1,
      title: 'Stage 1: Accumulation',
      sub: 'ช่วงสะสมพลัง (สร้างฐานราก รายใหญ่แอบเก็บเงียบๆ)',
      action: 'เฝ้ารอ: ใส่ Watchlist รอดักตอนเบรกเอาท์ผ่านกรอบ',
      isActive: isStage1,
      activeColor: 'bg-sky-500 text-white border-sky-600 shadow-[0_0_12px_rgba(14,165,233,0.4)]',
      barColor: isStage1 ? 'bg-sky-500' : 'bg-zinc-200',
    },
    {
      num: 2,
      title: 'Stage 2: Markup',
      sub: 'ช่วงขาขึ้นจริง (ราคายืนเหนือ MA200 วิ่งรันเทรนด์แข็งแกร่ง)',
      action: 'สเตจที่ดีที่สุด: ทยอยสะสมเมื่อราคาย่อตัวใน Buy Zone',
      isActive: isStage2,
      activeColor: 'bg-emerald-600 text-white border-emerald-700 shadow-[0_0_12px_rgba(16,185,129,0.5)]',
      barColor: isStage2 ? 'bg-emerald-500' : 'bg-zinc-200',
    },
    {
      num: 3,
      title: 'Stage 3: Distribution',
      sub: 'ช่วงแจกจ่าย (รายใหญ่เริ่มปล่อยของ ราคากวัดแกว่งตึงตัว)',
      action: 'ระวังตัว: ทยอยล็อคกำไร ห้ามไล่ราคาเด็ดขาด',
      isActive: isStage3,
      activeColor: 'bg-amber-500 text-white border-amber-600 shadow-[0_0_12px_rgba(245,158,11,0.4)]',
      barColor: isStage3 ? 'bg-amber-500' : 'bg-zinc-200',
    },
    {
      num: 4,
      title: 'Stage 4: Markdown',
      sub: 'ช่วงขาลง (แรงเทขายต่อเนื่อง หลุดเส้น MA200 เป็นแนวโน้มลง)',
      action: 'อันตรายสูงสุด: ถือเงินสด ห้ามรับมีดหรือถัวเฉลี่ยเด็ดขาด',
      isActive: isStage4,
      activeColor: 'bg-rose-600 text-white border-rose-700 shadow-[0_0_12px_rgba(244,63,94,0.5)]',
      barColor: isStage4 ? 'bg-rose-500' : 'bg-zinc-200',
    },
  ]

  return (
    <div
      data-testid="tactical-stage-cycle-bar"
      className={`rounded-xl border border-edge/80 bg-surface/90 p-4 space-y-3.5 ${className}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-edge/50 pb-2.5">
        <div className="flex items-center gap-2">
          <span className="text-base">🔄</span>
          <span className="text-xs font-bold text-zinc-800 tracking-tight">
            Stan Weinstein 4-Stage Market Cycle Timeline
          </span>
          <ContextTooltip
            title="วัฏจักรราคา 4 สเตจ (Stan Weinstein 4-Stage Market Cycle)"
            whatIsIt="ทฤษฎีวิเคราะห์วงจรชีวิตราคาหุ้นของ Stan Weinstein ที่แบ่งพฤติกรรมราคาหุ้นออกเป็น 4 ช่วงตามแนวโน้มและเส้นค่าเฉลี่ย 200 วัน"
            howToRead="Stage 1: สะสมพลัง ➔ Stage 2: ขาขึ้นจริง (จังหวะซื้อและถือทำกำไร) ➔ Stage 3: กระจายของ (เตรียมขายทำกำไร) ➔ Stage 4: ขาลง (หลีกเลี่ยง ไม่ควรรับมีด)"
            impactOnMoney="ช่วยป้องกันการเข้าซื้อผิดจังหวะ โดยเฉพาะหุ้น Stage 4 ที่ราคาลงต่อเนื่องแม้จะดูเหมือนถูกก็ตาม"
          />
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[11px] text-zinc-400">Current Phase:</span>
          <span
            className={`px-2.5 py-0.5 text-xs font-bold rounded-full uppercase border ${
              isStage2
                ? 'bg-emerald-50 text-emerald-800 border-emerald-300'
                : isStage4
                ? 'bg-rose-50 text-rose-800 border-rose-300'
                : 'bg-sky-50 text-sky-800 border-sky-300'
            }`}
          >
            {stage.replace(/_/g, ' ')}
          </span>
        </div>
      </div>

      {/* 4-Step Cycle Visual Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
        {stages.map((st) => (
          <div
            key={st.num}
            data-testid={`stage-step-${st.num}`}
            className={`relative rounded-xl p-3 border transition-all ${
              st.isActive
                ? `${st.activeColor} ring-2 ring-offset-2 ring-offset-surface ring-current scale-[1.02]`
                : 'bg-surface border-edge/60 text-zinc-600 hover:border-zinc-300'
            }`}
          >
            <div className="flex items-center justify-between mb-1">
              <span
                className={`text-[10px] font-black uppercase tracking-wider px-1.5 py-0.2 rounded ${
                  st.isActive ? 'bg-black/20 text-white' : 'bg-zinc-100 text-zinc-500'
                }`}
              >
                Stage {st.num}
              </span>
              {st.isActive && (
                <span className="flex h-2 w-2 relative">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-white" />
                </span>
              )}
            </div>

            <div className={`text-xs font-bold ${st.isActive ? 'text-white' : 'text-zinc-900'} truncate`}>
              {st.title.split(': ')[1]}
            </div>

            <div className={`text-[10px] mt-0.5 leading-snug ${st.isActive ? 'text-white/90' : 'text-zinc-500'}`}>
              {st.sub}
            </div>

            <div
              className={`mt-2 pt-1.5 border-t text-[10px] font-semibold ${
                st.isActive
                  ? 'border-white/20 text-white font-bold'
                  : 'border-edge/40 text-zinc-400'
              }`}
            >
              👉 {st.action}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default TacticalStageCycleBar
